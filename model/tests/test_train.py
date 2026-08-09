"""Tests for smartfour.train — replay buffer, checkpointing, best-model, loop."""

import math

import pytest
import torch

from smartfour.arena import play_arena
from smartfour.config import Config, MCTSConfig, NetworkConfig, TrainingConfig
from smartfour.encode import action_mask, apply_d4, apply_d4_policy, d4_perms, encode
from smartfour.game import BLACK, WHITE, apply_move, initial_state, legal_moves
from smartfour.network import ResNet, loss_fn
from smartfour.train import ReplayBuffer, Trainer


def tiny_training(**kw):
    kw.setdefault("selfplay_games", 3)
    kw.setdefault("train_epochs", 1)
    kw.setdefault("batch_size", 16)
    kw.setdefault("replay_capacity", 10_000)
    kw.setdefault("learning_rate", 0.001)
    kw.setdefault("weight_decay", 0.0)
    kw.setdefault("symmetry_augment", True)
    kw.setdefault("eval_games", 2)
    kw.setdefault("eval_simulations", 10)
    kw.setdefault("arena_win_ratio", 0.55)
    kw.setdefault("seed", 0)
    return TrainingConfig(**kw)


def tiny_mcts(**kw):
    kw.setdefault("simulations", 20)
    kw.setdefault("c_puct", 1.0)
    kw.setdefault("dirichlet_alpha", 0.3)
    kw.setdefault("dirichlet_epsilon", 0.25)
    kw.setdefault("temperature_threshold", 12)
    kw.setdefault("batch_eval_size", 16)
    return MCTSConfig(**kw)


def tiny_net_cfg(**kw):
    kw.setdefault("input_channels", 16)
    kw.setdefault("blocks", 1)
    kw.setdefault("base_channels", 8)
    kw.setdefault("policy_channels", 4)
    kw.setdefault("value_channels", 4)
    kw.setdefault("value_fc", 8)
    return NetworkConfig(**kw)


def make_config(tmp_path, **kw):
    return Config(
        network=tiny_net_cfg(),
        mcts=tiny_mcts(),
        training=TrainingConfig(**{**tiny_training().__dict__, **kw, "checkpoint_dir": str(tmp_path)}),
    )


def some_states():
    s = initial_state()
    out = []
    for x, z in [(0, 0), (4, 4), (1, 1), (0, 1), (3, 3)]:
        s = apply_move(s, x, z)
        out.append(s)
    return out


def uniform_pi(state):
    mask = action_mask(state)
    return mask / mask.sum()


# ---------------------------------------------------------------- replay buffer

def test_buffer_push_and_sample_shapes():
    buf = ReplayBuffer(1000)
    samples = [(encode(s), uniform_pi(s), s.current, 1.0) for s in some_states()]
    buf.push(samples)
    assert len(buf) == len(samples)
    s, pi, z = buf.sample(4, augment=False)
    assert s.shape == (4, 16, 5, 5)
    assert pi.shape == (4, 125)
    assert z.shape == (4, 1)


def test_buffer_no_augment_returns_exact():
    buf = ReplayBuffer(1000)
    samples = [(encode(s), uniform_pi(s), s.current, 1.0) for s in some_states()]
    buf.push(samples)
    stored_s = torch.stack([x[0] for x in samples])
    stored_pi = torch.stack([x[1] for x in samples])
    s, pi, z = buf.sample(5, augment=False)
    # Every drawn sample is one of the stored ones, verbatim.
    for i in range(5):
        assert (s[i] == stored_s).all(dim=(1, 2, 3)).any()
        assert (pi[i] == stored_pi).all(dim=1).any()
        assert z[i, 0] == 1.0


def test_buffer_augment_consistent_with_d4():
    """Every augmented sample is a D4 transform of a stored one, z invariant."""
    buf = ReplayBuffer(1000)
    states = some_states()
    samples = [(encode(s), uniform_pi(s), s.current, 1.0) for s in states]
    buf.push(samples)
    s, pi, z = buf.sample(24, augment=True)
    perms = d4_perms()
    for i in range(24):
        found = False
        for j, (st, pj, _, zj) in enumerate(samples):
            for perm in perms:
                if torch.equal(s[i], apply_d4(st, perm)) and torch.equal(pi[i], apply_d4_policy(pj, perm)):
                    found = True
                    break
            if found:
                break
        assert found, f"sample {i} is not a D4 transform of any stored sample"
        assert z[i, 0] == 1.0


def test_buffer_capacity_evicts_oldest():
    buf = ReplayBuffer(4)
    states = some_states()
    samples = [(encode(s), uniform_pi(s), s.current, 1.0) for s in states]
    buf.push(samples)
    assert len(buf) == 4
    buf.push([(encode(initial_state()), uniform_pi(initial_state()), WHITE, 0.0)])
    assert len(buf) == 4  # evicted the oldest


# ---------------------------------------------------------------- checkpointing

def test_checkpoint_round_trip(tmp_path):
    cfg = make_config(tmp_path)
    t = Trainer(cfg)
    t.iteration = 7
    # Take one optimizer step so Adam state exists to round-trip.
    s = torch.zeros(2, 16, 5, 5)
    pi = torch.full((2, 125), 1 / 125)
    z = torch.ones(2, 1)
    logits, value = t.net(s)
    loss = loss_fn(logits, value, pi, z)
    t.optimizer.zero_grad()
    loss.backward()
    t.optimizer.step()
    path = tmp_path / "test.pt"
    t.save_checkpoint(path, is_best=True)

    t2 = Trainer(cfg)
    t2.load_checkpoint(path)
    assert t2.iteration == 7
    for p1, p2 in zip(t.net.parameters(), t2.net.parameters()):
        assert torch.equal(p1, p2)
    # Optimizer state restored (Adam moments survive).
    assert t2.optimizer.state_dict()["state"]
    # Buffer restored.
    assert len(t2.buffer) == len(t.buffer)


def test_save_best_marks_iteration_file(tmp_path):
    cfg = make_config(tmp_path)
    t = Trainer(cfg)
    t.iteration = 3
    t.save_checkpoint(tmp_path / "best.pt", is_best=True)
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "best_iter_0003.pt").exists()


# ---------------------------------------------------------------- device portability

class _FakeTensor:
    """Records detach()/cpu() calls; stands in for real state-dict tensors."""

    def __init__(self):
        self.detach_called = False
        self.cpu_called = False

    def detach(self):
        self.detach_called = True
        return self

    def cpu(self):
        self.cpu_called = True
        return self


def test_save_checkpoint_cpuifies_state_before_saving(tmp_path, monkeypatch):
    """net/optimizer state must be moved to CPU before torch.save, so a GPU
    checkpoint resumes on CPU (and vice versa)."""
    from smartfour import train as train_mod

    cfg = make_config(tmp_path)
    t = Trainer(cfg)
    fake_state = {"w": _FakeTensor(), "b": _FakeTensor()}
    monkeypatch.setattr(t.net, "state_dict", lambda: fake_state)
    monkeypatch.setattr(t.optimizer, "state_dict", lambda: fake_state)
    captured = {}
    monkeypatch.setattr(
        train_mod.torch, "save",
        lambda payload, path: captured.update(payload=payload),
    )
    t.save_checkpoint(tmp_path / "x.pt")
    for key in ("net_state", "optimizer_state"):
        for name, value in captured["payload"][key].items():
            assert value.detach_called, f"{key}.{name} not detached"
            assert value.cpu_called, f"{key}.{name} not moved to cpu"


def _walk_tensors(obj):
    """Every torch.Tensor in a nested dict/list structure."""
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            out.extend(_walk_tensors(v))
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for v in obj:
            out.extend(_walk_tensors(v))
        return out
    return []


def test_save_checkpoint_payload_tensors_are_cpu(tmp_path, monkeypatch):
    """Saved payload carries CPU tensors (portable across machines), including
    optimizer moments (exp_avg/exp_avg_sq)."""
    from dataclasses import asdict

    from smartfour import train as train_mod

    cfg = make_config(tmp_path)
    t = Trainer(cfg)
    # Take one optimizer step so Adam moments exist to save.
    s = torch.zeros(2, 16, 5, 5)
    pi = torch.full((2, 125), 1 / 125)
    z = torch.ones(2, 1)
    logits, value = t.net(s)
    loss = loss_fn(logits, value, pi, z)
    t.optimizer.zero_grad()
    loss.backward()
    t.optimizer.step()
    captured = {}
    monkeypatch.setattr(
        train_mod.torch, "save",
        lambda payload, path: captured.update(payload=payload),
    )
    t.save_checkpoint(tmp_path / "x.pt")
    for key in ("net_state", "optimizer_state"):
        tensors = _walk_tensors(captured["payload"][key])
        assert tensors, f"{key} saved no tensors"
        assert all(v.device.type == "cpu" for v in tensors), key


def test_load_checkpoint_uses_map_location_cpu(tmp_path, monkeypatch):
    """Loading must force tensors to CPU (map_location) so a checkpoint saved
    on any device loads into a CPU Trainer."""
    from dataclasses import asdict

    from smartfour import train as train_mod

    cfg = make_config(tmp_path)
    t = Trainer(cfg)
    payload = {
        "iteration": 3,
        "network": asdict(cfg.network),
        "net_state": t.net.state_dict(),
        "optimizer_state": t.optimizer.state_dict(),
        "buffer": ([], [], []),
    }
    seen = {}

    def fake_load(path, **kwargs):
        seen.update(kwargs)
        return payload

    monkeypatch.setattr(train_mod.torch, "load", fake_load)
    t2 = Trainer(cfg)
    t2.load_checkpoint(tmp_path / "x.pt")
    assert seen.get("map_location") == "cpu"
    assert t2.iteration == 3


# ---------------------------------------------------------------- best-model selection

def test_best_replaced_only_above_ratio(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    t = Trainer(cfg)

    def fake_arena(net_a, net_b, games, eval_simulations):
        return (2, 0, 0)  # candidate wins 2 of 2 -> ratio 1.0

    monkeypatch.setattr(t, "_arena", fake_arena)
    t._maybe_update_best()
    assert (tmp_path / "best.pt").exists()
    # The best checkpoint holds the candidate's weights (iter 1).
    t2 = Trainer(cfg)
    t2.load_checkpoint(tmp_path / "best.pt")
    for p1, p2 in zip(t.net.parameters(), t2.net.parameters()):
        assert torch.equal(p1, p2)


def test_best_not_replaced_when_losing(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    t = Trainer(cfg)

    def fake_arena(net_a, net_b, games, eval_simulations):
        return (0, 2, 0)  # candidate loses

    monkeypatch.setattr(t, "_arena", fake_arena)
    t._maybe_update_best()
    # Candidate lost: no best checkpoint is written (nothing improved yet).
    assert not (tmp_path / "best.pt").exists()


# ---------------------------------------------------------------- training loop

def test_train_iteration_end_to_end(tmp_path):
    torch.manual_seed(0)
    cfg = make_config(tmp_path)
    t = Trainer(cfg)
    stats = t.train_iteration()
    assert stats["iteration"] == 1
    assert stats["buffer_size"] > 0
    assert (tmp_path / "latest.pt").exists()
    assert "loss_mean" in stats
    assert 0.0 < stats["loss_mean"] < 20.0


def test_training_reduces_loss_on_heldout(tmp_path):
    """After one iteration the net's loss on a fixed buffer batch drops."""
    torch.manual_seed(1)
    cfg = make_config(tmp_path, train_epochs=5, learning_rate=0.003, symmetry_augment=False)
    t = Trainer(cfg)
    # Pre-fill the buffer with self-play data, then freeze an eval batch.
    for _ in range(3):
        t._selfplay(t.net)
    eval_s, eval_pi, eval_z = t.buffer.sample(16, augment=True)
    with torch.no_grad():
        logits, value = t.net(eval_s)
        before = loss_fn(logits, value, eval_pi, eval_z).item()
    t._optimize()
    with torch.no_grad():
        logits, value = t.net(eval_s)
        after = loss_fn(logits, value, eval_pi, eval_z).item()
    assert after < before * 0.98, (before, after)


def test_resume_continues_iteration(tmp_path):
    torch.manual_seed(2)
    cfg = make_config(tmp_path)
    t = Trainer(cfg)
    t.train_iteration()
    t2 = Trainer(cfg)
    t2.load_checkpoint(tmp_path / "latest.pt")
    assert t2.iteration == 1
    stats = t2.train_iteration()
    assert stats["iteration"] == 2


def test_config_loads_from_toml():
    from smartfour.config import load_config

    cfg = load_config("config.toml")
    assert cfg.network.input_channels == 16
    assert cfg.network.blocks == 5
    assert cfg.mcts.simulations == 200
    assert cfg.training.selfplay_games == 100
    assert cfg.training.selfplay_workers == 1
