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
    kw.setdefault("selfplay_workers", 1)  # unit tests stay sequential
    kw.setdefault("train_epochs", 5)
    kw.setdefault("batch_size", 8)
    kw.setdefault("replay_capacity", 10_000)
    kw.setdefault("learning_rate", 0.001)
    kw.setdefault("weight_decay", 0.0)
    kw.setdefault("symmetry_augment", True)
    kw.setdefault("eval_games", 2)
    kw.setdefault("arena_workers", 1)  # unit tests stay sequential
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
    # Make best_net differ from the current net so the round-trip is honest.
    with torch.no_grad():
        for p in t.best_net.parameters():
            p.add_(0.5)
    t.best_iteration = 4
    t.save_checkpoint(path, include_buffer=True)

    t2 = Trainer(cfg)
    t2.load_checkpoint(path)
    assert t2.iteration == 7
    for p1, p2 in zip(t.net.parameters(), t2.net.parameters()):
        assert torch.equal(p1, p2)
    # Optimizer state restored (Adam moments survive).
    assert t2.optimizer.state_dict()["state"]
    # Buffer restored.
    assert len(t2.buffer) == len(t.buffer)
    # The arena baseline (best_net) is restored, not re-initialized to net.
    for p1, p2 in zip(t.best_net.parameters(), t2.best_net.parameters()):
        assert torch.equal(p1, p2)
    assert t2.best_iteration == 4


def test_save_best_writes_slim_snapshot(tmp_path):
    """best.pt is an inference snapshot (best weights + iteration), not a
    training state: no buffer, no optimizer."""
    cfg = make_config(tmp_path)
    t = Trainer(cfg)
    t.iteration = 5
    t.best_iteration = 4
    with torch.no_grad():
        for p in t.best_net.parameters():
            p.add_(1.0)  # best differs from the current net
    t.save_best()
    payload = torch.load(tmp_path / "best.pt", weights_only=False)
    assert payload["iteration"] == 4
    assert "net_state" in payload
    assert "buffer" not in payload
    assert "optimizer_state" not in payload
    # best.pt carries the best net's weights, not the current net's.
    t2 = Trainer(cfg)
    t2.best_net.load_state_dict(payload["net_state"])
    for p1, p2 in zip(t.best_net.parameters(), t2.best_net.parameters()):
        assert torch.equal(p1, p2)


# ---------------------------------------------------------------- best-model selection

def test_best_replaced_only_above_ratio(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    t = Trainer(cfg)

    def fake_arena(net_a, net_b, games):
        return (2, 0, 0)  # candidate wins 2 of 2 -> ratio 1.0

    monkeypatch.setattr(t, "_arena", fake_arena)
    t._maybe_update_best(1)
    assert (tmp_path / "best.pt").exists()
    assert t.best_iteration == 1
    # best.pt holds the candidate's weights (the net that won the arena).
    payload = torch.load(tmp_path / "best.pt", weights_only=False)
    t2 = Trainer(cfg)
    t2.best_net.load_state_dict(payload["net_state"])
    for p1, p2 in zip(t.net.parameters(), t2.best_net.parameters()):
        assert torch.equal(p1, p2)
    # the trainer's own arena baseline is the candidate too
    for p1, p2 in zip(t.net.parameters(), t.best_net.parameters()):
        assert torch.equal(p1, p2)


def test_best_not_replaced_when_losing(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    t = Trainer(cfg)

    def fake_arena(net_a, net_b, games):
        return (0, 2, 0)  # candidate loses

    monkeypatch.setattr(t, "_arena", fake_arena)
    t._maybe_update_best(1)
    # Candidate lost: no best checkpoint is written (nothing improved yet).
    assert not (tmp_path / "best.pt").exists()
    assert t.best_iteration == 0


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


# ---------------------------------------------------------------- per-iteration checkpoints

def test_every_iteration_writes_iter_and_latest(tmp_path):
    cfg = make_config(tmp_path, eval_games=2)
    t = Trainer(cfg)
    t.train_iteration()
    t.train_iteration()
    assert (tmp_path / "iter_0001.pt").exists()
    assert (tmp_path / "iter_0002.pt").exists()
    iter_payload = torch.load(tmp_path / "iter_0002.pt", weights_only=False)
    assert iter_payload["iteration"] == 2
    assert "best_net_state" in iter_payload
    assert "buffer" not in iter_payload  # iter files are light snapshots
    latest = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert latest["iteration"] == 2
    assert "buffer" in latest  # latest is the exact resume anchor
    assert len(latest["buffer"][0]) == len(t.buffer)


def test_iteration_counter_advances_only_on_completion(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    t = Trainer(cfg)
    t.train_iteration()
    assert t.iteration == 1

    def boom(net):
        raise KeyboardInterrupt

    # instance attribute: plain function, no implicit self
    monkeypatch.setattr(t, "_selfplay", boom)
    with pytest.raises(KeyboardInterrupt):
        t.train_iteration()
    assert t.iteration == 1  # the interrupted iteration is never counted
    # the interrupt handler's save therefore carries the last completed one
    t.save_checkpoint(tmp_path / "latest.pt", include_buffer=True)
    assert torch.load(tmp_path / "latest.pt", weights_only=False)["iteration"] == 1


def test_train_stops_at_target(tmp_path):
    cfg = make_config(tmp_path, eval_games=2)
    t = Trainer(cfg)
    t.iteration = 2
    stats = t.train(4)
    assert [s["iteration"] for s in stats] == [3, 4]
    assert (tmp_path / "iter_0003.pt").exists()
    assert (tmp_path / "iter_0004.pt").exists()


# ---------------------------------------------------------------- atomic saves

def test_save_is_atomic_on_failure(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    t = Trainer(cfg)
    t.iteration = 3
    target = tmp_path / "latest.pt"
    t.save_checkpoint(target, include_buffer=True)
    baseline = target.read_bytes()
    real_save = torch.save

    def failing_save(obj, f):
        real_save(obj, f)
        raise OSError("disk full")

    monkeypatch.setattr("smartfour.train.torch.save", failing_save)
    with pytest.raises(OSError):
        t.save_checkpoint(target, include_buffer=True)
    assert target.read_bytes() == baseline  # previous checkpoint untouched
    assert not list(tmp_path.glob("*.tmp"))  # temp file cleaned up


# ---------------------------------------------------------------- optimize guards

def test_optimize_skipped_when_buffer_empty(tmp_path, capsys):
    cfg = make_config(tmp_path, batch_size=8)
    t = Trainer(cfg)
    loss = t._optimize()  # no self-play was ever run
    assert math.isnan(loss)
    out, err = capsys.readouterr()
    assert "skipping optimize" in out + err


def test_bn_running_stats_update_during_optimize(tmp_path):
    """MCTS leaves the net in eval mode; _optimize must switch back to train
    mode so BatchNorm running statistics actually update."""
    cfg = make_config(tmp_path, batch_size=8, train_epochs=2)
    t = Trainer(cfg)
    t._selfplay(t.net)
    assert not t.net.training  # MCTS forced eval
    bn = t.net.blocks[0].bn1
    before = bn.running_mean.clone()
    t._optimize()
    after = bn.running_mean.clone()
    assert not torch.equal(before, after)


def test_config_loads_from_toml():
    from smartfour.config import load_config

    cfg = load_config("config.toml")
    assert cfg.network.input_channels == 16
    assert cfg.network.blocks == 5
    assert cfg.mcts.simulations == 200
    assert cfg.training.selfplay_games == 100
    assert cfg.training.selfplay_workers == 8
