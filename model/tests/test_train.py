"""Tests for smartfour.train — replay buffer, checkpointing, best-model, loop."""

import math

import pytest
import torch

from smartfour.arena import play_arena
from smartfour.config import Config, MCTSConfig, NetworkConfig, TrainingConfig
from smartfour.encode import action_mask, apply_d4, apply_d4_policy, d4_perms, encode
from smartfour.game import WHITE, apply_move, initial_state, legal_moves
from smartfour.network import ResNet, loss_fn
from smartfour.train import ReplayBuffer, Trainer, plys_postfix


def tiny_training(**kw):
    kw.setdefault("selfplay_games", 1)
    kw.setdefault("workers", 1)  # unit tests stay sequential
    kw.setdefault("train_epochs", 5)
    kw.setdefault("batch_size", 8)
    kw.setdefault("replay_capacity_games", 10_000)
    kw.setdefault("learning_rate", 0.001)
    kw.setdefault("weight_decay", 0.0)
    kw.setdefault("symmetry_augment", True)
    kw.setdefault("eval_games", 1)
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


# ---------------------------------------------------------------- self-play bar metric

def test_plys_postfix_rounds_to_integer():
    assert plys_postfix(0, 0) == "0 plys/game"      # nothing played yet
    assert plys_postfix(44, 2) == "22 plys/game"    # exact
    assert plys_postfix(47, 3) == "16 plys/game"    # 15.67 -> 16
    assert plys_postfix(45, 2) == "22 plys/game"    # 22.5 -> round-half-even


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


def game_samples(seed_state, n):
    """n samples standing in for one game's positions."""
    s = seed_state
    out = []
    for _ in range(n):
        out.append((encode(s), uniform_pi(s), s.current, 1.0))
        x, z = legal_moves(s)[0]
        s = apply_move(s, x, z)
    return out


def test_buffer_capacity_evicts_whole_games():
    """Capacity counts games: pushing a 6th game into a capacity-5 buffer
    evicts the entire 1st game, never splits one."""
    buf = ReplayBuffer(5)
    states = some_states()
    for i, s in enumerate(states):
        buf.push(game_samples(s, 2 + i))  # games of 2,3,4,5,6 samples
    assert buf.games == 5
    assert len(buf) == 2 + 3 + 4 + 5 + 6
    buf.push(game_samples(initial_state(), 3))
    assert buf.games == 5
    assert len(buf) == 3 + 4 + 5 + 6 + 3  # the first game (2 samples) is gone


def test_buffer_sample_only_from_retained_games():
    buf = ReplayBuffer(2)
    states = some_states()
    for s in states:
        buf.push(game_samples(s, 3))
    old = [encode(states[0]), encode(states[1]), encode(states[2])]
    for _ in range(20):
        s, pi, z = buf.sample(8, augment=False)
        for i in range(8):
            assert not any((s[i] == o).all() for o in old)


def test_buffer_state_load_round_trip_preserves_games():
    buf = ReplayBuffer(10)
    for i, s in enumerate(some_states()):
        buf.push(game_samples(s, 2 + i))
    buf2 = ReplayBuffer(10)
    buf2.load_state(buf.state())
    assert buf2.games == buf.games
    assert len(buf2) == len(buf)
    # Same flat contents in the same order.
    a, b = buf.state(), buf2.state()
    assert a[3] == b[3]
    for x, y in zip(a[0], b[0]):
        assert torch.equal(x, y)


def test_buffer_load_rejects_ply_based_state():
    """Legacy ply-based checkpoints (3-tuple buffer state) hard-error with a
    --restart hint instead of silently mis-splitting games."""
    buf = ReplayBuffer(10)
    samples = game_samples(initial_state(), 4)
    with pytest.raises(SystemExit, match="--restart"):
        buf.load_state((samples[0][0], samples[0][1], samples[0][3]))


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
    t.save_checkpoint(path)

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
        return (2, 0, 0, 0)  # candidate wins 2 of 2 -> ratio 1.0

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
        return (0, 2, 0, 0)  # candidate loses

    monkeypatch.setattr(t, "_arena", fake_arena)
    t._maybe_update_best(1)
    # Candidate lost: no best checkpoint is written (nothing improved yet).
    assert not (tmp_path / "best.pt").exists()
    assert t.best_iteration == 0



def test_best_replaced_with_draws_scored_half(tmp_path, monkeypatch):
    """Draws count as half a win: 1 win + 1 draw + 0 losses = 0.75 >= 0.55."""
    cfg = make_config(tmp_path)
    t = Trainer(cfg)

    def fake_arena(net_a, net_b, games):
        return (1, 0, 1, 0)  # ratio = (1 + 0.5) / 2 = 0.75

    monkeypatch.setattr(t, "_arena", fake_arena)
    t._maybe_update_best(1)
    assert (tmp_path / "best.pt").exists()
    assert t.best_iteration == 1


def test_best_not_replaced_when_draws_dilute_below_ratio(tmp_path, monkeypatch):
    """10 draws + 12 wins + 8 losses over 30 games: 12+5=17, 17/30 < 0.55."""
    cfg = make_config(tmp_path, eval_games=30, arena_win_ratio=0.55)
    t = Trainer(cfg)

    def fake_arena(net_a, net_b, games):
        return (12, 8, 10, 0)  # ratio = (12 + 5) / 30 = 0.567 -> promotes

    monkeypatch.setattr(t, "_arena", fake_arena)
    t._maybe_update_best(1)
    assert (tmp_path / "best.pt").exists()

    def fake_arena_short(net_a, net_b, games):
        return (11, 8, 11, 0)  # ratio = (11 + 5.5) / 30 = 0.55 -> promotes (>=)

    monkeypatch.setattr(t, "_arena", fake_arena_short)
    t._maybe_update_best(2)
    assert t.best_iteration == 2

    def fake_arena_fail(net_a, net_b, games):
        return (11, 9, 10, 0)  # ratio = (11 + 5) / 30 = 0.533 -> no promote

    monkeypatch.setattr(t, "_arena", fake_arena_fail)
    t._maybe_update_best(3)
    assert t.best_iteration == 2

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
    for _ in range(2):
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

def test_iterations_write_only_latest(tmp_path):
    """Each completed iteration rewrites latest.pt; no iter_NNNN.pt
    snapshots are created (they were removed to save space)."""
    cfg = make_config(tmp_path, eval_games=2)
    t = Trainer(cfg)
    t.train_iteration()
    t.train_iteration()
    assert not list(tmp_path.glob("iter_*.pt"))
    latest = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert latest["iteration"] == 2
    assert "buffer" in latest  # latest holds the exact resume state
    assert len(latest["buffer"][0]) == len(t.buffer)


def test_iteration_counter_advances_only_on_completion(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    t = Trainer(cfg)
    t.train_iteration()
    assert t.iteration == 1

    def boom(net):
        raise KeyboardInterrupt

    monkeypatch.setattr(t, "_selfplay", boom)
    with pytest.raises(KeyboardInterrupt):
        t.train_iteration()
    assert t.iteration == 1  # the interrupted iteration is never counted


def test_checkpoint_without_buffer_hard_errors(tmp_path):
    """latest.pt always carries the replay buffer; a bufferless payload is no
    longer a valid checkpoint (old iter_*.pt format) and hard-errors."""
    cfg = make_config(tmp_path)
    t = Trainer(cfg)
    t.iteration = 3
    payload = t._payload()
    del payload["buffer"]
    t2 = Trainer(cfg)
    with pytest.raises(SystemExit, match="no replay buffer"):
        t2._apply_payload(payload)


def test_train_stops_at_target(tmp_path):
    cfg = make_config(tmp_path, eval_games=2)
    t = Trainer(cfg)
    t.iteration = 2
    stats = t.train(4)
    assert [s["iteration"] for s in stats] == [3, 4]
    assert not list(tmp_path.glob("iter_*.pt"))
    assert (tmp_path / "latest.pt").exists()


# ---------------------------------------------------------------- atomic saves

def test_save_is_atomic_on_failure(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    t = Trainer(cfg)
    t.iteration = 3
    target = tmp_path / "latest.pt"
    t.save_checkpoint(target)
    baseline = target.read_bytes()
    real_save = torch.save

    def failing_save(obj, f):
        real_save(obj, f)
        raise OSError("disk full")

    monkeypatch.setattr("smartfour.train.torch.save", failing_save)
    with pytest.raises(OSError):
        t.save_checkpoint(target)
    assert target.read_bytes() == baseline  # previous checkpoint untouched
    assert not list(tmp_path.glob("*.tmp"))  # temp file cleaned up


# ---------------------------------------------------------------- optimize guards

def test_optimize_skipped_when_buffer_empty(tmp_path, capsys):
    cfg = make_config(tmp_path, batch_size=8)
    t = Trainer(cfg)
    loss, pol, val = t._optimize()  # no self-play was ever run
    assert math.isnan(loss) and math.isnan(pol) and math.isnan(val)
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

    cfg = load_config("config_small.toml")
    assert cfg.network.input_channels == 16
    assert cfg.mcts.simulations == 400
    assert cfg.training.selfplay_games == 50
