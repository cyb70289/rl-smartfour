"""Tests for parallel self-play — worker split, process workers, collection.

Self-play parallelism uses spawned processes (Python's GIL serializes the
MCTS tree logic, so threads cannot use more than one CPU core). Each worker
plays its share of games with its own net copy and ships samples back over a
queue; the trainer collects exactly `games` games and fails fast on worker
errors.
"""

import multiprocessing as mp
import queue
import signal

import pytest
import torch

from smartfour.config import Config, MCTSConfig, NetworkConfig, TrainingConfig, load_config
from smartfour.encode import action_mask, encode
from smartfour.game import BLACK, WHITE, initial_state
from smartfour.network import ResNet
from smartfour.selfplay import (
    samples_from_ipc,
    samples_to_ipc,
    selfplay_worker,
    split_games,
)

# ---------------------------------------------------------------- split_games


def test_split_games_even():
    assert split_games(10, 2) == [5, 5]
    assert split_games(9, 3) == [3, 3, 3]


def test_split_games_remainder_to_first_workers():
    assert split_games(10, 3) == [4, 3, 3]
    assert split_games(7, 4) == [2, 2, 2, 1]
    assert split_games(1, 3) == [1, 0, 0]
    assert split_games(0, 2) == [0, 0]


def test_split_games_invariants():
    for games in range(0, 13):
        for workers in range(1, 6):
            parts = split_games(games, workers)
            assert sum(parts) == games
            assert len(parts) == workers
            assert all(p >= 0 for p in parts)
            if games:
                assert max(parts) - min(parts) <= 1


def test_split_games_rejects_bad_input():
    with pytest.raises(ValueError):
        split_games(10, 0)
    with pytest.raises(ValueError):
        split_games(-1, 2)


# ---------------------------------------------------------------- worker process

def tiny_net_cfg(**kw):
    kw.setdefault("input_channels", 16)
    kw.setdefault("blocks", 1)
    kw.setdefault("base_channels", 8)
    kw.setdefault("policy_channels", 4)
    kw.setdefault("value_channels", 4)
    kw.setdefault("value_fc", 8)
    return NetworkConfig(**kw)


def tiny_mcts(**kw):
    kw.setdefault("simulations", 5)
    kw.setdefault("c_puct", 1.0)
    kw.setdefault("dirichlet_alpha", 0.3)
    kw.setdefault("dirichlet_epsilon", 0.25)
    kw.setdefault("temperature_threshold", 12)
    kw.setdefault("batch_eval_size", 4)
    return MCTSConfig(**kw)


def uniform_pi(state):
    mask = action_mask(state)
    return mask / mask.sum()


def check_samples(samples):
    assert samples, "worker returned an empty game"
    for s, pi, player, z in samples:
        assert s.shape == (16, 5, 5)
        assert pi.shape == (125,)
        assert abs(pi.sum().item() - 1.0) < 1e-6
        assert z in (-1.0, 0.0, 1.0)
        assert player in (WHITE, BLACK)


def test_selfplay_worker_plays_assigned_games():
    torch.manual_seed(0)
    net_cfg = tiny_net_cfg()
    net = ResNet(net_cfg)
    # Match production: forking after PyTorch has used its native thread pool
    # can deadlock the child before it reports a result.
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=selfplay_worker,
        args=(net.state_dict(), net_cfg, tiny_mcts(), 12, 2, 7, None, q),
    )
    p.start()
    received = []
    stats_received = []
    for _ in range(2):
        msg = q.get(timeout=120)
        assert not (isinstance(msg, tuple) and msg and msg[0] == "__worker_error__"), msg
        assert isinstance(msg, tuple) and len(msg) == 2, "worker must ship (samples, stats)"
        received.append(samples_from_ipc(msg[0]))
        stats_received.append(msg[1])
    p.join(timeout=120)
    assert p.exitcode == 0
    assert len(received) == 2
    for samples, stats in zip(received, stats_received):
        check_samples(samples)
        assert stats["plies"] == len(samples)
        assert stats["winner"] in ("white", "black", "draw")
        assert len(stats["sample_hashes"]) == len(samples)


def test_selfplay_worker_reports_failure_in_band():
    """A worker crash (bad net state) surfaces as an error marker, not a hang."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    bad_state = {"nope": torch.zeros(1)}
    p = ctx.Process(
        target=selfplay_worker,
        args=(bad_state, tiny_net_cfg(), tiny_mcts(), 12, 1, 7, None, q),
    )
    p.start()
    msg = q.get(timeout=120)
    assert isinstance(msg, tuple) and msg[0] == "__worker_error__"
    assert msg[1]
    p.join(timeout=120)


# ---------------------------------------------------------------- collection logic

class FakeProc:
    def __init__(self, alive=True, exitcode=0):
        self._alive = alive
        self.exitcode = exitcode
        self.terminated = False

    def start(self):
        pass

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self._alive = False

    def terminate(self):
        self.terminated = True
        self._alive = False


class FakeQueue:
    def __init__(self, msgs):
        self.msgs = list(msgs)

    def get(self, timeout=None):
        if not self.msgs:
            raise queue.Empty
        return self.msgs.pop(0)


def make_config(tmp_path, **kw):
    defaults = dict(
        train_epochs=1,
        batch_size=16,
        replay_capacity_games=10_000,
        learning_rate=0.001,
        weight_decay=0.0,
        symmetry_augment=True,
        eval_games=2,
        workers=1,  # parallel tests override explicitly
        arena_win_ratio=0.55,
        seed=0,
        checkpoint_dir=str(tmp_path),
    )
    defaults.update(kw)
    return Config(
        network=tiny_net_cfg(),
        mcts=tiny_mcts(),
        training=TrainingConfig(**defaults),
    )


def fake_bar():
    return type("Bar", (), {"set_postfix_str": lambda self, s: None, "update": lambda self, n: None})()


def one_sample():
    s = initial_state()
    return (encode(s), uniform_pi(s), WHITE, 1.0)


def test_collect_pushes_all_games(tmp_path):
    from smartfour.train import Trainer

    t = Trainer(make_config(tmp_path))
    ipc = samples_to_ipc([one_sample()])
    q = FakeQueue([ipc, ipc])
    t._collect_selfplay(2, [FakeProc(), FakeProc()], q, fake_bar())
    assert len(t.buffer) == 2
    s, pi, z = t.buffer.sample(2, augment=False)
    assert s.shape == (2, 16, 5, 5)
    assert pi.shape == (2, 125)


def test_collect_reports_avg_plys_per_game(tmp_path):
    """The self-play bar shows the running average plies per game, refreshed
    once per completed game. Plies = stored samples = moves (one move per
    player per ply; a turn is two plies)."""
    from smartfour.train import Trainer

    t = Trainer(make_config(tmp_path))
    games = [samples_to_ipc([one_sample()] * 1), samples_to_ipc([one_sample()] * 3)]
    postfixes = []

    class RecBar:
        def set_postfix_str(self, s):
            postfixes.append(s)

        def update(self, n):
            pass

    t._collect_selfplay(2, [FakeProc(), FakeProc()], FakeQueue(games), RecBar())
    assert postfixes == ["1 plys/game", "2 plys/game"]  # (1 + 3) / 2 = 2
    assert len(t.buffer) == 4


def test_collect_raises_on_worker_error_marker(tmp_path):
    from smartfour.train import Trainer

    t = Trainer(make_config(tmp_path))
    q = FakeQueue([("__worker_error__", "RuntimeError: boom")])
    procs = [FakeProc(), FakeProc()]
    with pytest.raises(RuntimeError, match="boom"):
        t._collect_selfplay(3, procs, q, fake_bar())
    assert all(p.terminated for p in procs)  # survivors cleaned up


def test_collect_raises_when_workers_die_early(tmp_path):
    from smartfour.train import Trainer

    t = Trainer(make_config(tmp_path))
    q = FakeQueue([])
    procs = [FakeProc(alive=False)]
    with pytest.raises(RuntimeError, match="exited early"):
        t._collect_selfplay(3, procs, q, fake_bar())


def test_collect_raises_on_nonzero_worker_exit(tmp_path):
    from smartfour.train import Trainer

    t = Trainer(make_config(tmp_path))
    q = FakeQueue([samples_to_ipc([one_sample()])])
    procs = [FakeProc(exitcode=1)]
    with pytest.raises(RuntimeError, match="exit"):
        t._collect_selfplay(1, procs, q, fake_bar())


# ---------------------------------------------------------------- trainer integration

def test_trainer_selfplay_with_workers(tmp_path, monkeypatch):
    torch.manual_seed(0)
    import smartfour.train as train_mod

    spawn_ctx = mp.get_context("spawn")
    monkeypatch.setattr(train_mod.multiprocessing, "get_context", lambda name: spawn_ctx)
    cfg = make_config(tmp_path, selfplay_games=2, workers=2)
    from smartfour.train import Trainer

    t = Trainer(cfg)
    t._selfplay(t.net)
    assert len(t.buffer) > 0
    states, pis, zs, lens = t.buffer.state()
    assert sum(lens) == len(states) == len(t.buffer)
    for s, pi, z in zip(states, pis, zs):
        assert s.shape == (16, 5, 5)
        assert pi.shape == (125,)
        assert abs(pi.sum().item() - 1.0) < 1e-6
        assert z in (-1.0, 0.0, 1.0)


def test_trainer_selfplay_one_matches_sequential(tmp_path):
    """workers=1 must keep the exact sequential code path (no spawn)."""
    torch.manual_seed(0)
    from smartfour.train import Trainer

    t = Trainer(make_config(tmp_path, selfplay_games=2, workers=1))
    t._selfplay(t.net)
    assert len(t.buffer) >= 2  # every sample of both games recorded


# ---------------------------------------------------------------- config validation

def test_load_config_rejects_zero_workers(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text("[training]\nworkers = 0\n")
    with pytest.raises(ValueError, match="workers"):
        load_config(str(p))


def test_load_config_defaults_workers_to_eight(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text("")
    assert load_config(str(p)).training.workers == 8


# ---------------------------------------------------------------- interrupt robustness

def _sigint_probe(q):
    """Spawn target: report whether smartfour.selfplay.ignore_sigint stuck."""
    from smartfour.selfplay import ignore_sigint

    ignore_sigint()
    q.put(signal.getsignal(signal.SIGINT) == signal.SIG_IGN)


def test_worker_ignores_sigint():
    """Ctrl-C must not kill workers; the parent alone decides when to stop.

    The trainer's interrupt path terminates workers explicitly; a worker that
    reacted to the terminal's SIGINT would race the parent's cleanup.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_sigint_probe, args=(q,))
    p.start()
    assert q.get(timeout=60) is True
    p.join(timeout=60)
    assert p.exitcode == 0


def test_selfplay_spawns_daemonic_workers(tmp_path, monkeypatch):
    """Workers are daemonic so a dying parent can never orphan them."""
    import smartfour.train as train_mod
    from smartfour.train import Trainer

    seen = {}

    class RecCtx:
        def Queue(self):
            return FakeQueue([samples_to_ipc([one_sample()]), samples_to_ipc([one_sample()])])

        def Process(self, *args, **kwargs):
            seen["daemon"] = kwargs.get("daemon")
            return FakeProc()

    monkeypatch.setattr(train_mod.multiprocessing, "get_context", lambda name: RecCtx())
    t = Trainer(make_config(tmp_path, selfplay_games=2, workers=2))
    t._selfplay_parallel(t.net, 2, 2, fake_bar())
    assert seen["daemon"] is True
    assert len(t.buffer) == 2
