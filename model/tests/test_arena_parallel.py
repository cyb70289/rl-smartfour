"""Tests for parallel arena — worker process, result collection, dispatch.

Arena parallelism mirrors self-play: one spawned process per `workers` plays
its share of the alternating-color games with fresh copies of both nets and
ships per-game results (winner color in net_a's frame) back over a queue; the
trainer counts exactly `games` games and fails fast on worker errors.
`workers=1` keeps the exact sequential code path (no spawn).
"""

import multiprocessing as mp
import queue

import pytest
import torch

from smartfour.arena import _result_in_a_frame, arena_worker
from smartfour.config import Config, MCTSConfig, NetworkConfig, TrainingConfig
from smartfour.game import BLACK, DRAW, WHITE
from smartfour.network import ResNet


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


def make_config(tmp_path, **kw):
    defaults = dict(
        train_epochs=1,
        batch_size=16,
        replay_capacity=10_000,
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


# ---------------------------------------------------------------- result mapping

def test_result_in_a_frame_identity_when_a_is_white():
    assert _result_in_a_frame(WHITE, True) == WHITE
    assert _result_in_a_frame(BLACK, True) == BLACK
    assert _result_in_a_frame(DRAW, True) == DRAW


def test_result_in_a_frame_flips_when_a_is_black():
    assert _result_in_a_frame(WHITE, False) == BLACK
    assert _result_in_a_frame(BLACK, False) == WHITE
    assert _result_in_a_frame(DRAW, False) == DRAW


# ---------------------------------------------------------------- worker process

def test_arena_worker_plays_assigned_games():
    torch.manual_seed(0)
    net_cfg = tiny_net_cfg()
    net = ResNet(net_cfg)
    # fork (not spawn): the child inherits the already-imported torch, so the
    # probe costs ~0.1s instead of a ~3s re-import per spawn. Production
    # keeps spawn; this only exercises the worker function itself.
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(
        target=arena_worker,
        args=(net.state_dict(), net.state_dict(), net_cfg, tiny_mcts(),
              4, 0, 7, None, q),
    )
    p.start()
    results = []
    for _ in range(4):
        msg = q.get(timeout=120)
        assert not (isinstance(msg, tuple) and msg and msg[0] == "__worker_error__"), msg
        results.append(msg)
    p.join(timeout=120)
    assert p.exitcode == 0
    assert len(results) == 4
    assert all(r in (WHITE, BLACK, DRAW) for r in results)


def test_arena_worker_continues_color_schedule_from_start():
    """A worker with start=1 must still report results in net_a's frame: the
    first game (global index 1) is the swapped color, so a WHITE outcome
    counts as a net_b win and arrives as BLACK."""
    torch.manual_seed(0)
    net_cfg = tiny_net_cfg()
    net = ResNet(net_cfg)
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(
        target=arena_worker,
        args=(net.state_dict(), net.state_dict(), net_cfg, tiny_mcts(),
              2, 1, 8, None, q),
    )
    p.start()
    results = []
    for _ in range(2):
        msg = q.get(timeout=120)
        assert not (isinstance(msg, tuple) and msg and msg[0] == "__worker_error__"), msg
        results.append(msg)
    p.join(timeout=120)
    assert p.exitcode == 0
    assert all(r in (WHITE, BLACK, DRAW) for r in results)


def test_arena_worker_reports_failure_in_band():
    """A worker crash (bad net state) surfaces as an error marker, not a hang."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    bad_state = {"nope": torch.zeros(1)}
    p = ctx.Process(
        target=arena_worker,
        args=(bad_state, bad_state, tiny_net_cfg(), tiny_mcts(),
              1, 0, 7, None, q),
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


def fake_bar():
    return type("Bar", (), {"set_postfix": lambda self, **k: None, "update": lambda self, n: None})()


def test_collect_arena_counts_results_in_a_frame(tmp_path):
    from smartfour.train import Trainer

    t = Trainer(make_config(tmp_path))
    q = FakeQueue([WHITE, BLACK, DRAW, WHITE])
    procs = [FakeProc(), FakeProc()]
    a_wins, b_wins, draws = t._collect_arena(4, procs, q, fake_bar())
    assert (a_wins, b_wins, draws) == (2, 1, 1)


def test_collect_arena_raises_on_worker_error_marker(tmp_path):
    from smartfour.train import Trainer

    t = Trainer(make_config(tmp_path))
    q = FakeQueue([("__worker_error__", "RuntimeError: boom")])
    procs = [FakeProc(), FakeProc()]
    with pytest.raises(RuntimeError, match="boom"):
        t._collect_arena(3, procs, q, fake_bar())
    assert all(p.terminated for p in procs)  # survivors cleaned up


def test_collect_arena_raises_when_workers_die_early(tmp_path):
    from smartfour.train import Trainer

    t = Trainer(make_config(tmp_path))
    q = FakeQueue([])
    procs = [FakeProc(alive=False)]
    with pytest.raises(RuntimeError, match="exited early"):
        t._collect_arena(3, procs, q, fake_bar())


def test_collect_arena_raises_on_nonzero_worker_exit(tmp_path):
    from smartfour.train import Trainer

    t = Trainer(make_config(tmp_path))
    q = FakeQueue([WHITE])
    procs = [FakeProc(exitcode=1)]
    with pytest.raises(RuntimeError, match="exit"):
        t._collect_arena(1, procs, q, fake_bar())


# ---------------------------------------------------------------- trainer integration

def test_trainer_arena_one_matches_sequential(tmp_path, monkeypatch):
    """workers=1 must keep the exact sequential code path (no spawn)."""
    import smartfour.train as train_mod
    from smartfour.train import Trainer

    calls = []

    def fake_play_arena(net_a, net_b, mcts_cfg, games, progress=None):
        calls.append((net_a, net_b, mcts_cfg, games))
        return (3, 1, 0)

    def should_not_spawn(*args, **kwargs):
        raise AssertionError("workers=1 must not spawn workers")

    monkeypatch.setattr(train_mod, "play_arena", fake_play_arena)
    monkeypatch.setattr(Trainer, "_arena_parallel", should_not_spawn)
    t = Trainer(make_config(tmp_path, workers=1))
    result = t._arena(t.net, t.best_net, 4)
    assert result == (3, 1, 0)
    assert calls == [(t.net, t.best_net, t.cfg.mcts, 4)]


def test_trainer_arena_with_workers(tmp_path, monkeypatch):
    torch.manual_seed(0)
    import smartfour.train as train_mod

    # Exercise the parallel dispatch/collection path without the ~3s
    # torch re-import per spawned worker; the spawn context itself is
    # covered by test_arena_spawns_daemonic_workers.
    fork_ctx = mp.get_context("fork")
    monkeypatch.setattr(train_mod.multiprocessing, "get_context", lambda name: fork_ctx)
    from smartfour.train import Trainer

    t = Trainer(make_config(tmp_path, workers=2))
    wins, losses, draws = t._arena(t.net, t.best_net, 2)
    assert wins + losses + draws == 2


def test_arena_spawns_daemonic_workers(tmp_path, monkeypatch):
    """Workers are daemonic so a dying parent can never orphan them."""
    import smartfour.train as train_mod
    from smartfour.train import Trainer

    seen = {}

    class RecCtx:
        def Queue(self):
            return FakeQueue([WHITE, BLACK, DRAW, WHITE])

        def Process(self, *args, **kwargs):
            seen["daemon"] = kwargs.get("daemon")
            return FakeProc()

    monkeypatch.setattr(train_mod.multiprocessing, "get_context", lambda name: RecCtx())
    t = Trainer(make_config(tmp_path, workers=2))
    assert t._arena(t.net, t.best_net, 4) == (2, 1, 1)
    assert seen["daemon"] is True
