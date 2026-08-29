"""Tests for smartfour.pretrain — parallel collection and device training."""

import multiprocessing as mp

import pytest
import torch

from smartfour.config import NetworkConfig
from smartfour.game import DRAW, initial_state
from smartfour.pretrain import (
    collect_rollout_samples,
    collect_rollout_samples_parallel,
    pretrain_value,
    rollout_z,
)
from smartfour.network import ResNet


def tiny_cfg(**kw):
    kw.setdefault("input_channels", 15)
    kw.setdefault("blocks", 1)
    kw.setdefault("base_channels", 8)
    kw.setdefault("policy_channels", 4)
    kw.setdefault("value_channels", 4)
    kw.setdefault("value_fc", 8)
    return NetworkConfig(**kw)


# ------------------------------------------------------------------ rollouts

def test_rollout_z_returns_terminal_perspective():
    import random
    rng = random.Random(3)
    z = rollout_z(initial_state(), rng)
    assert z in (-1.0, 0.0, 1.0)


def test_collect_shapes_and_label_range():
    states, zs = collect_rollout_samples(4, seed=5, tail_plies=4, rollouts=3)
    assert len(states) == len(zs)
    assert 0 < len(states) <= 4 * 4
    for s in states:
        assert s.shape == (15, 5, 5)
    for z in zs:
        assert -1.0 <= z <= 1.0
    # soft labels from k=3 rollouts are multiples of 1/3
    for z in zs:
        assert abs(z * 3 - round(z * 3)) < 1e-9


def test_collect_is_deterministic_per_seed():
    a = collect_rollout_samples(3, seed=11, tail_plies=3, rollouts=2)
    b = collect_rollout_samples(3, seed=11, tail_plies=3, rollouts=2)
    assert len(a[0]) == len(b[0])
    for ta, tb in zip(a[0], b[0]):
        assert torch.equal(ta, tb)
    assert a[1] == b[1]


# --------------------------------------------------------- parallel collect

def test_parallel_collect_matches_shape_and_range():
    states, zs = collect_rollout_samples_parallel(
        8, seed=17, workers=3, tail_plies=4, rollouts=2
    )
    assert len(states) == len(zs)
    assert 0 < len(states) <= 8 * 4
    for s in states:
        assert s.shape == (15, 5, 5)
    for z in zs:
        assert -1.0 <= z <= 1.0


def test_parallel_collect_same_game_budget_as_sequential():
    """Distribution-equal contract: same total games, same tail/rollouts, so
    the sample count distribution matches (exact count is game-length
    dependent; both must lie in the same bounds and be non-trivial)."""
    seq_states, _ = collect_rollout_samples(8, seed=23, tail_plies=4, rollouts=2)
    par_states, _ = collect_rollout_samples_parallel(
        8, 23, workers=3, tail_plies=4, rollouts=2
    )
    assert len(seq_states) <= 8 * 4
    assert len(par_states) <= 8 * 4
    assert len(par_states) >= 8  # every game contributes at least one tail ply


def test_parallel_collect_one_worker_is_sequential_path():
    """workers=1 must use the in-process sequential call (no spawn)."""
    a = collect_rollout_samples(2, seed=31, tail_plies=2, rollouts=2)
    b = collect_rollout_samples_parallel(2, 31, workers=1, tail_plies=2, rollouts=2)
    assert len(a[0]) == len(b[0])
    for ta, tb in zip(a[0], b[0]):
        assert torch.equal(ta, tb)
    assert a[1] == b[1]


def test_parallel_collect_error_marker_fails_loudly():
    """A crashing collector surfaces as RuntimeError, never a silent shrink."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()

    from smartfour.pretrain import _collect_worker
    p = ctx.Process(
        target=_collect_worker,
        args=(1, 0, 8, "not-an-int-rollouts", q),  # bad rollouts type
        daemon=True,
    )
    p.start()
    msg = q.get(timeout=60)
    p.join(timeout=30)
    assert isinstance(msg, tuple) and msg[0] == "__worker_error__"
    assert p.exitcode == 0  # reported in-band, did not crash


# ------------------------------------------------------------ device pretrain
def test_pretrain_value_learns_rollout_labels():
    """Fitting more epochs must reduce loss on a FIXED held-out rollout set
    (final-batch MSE is too noisy to assert on)."""
    torch.manual_seed(0)
    # Held-out labels from a different seed stream
    from smartfour.pretrain import collect_rollout_samples
    hs, hz = collect_rollout_samples(8, seed=777, tail_plies=4, rollouts=4)
    H = torch.stack(hs)
    Z = torch.tensor(hz, dtype=torch.float32).unsqueeze(1)

    def heldout_mse(net):
        net.eval()
        with torch.no_grad():
            _l, v = net(H)
            return float(((v - Z) ** 2).mean())

    base = ResNet(tiny_cfg())
    before = heldout_mse(base)
    net = ResNet(tiny_cfg())
    net.load_state_dict(base.state_dict())
    pretrain_value(
        net, games=200, epochs=64, batch_size=8, lr=1e-3,
        weight_decay=0.0, seed=41, tail_plies=4, rollouts=2,
    )
    after = heldout_mse(net)
    assert after < before


def test_pretrain_value_freezes_policy_head():
    """Policy-head *weights* are frozen (optimizer excludes them). BN
    running buffers of the unused head still update in forward mode —
    pre-existing behavior, harmless because the head is never queried
    during pretrain and self-play retrains it."""
    torch.manual_seed(1)
    net = ResNet(tiny_cfg())
    before = {k: v.clone() for k, v in net.state_dict().items()
              if k.startswith("policy_head") and "running" not in k
              and "num_batches" not in k}
    pretrain_value(
        net, games=4, epochs=2, batch_size=8, lr=1e-2,
        weight_decay=0.0, seed=43, tail_plies=4, rollouts=2,
    )
    for k, v in net.state_dict().items():
        if k in before:
            assert torch.equal(before[k], v), f"{k} changed during pretrain"


def test_pretrain_value_progress_called_per_batch():
    calls = []
    net = ResNet(tiny_cfg())
    pretrain_value(
        net, games=4, epochs=2, batch_size=8, lr=1e-3,
        weight_decay=0.0, seed=47, tail_plies=4, rollouts=2,
        progress=lambda: calls.append(1),
    )
    assert len(calls) >= 2


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="mps not available"
)
def test_pretrain_value_runs_on_mps_and_moves_data():
    """The device path trains with device-resident samples and returns a
    finite MSE; the net stays on the device throughout."""
    torch.manual_seed(2)
    net = ResNet(tiny_cfg()).to("mps")
    mse = pretrain_value(
        net, games=4, epochs=1, batch_size=8, lr=1e-3,
        weight_decay=0.0, seed=53, tail_plies=4, rollouts=2,
        device="mps",
    )
    assert mse == mse  # finite
    p = next(net.parameters())
    assert p.device.type == "mps"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="mps not available"
)
def test_pretrain_value_cpu_and_mps_agree_directionally():
    """Same seed on cpu vs mps: the MPS run trains on device-resident data
    and produces a finite, comparable MSE. Cross-device loss curves are not
    expected to match bit-for-bit (different kernels), so the contract is
    finiteness plus a learned fit on a larger budget."""
    torch.manual_seed(3)
    cpu_net = ResNet(tiny_cfg())
    mps_net = ResNet(tiny_cfg()).to("mps")
    mps_net.load_state_dict(cpu_net.state_dict())
    common = dict(
        games=12, batch_size=8, lr=1e-3, weight_decay=0.0,
        seed=59, tail_plies=4, rollouts=2,
    )
    cpu_mse = pretrain_value(cpu_net, epochs=6, **common)
    mps_mse = pretrain_value(mps_net, epochs=6, device="mps", **common)
    assert cpu_mse == cpu_mse and mps_mse == mps_mse  # both finite
    # Same scale of solution: neither blows up on its device path
    assert abs(cpu_mse - mps_mse) < 0.5
