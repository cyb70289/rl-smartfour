"""Tests for smartfour.inference_server and smartfour.device.

The server tests use the real spawned process (small net, CPU device) so the
protocol, batching, weight updates, and lifecycle are exercised end to end.
The cuda paths skip when CUDA is absent (they self-verify on a CUDA box).
"""

import multiprocessing as mp

import pytest
import torch

from smartfour.config import NetworkConfig
from smartfour.device import resolve_device, synchronize, VALID_DEVICES
from smartfour.game import apply_move, initial_state
from smartfour.inference_server import (
    InferenceServerHandle,
    RemoteEvaluator,
)
from smartfour.network import ResNet

TIMEOUT = 120


def tiny_cfg(**kw):
    kw.setdefault("input_channels", 16)
    kw.setdefault("blocks", 1)
    kw.setdefault("base_channels", 8)
    kw.setdefault("policy_channels", 4)
    kw.setdefault("value_channels", 4)
    kw.setdefault("value_fc", 8)
    return NetworkConfig(**kw)


@pytest.fixture(scope="module")
def server():
    cfg = tiny_cfg()
    net = ResNet(cfg)
    net.eval()
    state = {k: v.cpu() for k, v in net.state_dict().items()}
    h = InferenceServerHandle(cfg, "cpu", slots=2)
    h.start(initial_states=[state, state])
    yield h
    h.shutdown()


# ------------------------------------------------------------------ device

def test_resolve_device_auto_prefers_accelerator():
    d = resolve_device("auto")
    assert d in ("cpu", "mps", "cuda")
    if torch.cuda.is_available():
        assert d == "cuda"
    elif torch.backends.mps.is_available():
        assert d == "mps"


def test_resolve_device_explicit_passthrough():
    assert resolve_device("cpu") == "cpu"
    if torch.backends.mps.is_available():
        assert resolve_device("mps") == "mps"


def test_resolve_device_rejects_unknown():
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("tpu")


def test_resolve_device_rejects_absent_cuda():
    if torch.cuda.is_available():
        pytest.skip("cuda present; absence path untestable here")
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")


def test_synchronize_noop_on_cpu():
    synchronize("cpu")  # must not raise


def test_valid_devices_complete():
    assert VALID_DEVICES == ("auto", "cpu", "mps", "cuda")


# ----------------------------------------------------------------- protocol

def test_round_trip_shapes_and_legality(server):
    ev = RemoteEvaluator(server.address, slot=0)
    try:
        s0 = initial_state()
        s1 = apply_move(s0, 2, 2)
        priors, values = ev([s0, s1])
        assert priors.shape == (2, 125)
        assert values.shape == (2,)
        assert torch.allclose(priors.sum(1), torch.ones(2), atol=1e-5)
        from smartfour.encode import legal_actions
        for i, s in enumerate((s0, s1)):
            legal = set(legal_actions(s))
            nz = set((priors[i] > 0).nonzero(as_tuple=False).squeeze(1).tolist())
            assert nz == legal
    finally:
        ev.close()


def test_slot_isolation(server):
    """Slot 0 and slot 1 hold different nets after a slot-1 update."""
    cfg = tiny_cfg()
    net2 = ResNet(cfg)
    with torch.no_grad():
        for p in net2.parameters():
            p.add_(1.0)
    server.set_weights(1, net2.state_dict())
    ev0 = RemoteEvaluator(server.address, slot=0)
    ev1 = RemoteEvaluator(server.address, slot=1)
    try:
        s = initial_state()
        _p0, v0 = ev0([s])
        _p1, v1 = ev1([s])
        assert not torch.allclose(v0, v1)
    finally:
        ev0.close()
        ev1.close()


def test_set_weights_changes_outputs(server):
    cfg = tiny_cfg()
    net2 = ResNet(cfg)
    with torch.no_grad():
        for p in net2.parameters():
            p.mul_(2.0).add_(0.5)
    ev = RemoteEvaluator(server.address, slot=0)
    try:
        s = initial_state()
        _p, v_before = ev([s])
        server.set_weights(0, net2.state_dict())
        _p, v_after = ev([s])
        assert not torch.allclose(v_before, v_after)
    finally:
        ev.close()


def test_concurrent_clients_batch_together(server):
    """Several simultaneous clients all get correct-shape replies."""
    evs = [RemoteEvaluator(server.address, slot=i % 2) for i in range(4)]
    try:
        states = [initial_state()] * 8
        results = [ev(states) for ev in evs]
        for priors, values in results:
            assert priors.shape == (8, 125)
            assert values.shape == (8,)
            assert torch.allclose(priors.sum(1), torch.ones(8), atol=1e-5)
    finally:
        for ev in evs:
            ev.close()


def test_remote_evaluator_matches_local(server):
    """Server priors/values equal a local forward with the same weights."""
    cfg = tiny_cfg()
    net = ResNet(cfg)
    ev = RemoteEvaluator(server.address, slot=0)
    try:
        server.set_weights(0, net.state_dict())
        from smartfour.encode import encode
        from smartfour.mcts import LocalEvaluator
        local = LocalEvaluator(net)
        states = [initial_state(), apply_move(initial_state(), 1, 1)]
        lp, lv = local(states)
        rp, rv = ev(states)
        assert torch.allclose(lp, rp, atol=1e-4)
        assert torch.allclose(lv, rv, atol=1e-4)
    finally:
        ev.close()


# ------------------------------------------------------------- portability

def test_mps_checkpoint_round_trips_through_cpu():
    """A state_dict produced on an accelerator device loads on CPU.

    Uses mps when available (the real cross-device case); otherwise cpu.
    """
    cfg = tiny_cfg()
    src = "mps" if torch.backends.mps.is_available() else "cpu"
    net = ResNet(cfg).to(src)
    state = {k: v.cpu() for k, v in net.state_dict().items()}
    host = ResNet(cfg)
    host.load_state_dict(state)  # loads onto cpu without error
    x = torch.randn(2, 16, 5, 5)
    with torch.no_grad():
        _l, _v = host(x)
    assert True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_cuda_smoke():
    """Self-verifying on a CUDA box: server + client round trip on cuda."""
    cfg = tiny_cfg()
    net = ResNet(cfg)
    state = {k: v.cpu() for k, v in net.state_dict().items()}
    h = InferenceServerHandle(cfg, "cuda", slots=1)
    try:
        h.start(initial_states=[state])
        ev = RemoteEvaluator(h.address, slot=0)
        try:
            priors, values = ev([initial_state()])
            assert priors.shape == (1, 125)
            assert values.shape == (1,)
            assert torch.allclose(priors.sum(1), torch.ones(1), atol=1e-4)
        finally:
            ev.close()
    finally:
        h.shutdown()
