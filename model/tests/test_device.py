"""Tests for smartfour.device — accelerator auto-detection and labels.

`resolve_device` picks the best available accelerator (CUDA > MPS > CPU) and
`device_name` renders a human-readable label for startup banners. Availability
is monkeypatched here because CI hosts may lack GPUs.
"""

from types import SimpleNamespace

import pytest
import torch

from smartfour.device import device_name, resolve_device, state_to_cpu


def _no_accel(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    mps = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setattr(torch.backends, "mps", mps)


# ---------------------------------------------------------------- resolution

def test_auto_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends, "mps", SimpleNamespace(is_available=lambda: True))
    assert resolve_device() == torch.device("cuda")


def test_auto_prefers_mps_when_cuda_absent(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", SimpleNamespace(is_available=lambda: True))
    assert resolve_device() == torch.device("mps")


def test_auto_falls_back_to_cpu(monkeypatch):
    _no_accel(monkeypatch)
    assert resolve_device() == torch.device("cpu")


def test_mps_backend_missing_is_treated_as_unavailable(monkeypatch):
    """Very old torch builds lack torch.backends.mps entirely."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delattr(torch.backends, "mps")
    assert resolve_device() == torch.device("cpu")


def test_explicit_preferred_device_is_honored(monkeypatch):
    """An explicit device bypasses availability checks (user override)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends, "mps", SimpleNamespace(is_available=lambda: True))
    assert resolve_device("cpu") == torch.device("cpu")
    _no_accel(monkeypatch)
    assert resolve_device("cuda") == torch.device("cuda")


# ---------------------------------------------------------------- labels

def test_device_name_cuda_includes_gpu_name(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _i: "NVIDIA GeForce RTX 3090")
    assert device_name("cuda") == "cuda (NVIDIA GeForce RTX 3090)"


def test_device_name_mps_and_cpu():
    assert device_name("mps") == "mps (Apple Silicon)"
    assert device_name("cpu") == "cpu"
    assert device_name(torch.device("cpu")) == "cpu"


# ---------------------------------------------------------------- state_to_cpu

class _FakeTensor:
    def __init__(self):
        self.detach_called = False
        self.cpu_called = False

    def detach(self):
        self.detach_called = True
        return self

    def cpu(self):
        self.cpu_called = True
        return self


def test_state_to_cpu_detaches_and_moves_every_value():
    fa, fb = _FakeTensor(), _FakeTensor()
    out = state_to_cpu({"a": fa, "b": fb})
    assert out == {"a": fa, "b": fb}
    assert fa.detach_called and fa.cpu_called
    assert fb.detach_called and fb.cpu_called


def test_state_to_cpu_recurses_into_optimizer_state():
    """Optimizer state is nested (state + param_groups): tensors deep inside
    must be moved, plain values passed through untouched."""
    state = {
        "state": {"0": {"step": torch.tensor(1), "exp_avg": torch.randn(2)}},
        "param_groups": [{"lr": 0.001}],
    }
    out = state_to_cpu(state)
    assert out["state"]["0"]["exp_avg"].device.type == "cpu"
    assert out["state"]["0"]["step"].device.type == "cpu"
    assert out["param_groups"][0]["lr"] == 0.001


def test_state_to_cpu_detaches_grad_tracking():
    t = torch.randn(2, requires_grad=True)
    out = state_to_cpu(t)
    assert not out.requires_grad
