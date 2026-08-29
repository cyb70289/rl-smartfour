"""Tests for smartfour.network — resnet policy/value heads and training loss."""

import math

import pytest
import torch

from smartfour.config import NetworkConfig
from smartfour.encode import action_mask, encode
from smartfour.game import apply_move, initial_state
from smartfour.network import ResNet, loss_fn

CONF = NetworkConfig(input_channels=15, blocks=2, base_channels=16,
                     policy_channels=8, value_channels=8, value_fc=16)


def make_net(**kw):
    kw.setdefault("input_channels", 15)
    kw.setdefault("blocks", 2)
    kw.setdefault("base_channels", 16)
    kw.setdefault("policy_channels", 8)
    kw.setdefault("value_channels", 8)
    kw.setdefault("value_fc", 16)
    return ResNet(NetworkConfig(**kw))


def test_output_shapes_and_value_range():
    torch.manual_seed(0)
    net = make_net()
    s = apply_move(initial_state(), 1, 1)
    t = encode(s).unsqueeze(0)
    logits, value = net(t)
    assert logits.shape == (1, 125)
    assert value.shape == (1, 1)
    assert value.abs().max().item() <= 1.0 + 1e-6


def test_batch_forward():
    torch.manual_seed(1)
    net = make_net()
    batch = torch.zeros(4, 15, 5, 5)
    logits, value = net(batch)
    assert logits.shape == (4, 125)
    assert value.shape == (4, 1)


def test_deterministic_with_seed():
    torch.manual_seed(42)
    a = make_net()
    torch.manual_seed(42)
    b = make_net()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    t = torch.randn(2, 15, 5, 5)
    with torch.no_grad():
        la, va = a(t)
        lb, vb = b(t)
    assert torch.equal(la, lb) and torch.equal(va, vb)


def test_different_seeds_differ():
    torch.manual_seed(0)
    a = make_net()
    torch.manual_seed(1)
    b = make_net()
    t = torch.randn(1, 15, 5, 5)
    with torch.no_grad():
        assert not torch.equal(a(t)[0], b(t)[0])


def test_policy_loss_uniform_logits():
    """With uniform logits, cross-entropy against a one-hot target = log(125)."""
    net = make_net()
    logits = torch.zeros(1, 125)
    value = torch.zeros(1, 1)
    pi = torch.zeros(1, 125)
    pi[0, 0] = 1.0
    z = torch.zeros(1, 1)
    loss = loss_fn(logits, value, pi, z)
    assert math.isclose(loss.item(), math.log(125), rel_tol=1e-6)


def test_value_loss_reduces_on_fixed_task():
    """Gradient descent drives the value head toward a fixed target (z=1)."""
    torch.manual_seed(3)
    net = make_net()
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    states = [apply_move(initial_state(), x, z) for x, z in [(0, 0), (1, 1), (2, 2)]]
    t = torch.stack([encode(s) for s in states])
    pi = torch.full((3, 125), 1 / 125)
    z = torch.ones(3, 1)

    with torch.no_grad():
        _, v0 = net(t)
        baseline = float(loss_fn(net(t)[0], net(t)[1], pi, z).item())

    for _ in range(300):
        opt.zero_grad()
        logits, value = net(t)
        loss = loss_fn(logits, value, pi, z)
        loss.backward()
        opt.step()

    with torch.no_grad():
        _, v1 = net(t)
        final = float(loss_fn(net(t)[0], net(t)[1], pi, z).item())
    assert final < baseline  # value loss shrinks (policy loss floors at log 125)
    assert v1.mean().item() > 0.9  # value pushed toward +1


def test_net_accepts_masked_policy_target():
    """Training on a real encoded position with its legal-move target works."""
    torch.manual_seed(5)
    net = make_net()
    s = apply_move(initial_state(), 0, 0)
    t = encode(s).unsqueeze(0)
    mask = action_mask(s)
    pi = mask / mask.sum()
    z = torch.tensor([[1.0]])
    logits, value = net(t)
    loss = loss_fn(logits, value, pi.unsqueeze(0), z)
    assert torch.isfinite(loss)


def test_forward_grad_flows():
    torch.manual_seed(7)
    net = make_net()
    t = torch.randn(2, 15, 5, 5)
    logits, value = net(t)
    loss = logits.sum() + value.sum()
    loss.backward()
    for p in net.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
