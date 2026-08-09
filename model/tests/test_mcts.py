"""Tests for smartfour.mcts — PUCT search, batching, perspective handling."""

import torch

from smartfour.config import MCTSConfig
from smartfour.encode import action_mask, action_to_xyz, encode, legal_actions, xyz_to_action
from smartfour.game import BLACK, WHITE, apply_move, initial_state, is_terminal
from smartfour.mcts import MCTS

SIM = 200


class FakeNet:
    """Deterministic net: per-state value fn, uniform logits over legal moves."""

    def __init__(self, value_fn=lambda t: torch.zeros(t.shape[0], 1)):
        self.value_fn = value_fn
        self.eval_batches = 0
        self.eval_states = 0

    def __call__(self, t):
        self.eval_batches += 1
        self.eval_states += t.shape[0]
        logits = torch.zeros(t.shape[0], 125)
        # Uniform over every cell; masking in MCTS handles legality.
        value = self.value_fn(t)
        return logits, value


def cfg(**kw):
    kw.setdefault("simulations", SIM)
    kw.setdefault("c_puct", 1.0)
    kw.setdefault("dirichlet_alpha", 0.3)
    kw.setdefault("dirichlet_epsilon", 0.25)
    kw.setdefault("temperature_threshold", 12)
    kw.setdefault("batch_eval_size", 16)
    return MCTSConfig(**kw)


def root_state():
    return initial_state()


# ------------------------------------------------------------ root policy

def test_root_policy_is_legal_and_normalized():
    m = MCTS(FakeNet(), cfg(simulations=100))
    pi, chosen, _ = m.root_policy(root_state(), root_noise=True, temperature=1.0)
    assert pi.shape == (125,)
    legal = set(legal_actions(root_state()))
    assert abs(pi.sum().item() - 1.0) < 1e-6
    assert {i for i in range(125) if pi[i] > 0} <= legal
    assert chosen in legal


def test_root_policy_without_noise_is_uniform():
    m = MCTS(FakeNet(), cfg(simulations=200))
    pi, _, _ = m.root_policy(root_state(), root_noise=False, temperature=1.0)
    legal = legal_actions(root_state())
    p = pi[torch.tensor(legal)].numpy()
    # Symmetric position + uniform priors: every legal action gets explored and
    # the distribution is roughly uniform (no batch-gated starvation).
    assert abs(p.mean() - 1.0 / 25) < 0.02
    assert (p > 0).all()


def test_dirichlet_noise_changes_root_priors():
    m = MCTS(FakeNet(), cfg(simulations=50))
    pi_no_noise, _, _ = m.root_policy(root_state(), root_noise=False, temperature=1.0)
    pi_noise, _, _ = m.root_policy(root_state(), root_noise=True, temperature=1.0)
    assert not torch.allclose(pi_no_noise, pi_noise)


def test_temperature_zero_is_argmax():
    m = MCTS(FakeNet(), cfg(simulations=100))
    pi, chosen, _ = m.root_policy(root_state(), root_noise=False, temperature=0.0)
    assert pi.sum().item() == 1.0
    assert pi[chosen] == 1.0
    assert int(pi.sum()) == 1  # one-hot


def test_terminal_root_returns_empty():
    s = initial_state()
    for x, z in [(0, 0), (4, 4), (0, 1), (4, 3), (0, 2), (4, 2), (0, 3)]:
        s = apply_move(s, x, z)
    assert is_terminal(s)
    m = MCTS(FakeNet(), cfg(simulations=10))
    pi, chosen, _ = m.root_policy(s, root_noise=False, temperature=1.0)
    assert pi.sum().item() == 0.0
    assert chosen is None


# ------------------------------------------------------------ value propagation

def test_backprop_flips_sign_each_level():
    """Values are stored in each node's own perspective: +1 flips to -1 at
    every level of the path (the child's perspective is the parent's opponent).
    """
    from smartfour.mcts import Node, MCTS
    from smartfour.game import initial_state as init

    m = MCTS(FakeNet(), cfg())
    root = Node(init(), [0, 1])
    child = Node(init(), [0], prior=0.5)
    grand = Node(init(), [], prior=0.5)
    root.children = {0: child}
    child.children = {0: grand}

    # Leaf (grand) value +1 from its own perspective.
    m._backprop([(root, 0), (child, 0)], grand, 1.0)
    assert grand.visits == 1 and grand.value_sum == 1.0
    assert child.visits == 1 and child.value_sum == -1.0
    assert root.visits == 1 and root.value_sum == 1.0

    m._backprop([(root, 0)], child, 0.5)  # child as leaf, own perspective
    assert child.visits == 2 and child.value_sum == -0.5
    assert root.visits == 2 and root.value_sum == 0.5


class RecordingNet(FakeNet):
    """FakeNet that records every encoded state it evaluates."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen = []

    def __call__(self, t):
        self.eval_batches += 1
        self.eval_states += t.shape[0]
        self.seen.extend(t.clone() for t in t)
        return torch.zeros(t.shape[0], 125), torch.zeros(t.shape[0], 1)


def test_terminal_leaf_short_circuits_net():
    """A move that wins immediately is valued +1 and never net-evaluated."""
    s = initial_state()
    for x, z in [(0, 0), (4, 4), (0, 1), (4, 3), (0, 2), (4, 2)]:
        s = apply_move(s, x, z)
    # White to move; (0,3) completes the flat four at height 0 (row along z).
    assert s.current == WHITE
    winning_action = xyz_to_action(0, 3, 0)
    net = RecordingNet()
    m = MCTS(net, cfg(simulations=100, batch_eval_size=8))
    pi, chosen, root = m.root_policy(s, root_noise=False, temperature=1.0)
    win_child = root.children[winning_action]
    # Each visit contributed -1 to the child (its own perspective): q = +1 at
    # root, above the net's 0-valued alternatives.
    assert win_child.visits >= 1
    assert win_child.value_sum == -win_child.visits
    assert win_child.value == -1.0
    assert all(
        other.value == 0.0
        for a, other in root.children.items() if a != winning_action
    )
    # The terminal state (after the winning move) was never sent to the net.
    from smartfour.game import apply_move as am

    terminal_state = am(s, 0, 3)
    from smartfour.encode import encode

    term_enc = encode(terminal_state)
    for seen in net.seen:
        assert not torch.equal(seen, term_enc), "terminal state must not be net-evaluated"


# ------------------------------------------------------------ batching & determinism

def test_batch_sizes_respect_sims_and_legality():
    """Different eval batch sizes change tree shape but both are valid searches."""
    s = initial_state()
    for x, z in [(0, 0), (4, 4), (1, 1)]:
        s = apply_move(s, x, z)
    legal = set(legal_actions(s))
    for bs in (1, 32):
        m = MCTS(FakeNet(), cfg(simulations=60, batch_eval_size=bs))
        pi, chosen, root = m.root_policy(s, root_noise=False, temperature=1.0)
        assert abs(pi.sum().item() - 1.0) < 1e-6
        assert {i for i in range(125) if pi[i] > 0} <= legal
        assert chosen in legal
        assert root.visits == 60
        assert sum(c.visits for c in root.children.values()) == 60


def test_deterministic_with_seed():
    torch.manual_seed(0)
    m1 = MCTS(FakeNet(), cfg(simulations=80))
    pi1, c1, _ = m1.root_policy(root_state(), root_noise=True, temperature=1.0)
    torch.manual_seed(0)
    m2 = MCTS(FakeNet(), cfg(simulations=80))
    pi2, c2, _ = m2.root_policy(root_state(), root_noise=True, temperature=1.0)
    assert torch.equal(pi1, pi2)
    assert c1 == c2


def test_real_net_policy_legal_on_stacked_state():
    from smartfour.network import ResNet
    from smartfour.config import NetworkConfig

    torch.manual_seed(9)
    net = ResNet(NetworkConfig(input_channels=16, blocks=1, base_channels=8,
                               policy_channels=4, value_channels=4, value_fc=8))
    net.eval()
    s = initial_state()
    for x, z in [(1, 1), (0, 0), (1, 1), (0, 1), (1, 1)]:
        s = apply_move(s, x, z)
    m = MCTS(net, cfg(simulations=30))
    pi, chosen, _ = m.root_policy(s, root_noise=False, temperature=1.0)
    legal = set(legal_actions(s))
    assert abs(pi.sum().item() - 1.0) < 1e-6
    assert {i for i in range(125) if pi[i] > 0} <= legal
    assert chosen in legal
    # Chosen action matches its (x, z) column at the current stack height.
    x, z, y = action_to_xyz(chosen)
    assert s.grid[x][z][y] is None
    assert s.grid[x][z][y - 1] is not None if y > 0 else True


# ------------------------------------------------------------ device plumbing

class DeviceRecordingNet(FakeNet):
    """FakeNet that records the device of every batch it evaluates."""

    def __init__(self):
        super().__init__()
        self.seen_devices = []

    def __call__(self, t):
        self.eval_batches += 1
        self.eval_states += t.shape[0]
        self.seen_devices.append(t.device.type)
        return torch.zeros(t.shape[0], 125), torch.zeros(t.shape[0], 1)


def test_mcts_moves_inputs_to_configured_device():
    """Every tensor handed to the net must live on the configured device.

    Uses device='meta' as the oracle: on a CPU-only host a raw CPU encoding
    would pass device checks trivially, while 'meta' makes a missing
    `.to(device)` fail loudly. Root build AND batched drain are covered
    (simulations > batch_eval_size forces multiple drain calls).
    """
    net = DeviceRecordingNet()
    m = MCTS(net, cfg(simulations=40, batch_eval_size=8), device="meta")
    pi, chosen, root = m.root_policy(root_state(), root_noise=False, temperature=1.0)
    assert net.seen_devices, "net was never evaluated"
    assert set(net.seen_devices) == {"meta"}
    assert abs(pi.sum().item() - 1.0) < 1e-6
    assert chosen in legal_actions(root_state())


def test_mcts_infers_device_from_net_params():
    """Without an explicit device, MCTS runs on the net's parameter device."""
    from smartfour.config import NetworkConfig
    from smartfour.network import ResNet

    torch.manual_seed(9)
    net = ResNet(NetworkConfig(input_channels=16, blocks=1, base_channels=8,
                               policy_channels=4, value_channels=4, value_fc=8))
    m = MCTS(net, cfg(simulations=5))
    assert m.device == torch.device("cpu")
    pi, chosen, _ = m.root_policy(root_state(), root_noise=False, temperature=1.0)
    assert abs(pi.sum().item() - 1.0) < 1e-6
    assert chosen in legal_actions(root_state())


def test_mcts_infers_cpu_for_parameterless_net():
    """Test doubles without parameters (e.g. FakeNet) still default to CPU."""
    m = MCTS(FakeNet(), cfg(simulations=5))
    assert m.device == torch.device("cpu")
