"""Tests for the virtual-loss batched search mode (MCTS()).

The batched searcher shares PUCT math, priors, and the net with the
sequential searcher; what may differ is traversal order. These tests pin the
invariants that MUST hold regardless of order.
"""

import torch

from smartfour.config import MCTSConfig
from smartfour.encode import legal_actions
from smartfour.game import apply_move, initial_state, is_terminal, WHITE
from smartfour.mcts import MCTS, Node, _ChildStub


class UniformNet:
    """Uniform logits, zero value — same contract FakeNet has in test_mcts."""

    def __call__(self, t):
        return torch.zeros(t.shape[0], 125), torch.zeros(t.shape[0], 1)

    def eval(self):
        pass


def bcfg(**kw):
    kw.setdefault("simulations", 60)
    kw.setdefault("c_puct", 1.0)
    kw.setdefault("dirichlet_alpha", 0.3)
    kw.setdefault("dirichlet_epsilon", 0.25)
    kw.setdefault("temperature_threshold", 12)
    kw.setdefault("batch_eval_size", 16)
    return MCTSConfig(**kw)


def walk(root, fn):
    """Visit every materialized node in the tree."""
    fn(root)
    if root.children:
        for a, child in root.children.items():
            if not isinstance(child, _ChildStub):
                walk(child, fn)


def test_batched_produces_legal_normalized_policy():
    m = MCTS(UniformNet(), bcfg())
    pi, chosen, root = m.root_policy(initial_state())
    legal = legal_actions(initial_state())
    assert chosen in legal
    assert abs(pi.sum().item() - 1.0) < 1e-6
    assert all(pi[a] >= 0 for a in range(125))


def test_batched_conserves_visits():
    """Sum of root-child visits == sims_done; no visit lost or duplicated."""
    m = MCTS(UniformNet(), bcfg(simulations=100))
    _pi, _c, root = m.root_policy(initial_state(), root_noise=False)
    total = sum(c.visits for c in root.children.values())
    assert total == m.last_stats["sims_done"] == 100


def test_batched_leaves_no_pending_after_search():
    """Every virtual-loss charge is reconciled (pending == 0 on all nodes)."""
    m = MCTS(UniformNet(), bcfg(simulations=100, batch_eval_size=8))
    _pi, _c, root = m.root_policy(initial_state(), root_noise=False)
    bad = []
    walk(root, lambda n: bad.append(n) if getattr(n, "pending", 0) != 0 else None)
    assert not bad, f"{len(bad)} node(s) still carry pending penalties"


def test_batched_values_stay_bounded():
    """Backpropagated values keep the node mean in [-1, 1] (terminal bounds)."""
    m = MCTS(UniformNet(), bcfg(simulations=100))
    _pi, _c, root = m.root_policy(initial_state(), root_noise=False)
    bad = []

    def check(n):
        if n.visits:
            v = n.value_sum / n.visits
            if not (-1.001 <= v <= 1.001):
                bad.append((v, n.visits))

    walk(root, check)
    assert not bad


def test_batched_finds_immediate_win():
    """White has three in a row on z=0 (x=0..2, y=0); (3,0) completes it.
    The searcher must visit the winning move most at any sane budget."""
    s = initial_state()
    # White builds (0,0),(1,0),(2,0) while black plays far away at z=4.
    for x in range(3):
        s = apply_move(s, x, 0)      # white
        s = apply_move(s, x, 4)      # black
    m = MCTS(UniformNet(), bcfg(simulations=50))
    _pi, chosen, root = m.root_policy(s, root_noise=False)
    # With a uniform net the value head says nothing; terminal detection must
    # still make the winning column the most-visited root child.
    from smartfour.encode import xyz_to_action
    win_a = xyz_to_action(3, 0, 0)
    assert root.children[win_a].visits == max(
        c.visits for a, c in root.children.items() if not isinstance(c, _ChildStub)
    )


def test_batched_deterministic_with_seed():
    torch.manual_seed(5)
    a = MCTS(UniformNet(), bcfg(simulations=60)).root_policy(
        initial_state(), root_noise=False
    )[0]
    torch.manual_seed(5)
    b = MCTS(UniformNet(), bcfg(simulations=60)).root_policy(
        initial_state(), root_noise=False
    )[0]
    assert torch.equal(a, b)


def test_batched_bigger_batches_fewer_forwards():
    """The point of the mode: same sims, larger pass target, fewer forwards."""
    stats = {}
    for bes in (8, 64):
        m = MCTS(UniformNet(), bcfg(simulations=200, batch_eval_size=bes))
        m.root_policy(initial_state(), root_noise=False)
        stats[bes] = (m.last_stats["net_forwards"], m.last_stats["batch_size_mean"])
    assert stats[64][0] < stats[8][0]
    assert stats[64][1] > stats[8][1]


def test_batched_terminal_root_and_empty_policy():
    s = initial_state()
    for x, z in [(0, 0), (4, 4), (0, 1), (4, 3), (0, 2), (4, 2), (0, 3)]:
        s = apply_move(s, x, z)
    assert is_terminal(s)
    m = MCTS(UniformNet(), bcfg())
    pi, chosen, root = m.root_policy(s)
    assert chosen is None and root is None
    assert float(pi.sum()) == 0.0


def test_batched_plays_a_full_game():
    """End-to-end: batched searcher drives a whole game to termination."""
    torch.manual_seed(1)
    state = initial_state()
    m = MCTS(UniformNet(), bcfg(simulations=30))
    plies = 0
    while not is_terminal(state):
        _pi, chosen, _root = m.root_policy(state, root_noise=False, temperature=0.0)
        assert chosen is not None
        from smartfour.encode import action_to_xyz
        x, z, _y = action_to_xyz(chosen)
        state = apply_move(state, x, z)
        plies += 1
        assert plies < 70
    assert state.winner is not None


def test_stats_shape():
    """last_stats keys are the documented diagnostics contract."""
    m = MCTS(UniformNet(), bcfg(simulations=40))
    pi, chosen, root = m.root_policy(initial_state(), root_noise=False)
    assert chosen is not None
    assert set(m.last_stats) == {
        "sims", "depth_mean", "max_depth", "sims_done", "leaf_distinct",
        "terminal_hits", "nodes", "net_forwards", "batch_size_mean",
        "n_states", "node_hashes", "root_value",
        "root_policy_entropy", "root_entropy", "root_width", "chosen_prob",
    }
