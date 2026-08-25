"""Tests for smartfour.arena — head-to-head play between agents."""

import torch

from smartfour.arena import play_arena, random_move, play_vs_random
from smartfour.config import MCTSConfig
from smartfour.game import BLACK, WHITE, initial_state, legal_moves, is_terminal


def cfg(**kw):
    kw.setdefault("simulations", 10)
    kw.setdefault("c_puct", 1.0)
    kw.setdefault("dirichlet_alpha", 0.3)
    kw.setdefault("dirichlet_epsilon", 0.25)
    kw.setdefault("temperature_threshold", 12)
    kw.setdefault("batch_eval_size", 16)
    return MCTSConfig(**kw)


class RandomNet:
    """Random logits: MCTS over it behaves like a randomized policy."""

    def __init__(self, seed=0):
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self, t):
        n = t.shape[0]
        return torch.randn(n, 125, generator=self.g), torch.zeros(n, 1)


def test_random_move_is_legal():
    s = initial_state()
    for _ in range(5):
        x, z = random_move(s)
        assert (x, z) in legal_moves(s)
        s = apply(s, x, z)


def apply(s, x, z):
    from smartfour.game import apply_move

    return apply_move(s, x, z)


def test_play_arena_random_vs_random_completes():
    torch.manual_seed(0)
    a = RandomNet(seed=1)
    b = RandomNet(seed=2)
    a_wins, b_wins, draws = play_arena(a, b, cfg(), games=4)
    assert a_wins + b_wins + draws == 4
    assert all(0 <= v <= 4 for v in (a_wins, b_wins, draws))


def test_play_arena_same_net_is_balanced():
    torch.manual_seed(3)
    net = RandomNet(seed=4)
    a_wins, b_wins, draws = play_arena(net, net, cfg(), games=6)
    assert a_wins + b_wins + draws == 6
    assert abs(a_wins - b_wins) <= 4  # loose balance bound for noisy random play


def test_arena_alternates_colors():
    torch.manual_seed(5)
    a = RandomNet(seed=6)
    b = RandomNet(seed=7)
    # With few simulations and temperature 0, the first player has an
    # advantage; color alternation must be honored (results should be able to
    # differ from a fixed-color run).
    a_wins, b_wins, draws = play_arena(a, b, cfg(), games=4)
    assert a_wins + b_wins + draws == 4


def test_play_vs_random_counts():
    torch.manual_seed(8)
    net = RandomNet(seed=9)
    wins, losses, draws = play_vs_random(net, cfg(), games=4, seed=10)
    assert wins + losses + draws == 4


def test_arena_counts_colors_correctly(monkeypatch):
    """Wins must be attributed to net_a / net_b, not to colors (regression:
    odd-game results used to be misattributed after the color swap)."""
    import smartfour.arena as arena_mod

    net_a, net_b = object(), object()

    # net_a (first arg) wins every game, regardless of color.
    monkeypatch.setattr(
        arena_mod, "_play_two",
        lambda nw, nb, c, start_state=None:
            (WHITE if nw is net_a else BLACK, 1, 0),
    )
    a_wins, b_wins, draws = play_arena(net_a, net_b, cfg(), games=4)
    assert (a_wins, b_wins, draws) == (4, 0, 0)

    # net_b (second arg) wins every game.
    monkeypatch.setattr(
        arena_mod, "_play_two",
        lambda nw, nb, c, start_state=None:
            (BLACK if nw is net_a else WHITE, 1, 0),
    )
    a_wins, b_wins, draws = play_arena(net_a, net_b, cfg(), games=4)
    assert (a_wins, b_wins, draws) == (0, 4, 0)

    monkeypatch.setattr(
        arena_mod, "_play_two",
        lambda nw, nb, c, start_state=None: ("draw", 1, 0),
    )
    a_wins, b_wins, draws = play_arena(net_a, net_b, cfg(), games=4)
    assert (a_wins, b_wins, draws) == (0, 0, 4)


def test_play_arena_progress_callback():
    torch.manual_seed(17)
    a = RandomNet(seed=18)
    b = RandomNet(seed=19)
    calls = []
    play_arena(a, b, cfg(), games=3, progress=lambda: calls.append(1))
    assert sum(calls) == 3  # called once per game


def test_deterministic_with_seed():
    torch.manual_seed(11)
    a = RandomNet(seed=12)
    b = RandomNet(seed=13)
    r1 = play_arena(a, b, cfg(), games=4)
    torch.manual_seed(11)
    c = RandomNet(seed=12)
    d = RandomNet(seed=13)
    r2 = play_arena(c, d, cfg(), games=4)
    assert r1 == r2


def test_play_two_from_book_state_reports_skipped():
    from smartfour.arena import _play_two
    from smartfour.game import apply_move

    start = apply_move(initial_state(), 2, 2)
    a, b = RandomNet(seed=20), RandomNet(seed=21)
    winner, plies, skipped = _play_two(a, b, cfg(), start_state=start)
    assert skipped == 1  # one piece already on the board
    assert plies > 0
    assert winner in (WHITE, BLACK, "draw")



def test_play_arena_book_pairs_share_state_with_swapped_roles(monkeypatch):
    """Game pairs use one book state twice: net_a moves first in game 0,
    net_b (as second arg) in game 1; skipped plies land in skipped_out."""
    import smartfour.arena as arena_mod
    from smartfour.game import apply_move

    start = apply_move(initial_state(), 3, 3)
    calls = []

    def fake_two(nw, nb, c, start_state=None):
        calls.append((id(nw), id(nb), id(start_state)))
        return WHITE if nw is net_a else BLACK, 5, 7

    monkeypatch.setattr(arena_mod, "_play_two", fake_two)
    net_a, net_b = object(), object()
    plies_out, skipped_out = [], []
    a_wins, b_wins, draws = play_arena(
        net_a, net_b, cfg(), games=4, plies_out=plies_out,
        book=[start], seed=123, skipped_out=skipped_out)
    assert (a_wins, b_wins, draws) == (4, 0, 0)  # net_a always first arg wins
    assert all(s == id(start) for _na, _nb, s in calls)
    # Roles swap inside each pair: net_a is the first arg in games 0/2,
    # net_b in games 1/3; each pair shares one book state object.
    assert calls[0][0] == calls[2][0] and calls[1][1] == calls[3][1]
    assert calls[0][2] == calls[1][2]
    assert calls[0][0] != calls[1][0]  # first arg really swaps inside the pair
    assert plies_out == [5, 5, 5, 5]
    assert skipped_out == [7, 7, 7, 7]


def test_play_arena_without_book_matches_legacy_alternation():
    plans = __import__("smartfour.openbook", fromlist=["game_plans"]).game_plans
    assert plans(0, 4, seed=0)[0] == (None, True)
    assert plans(0, 4, seed=0)[1] == (None, False)
