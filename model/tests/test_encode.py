"""Tests for smartfour.encode — 15-channel current-player-perspective encoding."""

import random

import torch

from smartfour.encode import (
    N_CHANNELS,
    action_mask,
    action_to_xyz,
    apply_d4,
    apply_d4_policy,
    d4_perms,
    encode,
    legal_actions,
    xyz_to_action,
)
from smartfour.game import (
    BOARD_SIZE,
    STACK_HEIGHT,
    BLACK,
    WHITE,
    GameState,
    apply_move,
    empty_grid,
    initial_state,
    legal_moves,
)


def random_state(rng, n_pieces=(3, 10)):
    """A random non-terminal-ish state built by real moves from the start."""
    s = initial_state()
    moves = rng.sample([(x, z) for x in range(5) for z in range(5)], k=25) * 5
    for x, z in moves:
        if s.winner is not None:
            break
        s = apply_move(s, x, z)
        if sum(s.pieces_left.values()) <= 64 - rng.randint(*n_pieces):
            break
    return s


def transform_state(state, perm):
    """Apply D4 permutation `perm` to a game state (columns permuted).

    Convention matches apply_d4: out[j] = inp[perm[j]] — new cell at index j
    holds the old cell from index perm[j].
    """
    new_grid = empty_grid()
    for j in range(25):
        ox, oz = divmod(perm[j], BOARD_SIZE)
        nx, nz = divmod(j, BOARD_SIZE)
        new_grid[nx][nz] = state.grid[ox][oz][:]
    return GameState(
        grid=new_grid,
        pieces_left=dict(state.pieces_left),
        current=state.current,
        winner=state.winner,
    )


# ---------------------------------------------------------------- structure

def test_empty_initial_state_channels():
    t, mask = encode(initial_state()), action_mask(initial_state())
    assert t.shape == (N_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    # No pieces anywhere.
    assert t[:10].sum() == 0
    # Every column legal at plane 0.
    assert (t[10] == 1).all()
    assert t[11:15].sum() == 0
    # Mask: exactly the 25 ground-level actions.
    assert mask.shape == (125,)
    assert mask.sum() == 25
    legal = {xyz_to_action(x, z, 0) for x, z in legal_moves(initial_state())}
    for a in range(125):
        assert (mask[a] == 1.0) == (a in legal), a


def test_perspective_swap_after_white_move():
    s = apply_move(initial_state(), 2, 3)  # white plays, black to move
    t = encode(s)
    # White's piece is now the OPPONENT's (plane 5), at (2,3).
    assert t[5][2][3] == 1.0
    assert t[0].sum() == 0
    # Column (2,3) is legal at height 1; empty columns at height 0.
    assert t[10][2][3] == 0.0
    assert t[11][2][3] == 1.0
    assert t[10][0][0] == 1.0


def test_stacked_position_planes():
    # White at (0,0,0), black at (0,0,1), white to move again at (0,0,2).
    grid = empty_grid()
    grid[0][0][0] = WHITE
    grid[0][0][1] = BLACK
    grid[0][0][2] = WHITE
    s = GameState(grid=grid, pieces_left={WHITE: 29, BLACK: 31}, current=WHITE)
    t = encode(s)
    # Planes are t[height][x][z]: current player's pieces at heights 0 and 2.
    assert t[0][0][0] == 1.0 and t[2][0][0] == 1.0
    assert t[1][0][0] == 0.0
    # Opponent (black) piece at height 1 → plane 5+1, cell (0,0).
    assert t[6][0][0] == 1.0
    # Legality: column (0,0) is legal only at height 3 (top of stack).
    assert t[10][0][0] == 0.0 and t[11][0][0] == 0.0 and t[12][0][0] == 0.0
    assert t[13][0][0] == 1.0


def test_action_indexing_round_trip():
    for a in range(125):
        x, z, y = action_to_xyz(a)
        assert xyz_to_action(x, z, y) == a
        assert 0 <= x < 5 and 0 <= z < 5 and 0 <= y < 5
    assert xyz_to_action(2, 3, 4) == 4 * 25 + 2 * 5 + 3


def test_mask_matches_legal_moves():
    rng = random.Random(7)
    for _ in range(20):
        s = random_state(rng)
        mask = action_mask(s)
        expected = set()
        for x, z in legal_moves(s):
            expected.add(xyz_to_action(x, z, stack_height_of(s, x, z)))
        got = {a for a in range(125) if mask[a] == 1.0}
        assert got == expected


def stack_height_of(s, x, z):
    h = 0
    while h < STACK_HEIGHT and s.grid[x][z][h] is not None:
        h += 1
    return h


def test_legal_actions_indices():
    s = apply_move(initial_state(), 1, 1)
    acts = legal_actions(s)
    assert len(acts) == 25
    assert xyz_to_action(1, 1, 1) in acts  # stack on (1,1)
    assert xyz_to_action(1, 1, 0) not in acts


# ---------------------------------------------------------------- D4 symmetry

def test_d4_has_eight_transforms_and_identity():
    perms = d4_perms()
    assert len(perms) == 8
    assert perms[0] == list(range(25))
    # Each is a permutation of 0..24.
    for p in perms:
        assert sorted(p) == list(range(25))


def test_encode_is_d4_equivariant():
    """Encoding a transformed state equals transforming the encoding."""
    rng = random.Random(11)
    for _ in range(10):
        s = random_state(rng)
        t = encode(s)
        for perm in d4_perms():
            ts = transform_state(s, perm)
            assert torch.equal(apply_d4(t, perm), encode(ts)), perm
            # Legality mask permutes identically.
            assert torch.equal(apply_d4(action_mask(s).view(5, 5, 5), perm).view(-1), action_mask(ts))


def test_policy_permutation_keeps_distribution():
    rng = random.Random(3)
    s = random_state(rng)
    pi = torch.rand(125)
    pi = pi * action_mask(s)
    pi = pi / pi.sum()
    for perm in d4_perms():
        tp = apply_d4_policy(pi, perm)
        assert abs(tp.sum().item() - 1.0) < 1e-6
        # Support stays on the legal actions of the transformed state.
        legal_ts = {xyz_to_action(*action_to_xyz(a)) for a in legal_actions(transform_state(s, perm))}
        support = {int(a) for a in torch.nonzero(tp > 0).flatten().tolist()}
        assert support <= legal_ts


def test_composition_closed_under_group():
    perms = d4_perms()
    idx = {tuple(p): i for i, p in enumerate(perms)}
    for p in perms:
        for q in perms:
            composed = [p[q[j]] for j in range(25)]
            assert tuple(composed) in idx
