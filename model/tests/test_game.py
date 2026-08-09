"""Tests for smartfour.game — mirrors ui/tests/rules.test.ts semantics."""

import itertools

import pytest

from smartfour.game import (
    BOARD_SIZE,
    STACK_HEIGHT,
    BLACK,
    WHITE,
    GameState,
    IllegalMoveError,
    apply_move,
    find_win_run,
    initial_state,
    is_legal,
    is_terminal,
    legal_moves,
    stack_height,
    terminal_value,
)

# Canonical line directions as [dx, dz, dy] with dy >= 0:
# 4 flat + 1 vertical + 8 rising diagonals.
DIRS = [
    (1, 0, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 0, -1),
    (0, 1, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (1, 1, -1),
    (-1, 1, 0),
    (0, 1, -1),
    (-1, 1, -1),
    (-1, 1, 1),
]


def all_win_lines():
    """All (dir, cells) whose 4 cells stay inside the 5x5x5 cube."""
    lines = []
    for dir_ in DIRS:
        for x, z, y in itertools.product(range(BOARD_SIZE), repeat=3):
            cells = []
            ok = True
            for i in range(4):
                cx = x + dir_[0] * i
                cz = z + dir_[1] * i
                cy = y + dir_[2] * i
                if not (0 <= cx < BOARD_SIZE and 0 <= cz < BOARD_SIZE and 0 <= cy < STACK_HEIGHT):
                    ok = False
                    break
                cells.append((cx, cz, cy))
            if ok:
                lines.append((dir_, cells))
    return lines


def empty_grid():
    return [[[None for _ in range(STACK_HEIGHT)] for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def play_line(cells):
    """White fills `cells` in order; black fills a disjoint column between."""
    cols = {(x, z) for x, z, _y in cells}
    filler = next(
        (fx, fz)
        for fx in range(BOARD_SIZE)
        for fz in range(BOARD_SIZE)
        if (fx, fz) not in cols
    )
    state = initial_state()
    for i, (x, z, _y) in enumerate(cells):
        state = apply_move(state, x, z)
        if i < len(cells) - 1:
            state = apply_move(state, *filler)
    return state


# ---------------------------------------------------------------- initial state

def test_initial_state():
    s = initial_state()
    assert s.grid == empty_grid()
    assert s.pieces_left == {WHITE: 32, BLACK: 32}
    assert s.current == WHITE
    assert not is_terminal(s)
    assert legal_moves(s) == [(x, z) for x in range(5) for z in range(5)]


# ------------------------------------------------- placement and stacking

def test_place_lands_on_stack_top():
    s = initial_state()
    s = apply_move(s, 2, 3)          # white
    s = apply_move(s, 0, 0)          # black elsewhere
    s = apply_move(s, 2, 3)          # white stacks
    assert s.grid[2][3][0] == WHITE
    assert s.grid[2][3][1] == WHITE
    assert stack_height(s.grid, 2, 3) == 2
    assert s.current == BLACK
    assert s.pieces_left[WHITE] == 30


def test_full_column_is_illegal():
    s = initial_state()
    # Alternate players in the same column: W,B,W,B,W — never a same-color run.
    for _ in range(5):
        s = apply_move(s, 0, 0)
    assert stack_height(s.grid, 0, 0) == 5
    assert s.winner is None
    assert not is_legal(s, 0, 0)
    assert (0, 0) not in legal_moves(s)
    # A different column is still legal.
    assert is_legal(s, 1, 1)


def test_illegal_moves_raise():
    s = initial_state()
    with pytest.raises(IllegalMoveError):
        apply_move(s, -1, 0)
    with pytest.raises(IllegalMoveError):
        apply_move(s, 5, 0)
    s = apply_move(s, 0, 0)
    # Opponent may stack on the same column — that is legal.
    s2 = apply_move(s, 0, 0)
    assert s2.grid[0][0][1] == BLACK


def test_no_moves_after_game_over():
    s = initial_state()
    for x, z in [(0, 0), (4, 4), (0, 1), (4, 3), (0, 2), (4, 2), (0, 3)]:
        s = apply_move(s, x, z)  # white four in a row (0,0..3); black filler
    assert is_terminal(s)
    assert s.winner == WHITE
    assert not is_legal(s, 1, 1)
    assert legal_moves(s) == []
    with pytest.raises(IllegalMoveError):
        apply_move(s, 1, 1)


# ------------------------------------------------- win detection: exhaustive

@pytest.mark.parametrize("dir_,cells", all_win_lines(), ids=lambda v: str(v[0]))
def test_find_win_exhaustive_all_3d_lines(dir_, cells):
    # White occupies the line; every such line must be a win when completed.
    state = play_line(cells)
    assert is_terminal(state)
    assert state.winner == WHITE


def test_five_in_a_row_is_a_win_run():
    # A run of 5 is detected as a winning run of length 5 (>= 4).
    grid = empty_grid()
    for x in range(5):
        grid[x][0][0] = WHITE
    run = find_win_run(grid, 2, 0, 0, WHITE)
    assert run == [(x, 0, 0) for x in range(5)]


def test_no_win_on_gap():
    # Three in a row then a gap: no win.
    cells = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    state = play_line(cells)
    assert state.winner is None
    # Fill the gap column but not the far cell: still no win.
    state = apply_move(state, 4, 0)  # white places at (4,0) — not contiguous
    assert state.winner is None


def test_vertical_win_in_stack():
    s = initial_state()
    # White fills (1,1) up to height 4; black plays the far corner between.
    for i in range(4):
        s = apply_move(s, 1, 1)
        if i < 3:
            s = apply_move(s, 4, 4)
    assert s.grid[1][1][3] == WHITE
    assert s.winner == WHITE


def test_win_only_for_mover():
    # Build a board where white already has a 4-line; black moving cannot win on it.
    cells = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
    state = play_line(cells)
    assert state.winner == WHITE


# ------------------------------------------------- win detection: applyMove integration

def test_win_sets_terminal_and_perspective_value():
    cells = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
    state = play_line(cells)
    assert state.winner == WHITE
    # Terminal state has black to move; from black's perspective value is -1.
    assert state.current == BLACK
    assert terminal_value(state) == -1.0
    assert is_terminal(state)


# ------------------------------------------------- draw

def test_draw_when_all_pieces_placed_without_winner():
    s = initial_state(1)  # one piece each: game ends on the 2nd move
    s = apply_move(s, 0, 0)
    assert s.winner is None
    s = apply_move(s, 0, 1)
    assert s.winner == "draw"
    assert legal_moves(s) == []
    assert terminal_value(s) == 0.0


def test_win_takes_precedence_over_draw_on_last_move():
    # 4 pieces each; black completes a vertical 4-stack on its 4th (final) move.
    s = initial_state(4)
    seq = [(4, 4), (0, 0), (3, 3), (0, 1), (4, 2), (0, 2), (3, 1), (0, 3)]
    for x, z in seq:
        s = apply_move(s, x, z)
    assert s.pieces_left == {WHITE: 0, BLACK: 0}
    assert s.winner == BLACK
    assert is_terminal(s)
    assert terminal_value(s) == -1.0  # white to move, black won


def test_refuses_moves_after_draw():
    s = initial_state(1)
    s = apply_move(s, 0, 0)
    s = apply_move(s, 0, 1)
    assert s.winner == "draw"
    with pytest.raises(IllegalMoveError):
        apply_move(s, 1, 0)
