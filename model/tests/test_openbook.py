"""Tests for the opening book: format round-trip, strict parsing, data-block
rendering, and arena game plans."""

import random

import pytest

from smartfour import openbook
from smartfour.game import BLACK, WHITE, apply_move, initial_state
from smartfour.openbook import (
    DATA_BEGIN,
    book_key,
    entry_to_state,
    game_plans,
    load_book,
    state_to_entry,
)


def midgame_state(moves):
    state = initial_state()
    for x, z in moves:
        state = apply_move(state, x, z)
    return state


# ------------------------------------------------------------- round trip

def test_state_to_entry_roundtrip():
    state = midgame_state([(2, 2), (2, 2), (0, 4), (1, 1), (3, 3)])
    restored = entry_to_state(state_to_entry(state))
    assert restored == state
    assert restored.white == state.white
    assert restored.black == state.black
    assert restored.current == state.current
    assert restored.pieces_left == state.pieces_left


def test_first_move_state_current_is_black():
    state = apply_move(initial_state(), 2, 2)
    assert state.current == BLACK
    restored = entry_to_state(state_to_entry(state))
    assert restored.current == BLACK


def test_stack_string_layout_bottom_to_top():
    """Column (x=0, z=0): one white at y=0 under black at y=1 -> 'wb...'."""
    state = midgame_state([(0, 0), (0, 0)])
    entry = state_to_entry(state)
    assert entry[0][0] == "wb..."
    # Every other cell is empty.
    flat = [entry[x][z] for x in range(5) for z in range(5) if (x, z) != (0, 0)]
    assert all(c == "....." for c in flat)


# ----------------------------------------------------------- strict parse

def test_parse_rejects_wrong_shape():
    with pytest.raises(ValueError, match="entry 2"):
        entry_to_state([["....."]] * 4, index=2)
    with pytest.raises(ValueError):
        entry_to_state([["....."] * 4] * 5)  # 4 columns


def test_parse_rejects_bad_char_and_length():
    base = [["....."] * 5 for _ in range(5)]
    bad = [row[:] for row in base]
    bad[0][0] = "w....x"
    with pytest.raises(ValueError):
        entry_to_state(bad)
    bad2 = [row[:] for row in base]
    bad2[1][2] = "w..."  # too short
    with pytest.raises(ValueError):
        entry_to_state(bad2)


def test_parse_rejects_floating_piece():
    base = [["....."] * 5 for _ in range(5)]
    base[0][0] = ".w..."  # piece above an empty level
    with pytest.raises(ValueError, match="floating|above"):
        entry_to_state(base)


def test_parse_rejects_unbalanced_material():
    state = midgame_state([(0, 0), (0, 0), (1, 1)])  # white 2, black 1 -> ok
    entry_to_state(state_to_entry(state))  # must not raise (diff of 1)
    base = [["....."] * 5 for _ in range(5)]
    base[0][0] = "wwb.."  # two white vs one black is fine; make diff 2:
    base[1][1] = "w...."
    with pytest.raises(ValueError, match="unbalanced"):
        entry_to_state(base)


def test_parse_rejects_won_position():
    # Four whites along row x=0 (y=0) form a flat win line; blacks are
    # scattered non-collinear fillers keeping material balanced so the
    # balance guard does not fire first.
    base = [["....."] * 5 for _ in range(5)]
    for z in range(4):
        base[0][z] = "w...."
    base[4][0] = "b...."
    base[3][4] = "b...."
    base[2][4] = "b...."
    base[1][3] = "b...."
    with pytest.raises(ValueError, match="won"):
        entry_to_state(base)

def test_load_book_empty_by_default():
    saved = openbook.BOOK
    try:
        openbook.BOOK = []
        assert load_book() == []
    finally:
        openbook.BOOK = saved


def test_load_book_parses_and_rejects_duplicates():
    state = midgame_state([(2, 2), (2, 2)])
    entry = state_to_entry(state)
    saved = openbook.BOOK
    try:
        openbook.BOOK = [entry, entry]
        with pytest.raises(ValueError, match="duplicate"):
            load_book()
        openbook.BOOK = [entry, state_to_entry(midgame_state([(1, 1)]))]
        book = load_book()
        assert len(book) == 2
        assert book[0].white == state.white and book[0].black == state.black
    finally:
        openbook.BOOK = saved


# --------------------------------------------------------------- rendering

TEMPLATE = f'''"""doc"""\n{DATA_BEGIN}\n# old meta\nBOOK: list = []\n'''


def _full_template():
    from smartfour.openbook import DATA_END
    return TEMPLATE.rstrip("\n") + "\n" + DATA_END + "\n"


def test_apply_data_block_roundtrip():
    source = _full_template()
    states = [midgame_state([(2, 2), (2, 2)]), midgame_state([(0, 0)])]
    entries = [state_to_entry(s) for s in states]
    new_source = openbook.apply_data_block(source, entries, ["meta line"])
    assert new_source.count(DATA_BEGIN) == 1
    namespace: dict = {}
    exec(new_source, {"__name__": "openbook_test"}, namespace)  # noqa: S102
    parsed = [entry_to_state(e) for e in namespace["BOOK"]]
    assert [(s.white, s.black, s.current) for s in parsed] == \
        [(s.white, s.black, s.current) for s in states]


def test_apply_data_block_requires_markers():
    with pytest.raises(ValueError, match="data section"):
        openbook.apply_data_block("BOOK = []\n", [], [])


# ------------------------------------------------------------------ plans

def test_game_plans_without_book_alternates_colors():
    plans = game_plans(0, 5, seed=1)
    assert plans == [
        (None, True), (None, False), (None, True), (None, False), (None, True),
    ]


def test_game_plans_pairs_share_states_with_swapped_roles():
    plans = game_plans(7, 6, seed=42)
    assert len(plans) == 6
    for g in range(0, 6, 2):
        assert plans[g][0] == plans[g + 1][0], "pair shares one book state"
        assert plans[g][1] is True and plans[g + 1][1] is False


def test_game_plans_odd_games_last_state_once():
    plans = game_plans(5, 5, seed=9)
    assert len(plans) == 5
    # Two full pairs plus a single fifth game: only the last state plays once.
    assert plans[0][0] == plans[1][0]
    assert plans[2][0] == plans[3][0]
    assert plans[4][1] is True


def test_game_plans_deterministic():
    assert game_plans(11, 8, seed=3) == game_plans(11, 8, seed=3)
    assert game_plans(11, 8, seed=3) != game_plans(11, 8, seed=4)


def test_game_plans_indices_in_range():
    rng = random.Random(0)
    for _ in range(20):
        plans = game_plans(3, 10, seed=rng.randrange(1000))
        assert all(0 <= idx < 3 for idx, _role in plans)
