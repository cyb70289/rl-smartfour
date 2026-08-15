"""Smart-four game rules — faithful Python port of ui/src/game/rules.ts.

Performance-critical representation: each position is a pair of 125-bit
bitboards (one per player), bit index ``y * 25 + x * 5 + z`` matching the
action index.  Stacking semantics fall out of the bit layout: a column is a
contiguous run of low bits, so height is a popcount and legality is a single
AND.  Win detection walks precomputed line masks through the moved cell only.

The legacy ``grid[x][z][y]`` nested-list representation (UI interchange and
tests) is materialized lazily on attribute access.
"""

from dataclasses import dataclass, field
from typing import Optional

BOARD_SIZE = 5
STACK_HEIGHT = 5
DEFAULT_PIECES = 32
CELLS = BOARD_SIZE * BOARD_SIZE * STACK_HEIGHT  # 125

WHITE = 0
BLACK = 1

DRAW = "draw"

# Canonical line directions as [dx, dz, dy] with dy in {-1, 0, 1}: 4 flat
# (dy=0) + 1 vertical + 8 rising diagonals (dy=+/-1). Reverses are covered by
# counting both ways from a cell.
DIRS = (
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
)

Winner = Optional[int]  # None = in progress, DRAW = draw, else winning player


class IllegalMoveError(Exception):
    pass


# --------------------------------------------------------------- bit layout

def _cell_bit(x: int, z: int, y: int) -> int:
    """Bit layout: x*25 + z*5 + y — each column (x, z) is a contiguous
    5-bit slice, so height/legality are cheap slice ops."""
    return 1 << (x * 25 + z * 5 + y)


# All win lines as (frozenset_of_cells, mask); a line is any 4 collinear
# cells inside the 5x5x5 cube. Built once at import.
def _build_win_lines():
    lines = []
    for dx, dz, dy in DIRS:
        for x in range(BOARD_SIZE):
            for z in range(BOARD_SIZE):
                for y in range(STACK_HEIGHT):
                    ex = x + 3 * dx
                    ez = z + 3 * dz
                    ey = y + 3 * dy
                    if not (0 <= ex < BOARD_SIZE and 0 <= ez < BOARD_SIZE and 0 <= ey < STACK_HEIGHT):
                        continue

                    cells = tuple(
                        (x + i * dx, z + i * dz, y + i * dy) for i in range(4)
                    )
                    mask = 0
                    for cx, cz, cy in cells:
                        mask |= _cell_bit(cx, cz, cy)
                    lines.append((cells, mask))
    return tuple(lines)


_WIN_LINES = _build_win_lines()

_LINES_THROUGH_CELL = [[] for _ in range(125)]
for _cells, _mask in _WIN_LINES:
    for _cx, _cz, _cy in _cells:
        _LINES_THROUGH_CELL[_cx * 25 + _cz * 5 + _cy].append(_mask)
_LINES_THROUGH_CELL = tuple(tuple(m) for m in _LINES_THROUGH_CELL)

def _bitboard_to_grid(white: int, black: int) -> list:
    """Materialize the legacy grid[x][z][y] nested list from bitboards."""
    grid = [[[None] * STACK_HEIGHT for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for x in range(BOARD_SIZE):
        for z in range(BOARD_SIZE):
            col = grid[x][z]
            for y in range(STACK_HEIGHT):
                b = 1 << (x * 25 + z * 5 + y)
                if white & b:
                    col[y] = WHITE
                elif black & b:
                    col[y] = BLACK
    return grid

def _grid_to_bitboards(grid: list) -> tuple:
    """Parse a legacy grid[x][z][y] into (white_bits, black_bits)."""
    white = black = 0
    for x in range(BOARD_SIZE):
        for z in range(BOARD_SIZE):
            col = grid[x][z]
            for y in range(STACK_HEIGHT):
                p = col[y]
                if p is None:
                    continue
                b = _cell_bit(x, z, y)
                if p == WHITE:
                    white |= b
                else:
                    black |= b
    return white, black


_COLUMN_FULL = tuple(
    0b11111 << (x * 25 + z * 5)
    for x in range(BOARD_SIZE)
    for z in range(BOARD_SIZE)
)
_COLUMN_FULL_BY_XZ = {
    (x, z): 0b11111 << (x * 25 + z * 5)
    for x in range(BOARD_SIZE)
    for z in range(BOARD_SIZE)
}

# Height of each 32 possible 5-bit column slice: length of the low-bit run
# (legal play always fills from bit 0; a gapped slice parses as the prefix,
# matching the legacy stack_height scan).
_HEIGHTS = tuple(
    (c ^ (c + 1)).bit_length() - 1 for c in range(32)
)


def _column_mask(x: int, z: int) -> int:
    """Bits of the full 5-cell column (x, z), heights 0..4."""
    return 0b11111 << (x * 25 + z * 5)


def _column_height(occ: int, x: int, z: int) -> int:
    """Pieces stacked in column (x, z) of the combined occupancy bitboard."""
    return _HEIGHTS[(occ >> (x * 25 + z * 5)) & 0b11111]

@dataclass(frozen=True)
class GameState:
    white: int = 0                     # bitboard: white pieces
    black: int = 0                     # bitboard: black pieces
    pieces_left: dict = field(default_factory=lambda: {WHITE: DEFAULT_PIECES, BLACK: DEFAULT_PIECES})
    current: int = WHITE
    winner: Winner = None              # None | DRAW | WHITE | BLACK
    _grid: Optional[list] = None       # lazily materialized legacy representation

    def __init__(self, grid=None, white=0, black=0,
                 pieces_left=None, current=WHITE, winner=None, _grid=None):
        """Accepts either the bitboard fields (fast path) or the legacy
        `grid` nested list (tests/UI interchange), not both."""
        if grid is not None:
            if white or black:
                raise ValueError("pass either grid or white/black bitboards, not both")
            white, black = _grid_to_bitboards(grid)
        if pieces_left is None:
            pieces_left = {WHITE: DEFAULT_PIECES, BLACK: DEFAULT_PIECES}
        object.__setattr__(self, "white", white)
        object.__setattr__(self, "black", black)
        object.__setattr__(self, "pieces_left", pieces_left)
        object.__setattr__(self, "current", current)
        object.__setattr__(self, "winner", winner)


    def column_open(self, x: int, z: int) -> bool:
        """Column (x, z) has room (height < 5). No bounds/type checks: use
        after `is_legal`-style validation."""
        m = _column_mask(x, z)
        return (self.white | self.black) & m != m

    # -- legacy nested-list representation (UI JSON, tests) -----------------

    @property
    def grid(self):
        if self._grid is None:
            object.__setattr__(self, "_grid", _bitboard_to_grid(self.white, self.black))
        return self._grid

    def __eq__(self, other):
        if not isinstance(other, GameState):
            return NotImplemented
        return (
            self.white == other.white
            and self.black == other.black
            and self.current == other.current
            and self.winner == other.winner
            and self.pieces_left == other.pieces_left
        )

    def __hash__(self):
        return hash((self.white, self.black, self.current, self.winner))


def empty_grid() -> list:
    return [
        [[None for _ in range(STACK_HEIGHT)] for _ in range(BOARD_SIZE)]
        for _ in range(BOARD_SIZE)
    ]


def initial_state(pieces_per_player: int = DEFAULT_PIECES) -> GameState:
    return GameState(
        pieces_left={WHITE: pieces_per_player, BLACK: pieces_per_player},
        current=WHITE,
        winner=None,
    )


def other(player: int) -> int:
    return BLACK if player == WHITE else WHITE


def stack_height(grid: list, x: int, z: int) -> int:
    if not (0 <= x < BOARD_SIZE and 0 <= z < BOARD_SIZE):
        return 0
    h = 0
    while h < STACK_HEIGHT and grid[x][z][h] is not None:
        h += 1
    return h


def is_legal(state: GameState, x: int, z: int) -> bool:
    if state.winner is not None:
        return False
    if not isinstance(x, int) or not isinstance(z, int):
        return False
    if x < 0 or x >= BOARD_SIZE or z < 0 or z >= BOARD_SIZE:
        return False
    if state.pieces_left[state.current] <= 0:
        return False
    return state.column_open(x, z)


def legal_moves(state: GameState) -> list:
    if state.winner is not None or state.pieces_left[state.current] <= 0:
        return []
    return [
        (x, z)
        for x in range(BOARD_SIZE)
        for z in range(BOARD_SIZE)
        if state.column_open(x, z)
    ]


def find_win_run(grid: list, x: int, z: int, y: int, player: int):
    """Full same-color run through (x, z, y) if it contains 4+ cells, else None.

    Legacy grid interface (tests). The bitboard fast path uses
    `_wins_through`, this walks the grid directly.
    """
    for dx, dz, dy in DIRS:
        run = [(x, z, y)]
        cx, cz, cy = x + dx, z + dz, y + dy
        while (
            0 <= cx < BOARD_SIZE and 0 <= cz < BOARD_SIZE and 0 <= cy < STACK_HEIGHT
            and grid[cx][cz][cy] == player
        ):
            run.append((cx, cz, cy))
            cx, cz, cy = cx + dx, cz + dz, cy + dy
        cx, cz, cy = x - dx, z - dz, y - dy
        while (
            0 <= cx < BOARD_SIZE and 0 <= cz < BOARD_SIZE and 0 <= cy < STACK_HEIGHT
            and grid[cx][cz][cy] == player
        ):
            run.insert(0, (cx, cz, cy))
            cx, cz, cy = cx - dx, cz - dz, cy - dy
        if len(run) >= 4:
            return run
    return None


def _wins_through(white: int, black: int, x: int, z: int, y: int, player: int) -> bool:
    """Bitboard win check: any precomputed 4-line through (x, z, y) fully
    inside `player`'s bitboard."""
    bits = white if player == WHITE else black
    for mask in _LINES_THROUGH_CELL[x * 25 + z * 5 + y]:
        if mask & bits == mask:
            return True
    return False


def apply_move(state: GameState, x: int, z: int) -> GameState:
    if not is_legal(state, x, z):
        raise IllegalMoveError(f"illegal move ({x}, {z})")
    player = state.current
    occ = state.white | state.black
    y = _column_height(occ, x, z)
    bit = _cell_bit(x, z, y)

    if player == WHITE:
        white, black = state.white | bit, state.black
    else:
        white, black = state.white, state.black | bit

    pieces_left = dict(state.pieces_left)
    pieces_left[player] -= 1

    winning = _wins_through(white, black, x, z, y, player)
    winner: Winner
    if winning:
        winner = player
    elif pieces_left[WHITE] == 0 and pieces_left[BLACK] == 0:
        winner = DRAW
    else:
        winner = None

    return GameState(
        white=white,
        black=black,
        pieces_left=pieces_left,
        current=other(player),
        winner=winner,
    )


def is_terminal(state: GameState) -> bool:
    return state.winner is not None


def terminal_value(state: GameState) -> float:
    """Outcome from the perspective of `state.current` (the player to move)."""
    if state.winner == DRAW:
        return 0.0
    return 1.0 if state.winner == state.current else -1.0


# ------------------------------------------------------------- JSON contract
# Grid values are 0 (white) / 1 (black) / null, matching the UI's state format.
# This is the interchange format for the future UI integration.

_PLAYER_NAME = {WHITE: "white", BLACK: "black"}
_NAME_PLAYER = {"white": WHITE, "black": BLACK}


def state_to_json(state: GameState) -> dict:
    return {
        "grid": state.grid,
        "pieces_left": {
            "white": state.pieces_left[WHITE],
            "black": state.pieces_left[BLACK],
        },
        "current": _PLAYER_NAME[state.current],
        "winner": None if state.winner is None else (
            "draw" if state.winner == DRAW else _PLAYER_NAME[state.winner]
        ),
    }


def state_from_json(data: dict) -> GameState:
    winner = data.get("winner")
    grid = data["grid"]
    if len(grid) != BOARD_SIZE or any(
        len(col) != BOARD_SIZE or len(stack) != STACK_HEIGHT
        for col in grid for stack in col
    ):
        raise ValueError(
            f"grid must be {BOARD_SIZE}x{BOARD_SIZE}x{STACK_HEIGHT} (grid[x][z][y])"
        )
    pieces_left = data["pieces_left"]
    if pieces_left.get("white") is None or pieces_left.get("black") is None:
        raise ValueError("pieces_left must have 'white' and 'black' counts")
    current = data.get("current")
    if current not in _NAME_PLAYER:
        raise ValueError(f"current must be 'white' or 'black', got {current!r}")
    if winner is not None and winner != "draw" and winner not in _NAME_PLAYER:
        raise ValueError(f"winner must be null, 'draw', 'white' or 'black', got {winner!r}")
    white, black = _grid_to_bitboards(grid)
    return GameState(
        white=white,
        black=black,
        pieces_left={
            WHITE: pieces_left["white"],
            BLACK: pieces_left["black"],
        },
        current=_NAME_PLAYER[current],
        winner=None if winner is None else (
            DRAW if winner == "draw" else _NAME_PLAYER[winner]
        ),
    )
