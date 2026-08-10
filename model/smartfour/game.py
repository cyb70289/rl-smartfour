"""Smart-four game rules — faithful Python port of ui/src/game/rules.ts.

Players are ints: WHITE=0, BLACK=1. Grid is grid[x][z][y] like the UI.
Move = (x, z) column; the piece lands on top of the stack. Terminal when a
player lines up 4+ pieces (win) or both players exhaust their pieces (draw).
"""

from dataclasses import dataclass, field
from typing import Optional

BOARD_SIZE = 5
STACK_HEIGHT = 5
DEFAULT_PIECES = 32

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


@dataclass(frozen=True)
class GameState:
    grid: list  # grid[x][z][y] -> WHITE | BLACK | None
    pieces_left: dict = field(default_factory=lambda: {WHITE: DEFAULT_PIECES, BLACK: DEFAULT_PIECES})
    current: int = WHITE
    winner: Winner = None  # None | DRAW | WHITE | BLACK


def empty_grid() -> list:
    return [
        [[None for _ in range(STACK_HEIGHT)] for _ in range(BOARD_SIZE)]
        for _ in range(BOARD_SIZE)
    ]


def initial_state(pieces_per_player: int = DEFAULT_PIECES) -> GameState:
    return GameState(
        grid=empty_grid(),
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
    return stack_height(state.grid, x, z) < STACK_HEIGHT


def legal_moves(state: GameState) -> list:
    return [(x, z) for x in range(BOARD_SIZE) for z in range(BOARD_SIZE) if is_legal(state, x, z)]


def find_win_run(grid: list, x: int, z: int, y: int, player: int):
    """Full same-color run through (x, z, y) if it contains 4+ cells, else None."""
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


def apply_move(state: GameState, x: int, z: int) -> GameState:
    if not is_legal(state, x, z):
        raise IllegalMoveError(f"illegal move ({x}, {z})")
    y = stack_height(state.grid, x, z)
    player = state.current

    grid = [[col[:] for col in plane] for plane in state.grid]
    grid[x][z][y] = player

    pieces_left = dict(state.pieces_left)
    pieces_left[player] -= 1

    winning = find_win_run(grid, x, z, y, player)
    winner: Winner
    if winning:
        winner = player
    elif pieces_left[WHITE] == 0 and pieces_left[BLACK] == 0:
        winner = DRAW
    else:
        winner = None

    return GameState(grid=grid, pieces_left=pieces_left, current=other(player), winner=winner)


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
    return GameState(
        grid=grid,
        pieces_left={
            WHITE: pieces_left["white"],
            BLACK: pieces_left["black"],
        },
        current=_NAME_PLAYER[current],
        winner=None if winner is None else (
            DRAW if winner == "draw" else _NAME_PLAYER[winner]
        ),
    )
