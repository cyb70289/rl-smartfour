"""Smart-four AlphaZero model: game rules, encoding, resnet, MCTS, training.

Rules mirror the shipped UI engine (ui/src/game/rules.ts): draw when both
players exhaust their pieces, win on a run of 4+ along any 3D line.
"""

from .game import (
    BOARD_SIZE,
    STACK_HEIGHT,
    DEFAULT_PIECES,
    BLACK,
    WHITE,
    GameState,
    IllegalMoveError,
    initial_state,
    apply_move,
    is_legal,
    legal_moves,
    stack_height,
    is_terminal,
    terminal_value,
)

__all__ = [
    "BOARD_SIZE",
    "STACK_HEIGHT",
    "DEFAULT_PIECES",
    "BLACK",
    "WHITE",
    "GameState",
    "IllegalMoveError",
    "initial_state",
    "apply_move",
    "is_legal",
    "legal_moves",
    "stack_height",
    "is_terminal",
    "terminal_value",
]
