"""Head-to-head evaluation: MCTS agents and a random baseline."""

import random

from .encode import action_to_xyz
from .game import BLACK, DRAW, WHITE, apply_move, initial_state, is_terminal, legal_moves
from .mcts import MCTS


def random_move(state, rng: random.Random | None = None):
    moves = legal_moves(state)
    if not moves:
        return None
    return rng.choice(moves) if rng else random.choice(moves)


def _play_two(net_white, net_black, mcts_cfg, eval_simulations):
    """Greedy MCTS game (no noise, temperature 0). Returns the winner color."""
    white_mcts = MCTS(net_white, mcts_cfg)
    black_mcts = MCTS(net_black, mcts_cfg)
    state = initial_state()
    while not is_terminal(state):
        mcts = white_mcts if state.current == WHITE else black_mcts
        _pi, chosen, _root = mcts.root_policy(state, root_noise=False, temperature=0.0)
        x, z, _y = action_to_xyz(chosen)
        state = apply_move(state, x, z)
    if state.winner == DRAW:
        return DRAW
    return WHITE if state.winner == WHITE else BLACK


def play_arena(net_a, net_b, mcts_cfg, games: int, eval_simulations: int):
    """Pit net_a against net_b over `games` games, alternating colors.

    Returns (a_wins, b_wins, draws) counted from net_a's perspective.
    """
    a_wins = b_wins = draws = 0
    for i in range(games):
        if i % 2 == 0:
            result = _play_two(net_a, net_b, mcts_cfg, eval_simulations)
        else:
            result = _play_two(net_b, net_a, mcts_cfg, eval_simulations)
            # Express the result in net_a's frame: WHITE = net_a, BLACK = net_b.
            result = BLACK if result == WHITE else (WHITE if result == BLACK else result)
        if result == DRAW:
            draws += 1
        elif result == WHITE:
            a_wins += 1
        else:
            b_wins += 1
    return a_wins, b_wins, draws


def play_vs_random(net, mcts_cfg, games: int, eval_simulations: int, seed: int = 0):
    """Pit `net` against a uniform-random opponent. Returns (wins, losses, draws)."""
    rng = random.Random(seed)
    wins = losses = draws = 0
    mcts = MCTS(net, mcts_cfg)
    for i in range(games):
        net_is_white = i % 2 == 0
        state = initial_state()
        while not is_terminal(state):
            if (state.current == WHITE) == net_is_white:
                _pi, chosen, _root = mcts.root_policy(state, root_noise=False, temperature=0.0)
                x, z, _y = action_to_xyz(chosen)
            else:
                x, z = random_move(state, rng)
            state = apply_move(state, x, z)
        if state.winner == DRAW:
            draws += 1
        elif (state.winner == WHITE) == net_is_white:
            wins += 1
        else:
            losses += 1
    return wins, losses, draws
