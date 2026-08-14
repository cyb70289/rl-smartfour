"""Head-to-head evaluation: MCTS agents and a random baseline.

The arena can run in parallel: one spawned process per `workers` plays
its share of the games with fresh copies of both nets (the Python GIL
serializes the MCTS tree logic, so threads cannot use more than one CPU
core). Each worker ships per-game results over a queue; the trainer counts
them exactly like a sequential run.
"""

import random

import torch

from .config import MCTSConfig, NetworkConfig
from .encode import action_to_xyz
from .game import BLACK, DRAW, WHITE, apply_move, initial_state, is_terminal, legal_moves
from .mcts import MCTS
from .network import ResNet
from .selfplay import ignore_sigint


def random_move(state, rng: random.Random | None = None):
    moves = legal_moves(state)
    if not moves:
        return None
    return rng.choice(moves) if rng else random.choice(moves)


def _play_two(net_white, net_black, mcts_cfg):
    """Greedy MCTS game (no noise, temperature 0). Returns (winner, plies)."""
    white_mcts = MCTS(net_white, mcts_cfg)
    black_mcts = MCTS(net_black, mcts_cfg)
    state = initial_state()
    plies = 0
    while not is_terminal(state):
        mcts = white_mcts if state.current == WHITE else black_mcts
        _pi, chosen, _root = mcts.root_policy(state, root_noise=False, temperature=0.0)
        x, z, _y = action_to_xyz(chosen)
        state = apply_move(state, x, z)
        plies += 1
    if state.winner == DRAW:
        return DRAW, plies
    return (WHITE if state.winner == WHITE else BLACK), plies


def _result_in_a_frame(result, a_is_white: bool):
    """Map a raw winner color to net_a's frame: WHITE = net_a wins.

    `_play_two(net_a, net_b)` reports from net_a-as-white's perspective; a
    swapped game (`_play_two(net_b, net_a)`) must flip before counting.
    """
    if result == DRAW:
        return DRAW
    if a_is_white:
        return result
    return BLACK if result == WHITE else WHITE


def play_arena(net_a, net_b, mcts_cfg, games: int, progress=None, plies_out=None):
    """Pit net_a against net_b over `games` games, alternating colors.

    Returns (a_wins, b_wins, draws) counted from net_a's perspective.
    `progress`, if given, is called with no arguments after each game.
    `plies_out`, if given, is a list receiving each game's ply count.
    """
    a_wins = b_wins = draws = 0
    for i in range(games):
        a_is_white = i % 2 == 0
        if a_is_white:
            result, plies = _play_two(net_a, net_b, mcts_cfg)
        else:
            result, plies = _play_two(net_b, net_a, mcts_cfg)
        result = _result_in_a_frame(result, a_is_white)
        if result == DRAW:
            draws += 1
        elif result == WHITE:
            a_wins += 1
        else:
            b_wins += 1
        if plies_out is not None:
            plies_out.append(plies)
        if progress is not None:
            progress()
    return a_wins, b_wins, draws

def arena_worker(net_a_state, net_b_state, net_cfg: NetworkConfig, mcts_cfg: MCTSConfig,
                 games: int, start: int, seed: int, num_threads, out_q) -> None:
    """Process entry point: rebuild both nets, play `games` arena games.

    Color alternation continues from global game index `start`, so each
    worker follows the same parity schedule as a sequential run. Each result
    is put on out_q in net_a's frame (WHITE / BLACK / DRAW). Errors never
    crash the parent: they are reported as an ('__worker_error__', message)
    marker so the trainer can fail fast instead of hanging on a missing game.
    `num_threads` avoids core oversubscription when several workers share the
    machine.
    """
    ignore_sigint()
    try:
        torch.manual_seed(seed)
        if num_threads:
            torch.set_num_threads(max(1, int(num_threads)))
        net_a = ResNet(net_cfg)
        net_a.load_state_dict(net_a_state)
        net_a.eval()
        net_b = ResNet(net_cfg)
        net_b.load_state_dict(net_b_state)
        net_b.eval()
        for j in range(games):
            a_is_white = (start + j) % 2 == 0
            if a_is_white:
                result, plies = _play_two(net_a, net_b, mcts_cfg)
            else:
                result, plies = _play_two(net_b, net_a, mcts_cfg)
            out_q.put((_result_in_a_frame(result, a_is_white), plies))
    except Exception as exc:  # noqa: BLE001 — must never take the parent down
        out_q.put(("__worker_error__", f"{type(exc).__name__}: {exc}"))


def play_vs_random(net, mcts_cfg, games: int, seed: int = 0):
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
