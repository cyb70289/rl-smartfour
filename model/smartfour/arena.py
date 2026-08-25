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
from .openbook import game_plans
from .network import ResNet
from .selfplay import ignore_sigint


def random_move(state, rng: random.Random | None = None):
    moves = legal_moves(state)
    if not moves:
        return None
    return rng.choice(moves) if rng else random.choice(moves)


def _play_two(net_first, net_second, mcts_cfg, evaluator_first=None,
              evaluator_second=None, start_state=None):
    """Greedy MCTS game (no noise, temperature 0).

    Returns (winner, plies, skipped): `plies` counts moves played from
    `start_state`, `skipped` counts pieces already on the board when the
    game started (0 from the initial board). The first two arguments are
    the nets of the player to move at `start_state` and of their opponent;
    real colors map internally.
    """
    state = start_state if start_state is not None else initial_state()
    if state.current == WHITE:
        white_mcts = MCTS(net_first, mcts_cfg, evaluator=evaluator_first)
        black_mcts = MCTS(net_second, mcts_cfg, evaluator=evaluator_second)
    else:
        white_mcts = MCTS(net_second, mcts_cfg, evaluator=evaluator_second)
        black_mcts = MCTS(net_first, mcts_cfg, evaluator=evaluator_first)
    skipped = bin(state.white | state.black).count("1")
    plies = 0
    while not is_terminal(state):
        mcts = white_mcts if state.current == WHITE else black_mcts
        _pi, chosen, _root = mcts.root_policy(state, root_noise=False, temperature=0.0)
        x, z, _y = action_to_xyz(chosen)
        state = apply_move(state, x, z)
        plies += 1
    if state.winner == DRAW:
        return DRAW, plies, skipped
    return (WHITE if state.winner == WHITE else BLACK), plies, skipped


def _result_in_a_frame(result, a_moved_first: bool):
    """Map a raw winner color to net_a's frame: WHITE = net_a wins.

    `_play_two(net_a, net_b)` reports from net_a-moves-first's perspective;
    a swapped game (`_play_two(net_b, net_a)`) must flip before counting.
    """
    if result == DRAW:
        return DRAW
    if a_moved_first:
        return result
    return BLACK if result == WHITE else WHITE


def play_arena(net_a, net_b, mcts_cfg, games: int, progress=None, plies_out=None,
               book=(), seed: int = 0, skipped_out=None):
    """Pit net_a against net_b over `games` games.

    With an opening `book`, game g starts from book[game_plans(...)[g][0]]
    with roles swapped inside each pair (net_a to move vs net_b to move);
    without one it alternates colors from the initial board as before.

    Returns (a_wins, b_wins, draws) counted from net_a's perspective.
    `progress`, if given, is called with no arguments after each game.
    `plies_out`, if given, receives each game's played ply count (book
    plies excluded); `skipped_out` receives the skipped book plies.
    """
    plans = game_plans(len(book), games, seed)
    a_wins = b_wins = draws = 0
    for i in range(games):
        idx, a_first = plans[i]
        start_state = book[idx] if idx is not None else None
        if a_first:
            result, plies, skipped = _play_two(
                net_a, net_b, mcts_cfg, start_state=start_state)
        else:
            result, plies, skipped = _play_two(
                net_b, net_a, mcts_cfg, start_state=start_state)
        result = _result_in_a_frame(result, a_first)
        if result == DRAW:
            draws += 1
        elif result == WHITE:
            a_wins += 1
        else:
            b_wins += 1
        if plies_out is not None:
            plies_out.append(plies)
        if skipped_out is not None:
            skipped_out.append(skipped)
        if progress is not None:
            progress()
    return a_wins, b_wins, draws

def arena_worker(net_a_state, net_b_state, net_cfg: NetworkConfig, mcts_cfg: MCTSConfig,
                 games: int, start: int, seed: int, num_threads, out_q,
                 server_addr=None, book=(), plans=None) -> None:
    """Process entry point: rebuild both nets, play `games` arena games.

    `server_addr` set: evaluate through the central inference server, slot 0
    = net_a, slot 1 = net_b (roles map per game). With an opening `book`,
    `plans` (the parent's global game_plans() slice for this worker's games)
    fixes each game's book state and role so parallel runs match a sequential
    one; without them color alternation continues from global game index
    `start`. Each result is put on out_q in net_a's frame as
    (WHITE / BLACK / DRAW, plies, skipped). Errors never crash the parent:
    they are reported as an ('__worker_error__', message) marker so the
    trainer can fail fast instead of hanging on a missing game.
    `num_threads` avoids core oversubscription when several workers share
    the machine.
    """
    ignore_sigint()
    ev_a = ev_b = None
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
        if server_addr is not None:
            from .inference_server import RemoteEvaluator
            ev_a = RemoteEvaluator(server_addr, slot=0)
            ev_b = RemoteEvaluator(server_addr, slot=1)
        for j in range(games):
            if plans is not None:
                idx, a_first = plans[j]
                start_state = book[idx] if idx is not None else None
            else:
                a_first = (start + j) % 2 == 0
                start_state = None
            if a_first:
                result, plies, skipped = _play_two(
                    net_a, net_b, mcts_cfg, ev_a, ev_b, start_state)
            else:
                result, plies, skipped = _play_two(
                    net_b, net_a, mcts_cfg, ev_b, ev_a, start_state)
            out_q.put((_result_in_a_frame(result, a_first), plies, skipped))
    except Exception as exc:  # noqa: BLE001 — must never take the parent down
        out_q.put(("__worker_error__", f"{type(exc).__name__}: {exc}"))
    finally:
        if ev_a is not None:
            ev_a.close()
        if ev_b is not None:
            ev_b.close()


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
