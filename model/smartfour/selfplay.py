"""Self-play: one game = MCTS at every ply + outcome labeling.

Returns (samples, winner): samples are (state_tensor, pi, player, z) tuples
where z is the game outcome from `player`'s perspective, winner is
WHITE / BLACK / 'draw'.

Parallel self-play runs one spawned process per `selfplay_workers` (the
Python GIL serializes the MCTS tree logic, so threads cannot use more than
one CPU core); each worker plays its share of games with its own net copy
and ships the samples back over a queue.
"""

import os
import signal

import torch

from .config import MCTSConfig, NetworkConfig
from .encode import action_to_xyz, encode
from .game import BLACK, WHITE, GameState, apply_move, initial_state, is_terminal, terminal_value
from .mcts import MCTS
from .network import ResNet


def play_game(net, mcts_cfg, temperature_threshold: int, start_state: GameState | None = None):
    """Play one self-play game with dirichlet noise and temperature scheduling.

    temperature_threshold: plies below it use tau=1 (sample from visit
    counts), the rest use tau=0 (argmax).
    """
    state = start_state if start_state is not None else initial_state()
    mcts = MCTS(net, mcts_cfg)
    samples = []
    while not is_terminal(state):
        ply = len(samples)
        temperature = 1.0 if ply < temperature_threshold else 0.0
        pi, chosen, _ = mcts.root_policy(state, root_noise=True, temperature=temperature)
        samples.append((encode(state), pi, state.current, 0.0))
        x, z, _y = action_to_xyz(chosen)
        state = apply_move(state, x, z)

    # Outcome from white's perspective, then flip per stored player.
    winner = state.winner
    z_white = 1.0 if winner == WHITE else (-1.0 if winner == BLACK else 0.0)
    labeled = []
    for s, pi, player, _ in samples:
        z = z_white if player == WHITE else -z_white
        labeled.append((s, pi, player, z))
    return labeled, winner


def split_games(games: int, workers: int) -> list:
    """Split `games` into `workers` non-negative counts, each within 1 of the rest.

    The remainder goes to the first workers, so e.g. split_games(10, 3) ==
    [4, 3, 3]. Workers with a zero share are simply not spawned.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if games < 0:
        raise ValueError(f"games must be >= 0, got {games}")
    base, extra = divmod(games, workers)
    return [base + (1 if i < extra else 0) for i in range(workers)]


def samples_to_ipc(samples):
    """Convert samples to a queue-safe form (numpy arrays pickle as bytes).

    Raw torch tensors through a spawn Queue use shared-memory FD passing whose
    bookkeeping dies with the worker, so a late message can fail to unpickle.
    Numpy arrays avoid that entirely; the parent rebuilds tensors with
    `samples_from_ipc`.
    """
    return [(s.numpy(), pi.numpy(), player, z) for s, pi, player, z in samples]


def samples_from_ipc(samples):
    """Rebuild torch tensors from `samples_to_ipc` output (parent side)."""
    return [
        (torch.from_numpy(s), torch.from_numpy(pi), player, z)
        for s, pi, player, z in samples
    ]


def ignore_sigint() -> None:
    """Workers must not react to Ctrl-C: the parent alone decides when to stop
    and terminates the workers explicitly. A worker that died on the
    terminal's SIGINT would race the parent's cleanup."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def selfplay_worker(net_state, net_cfg: NetworkConfig, mcts_cfg: MCTSConfig,
                    temperature_threshold: int, games: int, seed: int,
                    num_threads, out_q) -> None:
    """Process entry point: rebuild the net, play `games` games, ship the
    samples of each game on out_q.

    Errors never crash the parent: they are reported as an
    ('__worker_error__', message) marker so the trainer can fail fast instead
    of hanging on a missing game. `num_threads` avoids core oversubscription
    when several workers share the machine.
    """
    ignore_sigint()
    try:
        torch.manual_seed(seed)
        if num_threads:
            torch.set_num_threads(max(1, int(num_threads)))
        net = ResNet(net_cfg)
        net.load_state_dict(net_state)
        net.eval()
        for _ in range(games):
            samples, _winner = play_game(
                net, mcts_cfg, temperature_threshold
            )
            out_q.put(samples_to_ipc(samples))
    except Exception as exc:  # noqa: BLE001 — must never take the parent down
        out_q.put(("__worker_error__", f"{type(exc).__name__}: {exc}"))


def worker_num_threads(workers: int) -> int:
    """Per-worker torch thread count.

    Each torch process otherwise defaults to all cores, so N workers would
    oversubscribe an N-core box; divide the cores evenly instead.
    """
    return max(1, (os.cpu_count() or 1) // workers)
