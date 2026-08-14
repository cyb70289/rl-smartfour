"""Probe a trained model's tactical competence (defense/attack awareness).

Usage:
  .venv/bin/python tools/probe.py --checkpoint checkpoints/best.pt --sims 200

Builds synthetic REACHABLE positions where the player to move has an
immediate win (win-in-1) and where the opponent threatens to complete a
4-in-a-row on the next move (block-in-1). Reports how often MCTS takes the
win / makes the block, plus a vs-random baseline win rate.

Reachability: every stack below a placed piece is filled with the opponent's
color, so each piece sits at its column's natural top and every completion
cell is a legal landing cell.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from smartfour.config import MCTSConfig
from smartfour.encode import action_to_xyz
from smartfour.game import (
    BLACK, WHITE, GameState, apply_move, empty_grid, legal_moves, other,
)
from smartfour.infer import SmartFourAgent
from smartfour.mcts import MCTS


def state_with(grid, current):
    return GameState(
        grid=grid,
        pieces_left={WHITE: 32, BLACK: 32},
        current=current,
        winner=None,
    )


def make_state(white_cells, black_cells, current):
    """Build a reachable state: cells below each piece are the opponent's."""
    grid = empty_grid()
    for x, z, y in white_cells:
        for fy in range(y):
            grid[x][z][fy] = BLACK
        grid[x][z][y] = WHITE
    for x, z, y in black_cells:
        for fy in range(y):
            grid[x][z][fy] = WHITE
        grid[x][z][y] = BLACK
    return state_with(grid, current)


def win_in_one_positions():
    """Positions where `current` completes a 4-in-a-row with one move."""
    cases = [
        # flat row: (0,0,0),(0,1,0),(0,2,0) -> complete at (0,3,0)
        ([(0, 0, 0), (0, 1, 0), (0, 2, 0)], [], WHITE, {(0, 3, 0)}),
        # flat diagonal: (4,0,0),(3,1,0),(2,2,0) -> complete at (1,3,0)
        ([(4, 0, 0), (3, 1, 0), (2, 2, 0)], [], WHITE, {(1, 3, 0)}),
        # rising diagonal: W at y0,y1,y2 over black stacks; the completion
        # column (3,0) carries a black stack of 3 so W lands at y=3
        ([(0, 0, 0), (1, 0, 1), (2, 0, 2)],
         [(1, 0, 0), (2, 0, 0), (2, 0, 1), (3, 0, 0), (3, 0, 1), (3, 0, 2)],
         WHITE, {(3, 0, 3)}),
        # black flat row
        ([], [(0, 4, 0), (0, 3, 0), (0, 2, 0)], BLACK, {(0, 1, 0)}),
        # black flat diagonal, both open ends
        ([], [(1, 1, 0), (1, 2, 0), (1, 3, 0)], BLACK, {(1, 4, 0), (1, 0, 0)}),
    ]
    return [
        (f"win-in-1 P{cur}", make_state(wc, bc, cur))
        for wc, bc, cur, _exp in cases
    ]


def block_in_one_positions():
    """Positions where the opponent completes a 4-in-a-row next move.

    Returns (name, state, expected_block_cells)."""
    cases = [
        # opp 3-in-a-row with one open end; current must take that cell
        ([(0, 0, 0), (0, 1, 0), (0, 2, 0)], [], WHITE, BLACK, {(0, 3, 0)}),
        ([(4, 0, 0), (3, 1, 0), (2, 2, 0)], [], WHITE, BLACK, {(1, 3, 0)}),
        ([(0, 0, 0), (1, 0, 1), (2, 0, 2)],
         [(1, 0, 0), (2, 0, 0), (2, 0, 1), (3, 0, 0), (3, 0, 1), (3, 0, 2)],
         WHITE, BLACK, {(3, 0, 3)}),
        ([], [(2, 2, 0), (2, 3, 0), (2, 4, 0)], BLACK, WHITE, {(2, 1, 0)}),
    ]
    return [
        (f"block-in-1 P{cur} vs P{opp}", make_state(wc, bc, cur), exp)
        for wc, bc, opp, cur, exp in cases
    ]

def fork_positions():
    """Deeper-tactics positions requiring ~4-ply lookahead.

    Fork: white has (0,1),(1,1) [row z=1] and (2,2),(2,3) [row x=2]; cell
    (2,1) completes BOTH lines into a 3-in-a-row with an open end, so white
    wins next move no matter what black blocks. Black to move must take (2,1)
    now (fork-defend). White to move must play it (fork-win).
    """
    wc = [(0, 1, 0), (1, 1, 0), (2, 2, 0), (2, 3, 0)]
    fork_cell = {(2, 1)}
    return [
        ("fork-defend P1", make_state(wc, [], BLACK), fork_cell),
        ("fork-win P0", make_state(wc, [], WHITE), fork_cell),
    ]


def immediate_wins(state):
    """Set of (x, z) moves that win immediately for state.current."""
    wins = set()
    for x, z in legal_moves(state):
        if apply_move(state, x, z).winner == state.current:
            wins.add((x, z))
    return wins


def probe(agent, sims, verbose=True):
    results = []
    for name, st in win_in_one_positions():
        wins = immediate_wins(st)
        mcts = MCTS(agent.net, MCTSConfig(simulations=sims))
        _pi, chosen, _root = mcts.root_policy(st, root_noise=False, temperature=0.0)
        x, z, _ = action_to_xyz(chosen)
        ok = (x, z) in wins
        results.append((name, ok, f"chose {(x, z)}; wins={sorted(wins)}"))
    for name, st, block in block_in_one_positions():
        mcts = MCTS(agent.net, MCTSConfig(simulations=sims))
        _pi, chosen, _root = mcts.root_policy(st, root_noise=False, temperature=0.0)
        x, z, _ = action_to_xyz(chosen)
        child = apply_move(st, x, z)
        ok = len(immediate_wins(child)) == 0  # opponent has no win after it
        results.append((name, ok, f"chose {(x, z)}; expected block {sorted(block)}"))
    for name, st, cell in fork_positions():
        mcts = MCTS(agent.net, MCTSConfig(simulations=sims))
        _pi, chosen, _root = mcts.root_policy(st, root_noise=False, temperature=0.0)
        x, z, _ = action_to_xyz(chosen)
        ok = (x, z) in cell
        results.append((name, ok, f"chose {(x, z)}; critical cell {sorted(cell)}"))
    if verbose:
        for name, ok, info in results:
            print(f"{'PASS' if ok else 'FAIL'}  {name:28s} {info}")
    return results


def vs_random(agent, sims, games, seed=0):
    from smartfour.arena import play_vs_random
    return play_vs_random(agent.net, MCTSConfig(simulations=sims), games, seed=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--random-games", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    agent = SmartFourAgent(args.checkpoint)
    print(f"checkpoint {args.checkpoint} (iteration {agent.iteration})")
    results = probe(agent, args.sims)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"tactics: {n_pass}/{len(results)} passed")
    if args.random_games > 0:
        w, l, d = vs_random(agent, args.sims, args.random_games, seed=1)
        total = w + l + d
        print(f"vs random ({args.random_games} games, {args.sims} sims): "
              f"{w}W/{l}L/{d}D  win ratio {(w + 0.5 * d) / total:.2f}")

if __name__ == "__main__":
    main()
