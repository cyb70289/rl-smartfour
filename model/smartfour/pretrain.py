"""Value-head pretraining on random-rollout game outcomes.

Why: a fresh network's value head is random noise, so MCTS leaf values are
worthless, PUCT never prefers a visited child over unvisited siblings, and
every node's ~25 children fill before any descends — the search freezes at
~3 plies of depth at any simulation budget (depth ~ log_25(sims)). Deep
tactics (forks, 2-ply threats) never enter the training data, the value
never learns, and play collapses into short races.

A rollout-trained value is cheap, unbiased, and tactically meaningful near
terminal positions: "does the current player win if both sides finish the
game at random" strongly reflects live threats (open 3-in-a-rows) and
near-wins — exactly the signal PUCT needs to concentrate visits and deepen
the tree.

Sampling is importance-weighted: uniform mid-game states are near-coinflips
under random completion (their outcome mean ~ 0, which teaches the value
nothing), so we collect the LAST `tail_plies` plies of each random game and
average K independent rollouts per state into a soft label.

The pretrain pass trains ONLY the value head (MSE against the soft labels).
The policy head stays at random init; self-play learns it from search
targets as usual.
"""

import random

import torch

from .encode import apply_d4, d4_perms, encode
from .game import (
    BLACK, DRAW, WHITE, apply_move, initial_state, is_terminal, legal_moves,
)
from .network import ResNet

def rollout_z(state, rng):
    """One random game from `state`; outcome from state.current's perspective."""
    st = state
    while not is_terminal(st):
        x, z = rng.choice(legal_moves(st))
        st = apply_move(st, x, z)
    winner = st.winner
    if winner == DRAW:
        return 0.0
    return 1.0 if winner == state.current else -1.0


def collect_rollout_samples(games: int, seed: int, tail_plies: int = 8,
                            rollouts: int = 20) -> tuple:
    """Random games; for each state in the last `tail_plies` plies of each
    game, average `rollouts` random completions into a soft label.

    Returns (states, zs): lists of (16,5,5) tensors and soft z labels in
    [-1, 1], each from the stored state's player perspective.
    """
    rng = random.Random(seed)
    states = []
    zs = []
    for _ in range(games):
        st = initial_state()
        path = []
        while not is_terminal(st):
            path.append(st)
            x, z = rng.choice(legal_moves(st))
            st = apply_move(st, x, z)
        for s in path[-tail_plies:]:
            acc = 0.0
            for _ in range(rollouts):
                acc += rollout_z(s, rng)
            states.append(encode(s))
            zs.append(acc / rollouts)
    return states, zs


def pretrain_value(net: ResNet, games: int, epochs: int, batch_size: int,
                   lr: float, weight_decay: float, seed: int,
                   tail_plies: int = 8, rollouts: int = 20,
                   progress=None) -> float:
    """Train the value head (and the shared trunk) on rollout labels.

    The policy head is frozen at its random init; self-play learns it later.
    `progress`, if given, is called after every batch. Returns the final MSE.
    """
    states, zs = collect_rollout_samples(games, seed, tail_plies, rollouts)
    n = len(states)
    if n == 0:
        raise ValueError("rollout collection produced no samples")
    optimizer = torch.optim.AdamW(
        [
            p for name, p in net.named_parameters()
            if not name.startswith("policy_head")
        ],
        lr=lr,
        weight_decay=weight_decay,
    )
    perms = d4_perms()
    n_batches = max(1, n // batch_size)
    final = float("nan")
    for _ in range(epochs):
        order = torch.randperm(n)
        for b in range(n_batches):
            idx = order[b * batch_size:(b + 1) * batch_size]
            s = torch.stack([states[i] for i in idx])
            z = torch.tensor(
                [[zs[i]] for i in idx], dtype=torch.float32
            )
            # D4 augmentation (value invariant, policy untouched)
            for i in range(s.shape[0]):
                s[i] = apply_d4(s[i], perms[int(torch.randint(8, (1,)).item())])
            _logits, value = net(s)
            loss = ((value - z) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final = loss.item()
            if progress is not None:
                progress()
    return final
