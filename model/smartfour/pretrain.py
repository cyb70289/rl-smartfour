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

Collection parallelizes across processes (pure-Python rollout play; the GPU
is idle). Labels are distribution-equal to a sequential run — worker i
draws from an independent sub-seed stream, so the exact sample set differs
from single-process collection but is drawn from the same distribution.

The pretrain pass trains ONLY the value head (MSE against the soft labels).
The policy head stays at random init; self-play learns it from search
targets as usual. Training runs on `device` (mps/cuda when selected):
samples are stacked once and moved to the device as one tensor, batches are
sliced on-device; D4 augmentation stays a per-batch CPU transform (the
~1MB upload is noise next to a ~60ms step) so the RNG semantics are
unchanged.
"""

import multiprocessing
import os

import torch

from .encode import apply_d4, d4_perms, encode
from .game import (
    BLACK, DRAW, WHITE, apply_move, initial_state, is_terminal, legal_moves,
)
from .network import ResNet
from .selfplay import ignore_sigint, split_games


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
    import random

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


def _collect_worker(games: int, seed: int, tail_plies: int, rollouts: int,
                    out_q) -> None:
    """Process entry point for parallel collection.

    Errors are reported as ('__worker_error__', message) like the self-play
    workers; a crash must fail the pretrain loudly, never shrink it.
    """
    ignore_sigint()
    try:
        states, zs = collect_rollout_samples(games, seed, tail_plies, rollouts)
        out_q.put((
            torch.stack(states).numpy() if states else torch.zeros(0, 16, 5, 5).numpy(),
            zs,
        ))
    except Exception as exc:  # noqa: BLE001
        out_q.put(("__worker_error__", f"{type(exc).__name__}: {exc}"))


def collect_rollout_samples_parallel(games: int, seed: int, workers: int,
                                     tail_plies: int = 8,
                                     rollouts: int = 20) -> tuple:
    """Parallel collect_rollout_samples: `workers` processes each play their
    share of the games from independent sub-seed streams.

    Distribution-equal to the sequential call (same game count, same tail /
    rollout parameters, independent RNG), not bit-equal. Returns the same
    (states, zs) shape, concatenated in worker order.
    """
    if workers <= 1 or games <= 1:
        return collect_rollout_samples(games, seed, tail_plies, rollouts)
    ctx = multiprocessing.get_context("spawn")
    out_q = ctx.Queue()
    procs = []
    try:
        for i, n in enumerate(split_games(games, workers)):
            if n == 0:
                continue
            p = ctx.Process(
                target=_collect_worker,
                args=(n, seed + i + 1, tail_plies, rollouts, out_q),
                daemon=True,
            )
            p.start()
            procs.append(p)
        all_states = []
        all_zs = []
        received = 0
        expected = len([q for q in split_games(games, workers) if q > 0])
        import queue as queue_mod
        while received < expected:
            try:
                msg = out_q.get(timeout=1.0)
            except queue_mod.Empty:
                if all(not p.is_alive() for p in procs):
                    raise RuntimeError(
                        f"rollout collection workers exited early: "
                        f"{received}/{expected} results collected"
                    )
                continue
            if (
                isinstance(msg, tuple) and len(msg) == 2
                and isinstance(msg[0], str) and msg[0] == "__worker_error__"
            ):
                raise RuntimeError(f"rollout collection worker failed: {msg[1]}")
            arr, zs = msg
            all_states.append(torch.from_numpy(arr))
            all_zs.extend(zs)
            received += 1
        for p in procs:
            p.join(timeout=60)
        bad = [p.exitcode for p in procs if p.exitcode not in (0, None)]
        if bad:
            raise RuntimeError(f"rollout collection worker(s) exited with code {bad}")
        if not all_states:
            return [], []
        return list(torch.cat(all_states)), all_zs
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)


def pretrain_value(net: ResNet, games: int, epochs: int, batch_size: int,
                   lr: float, weight_decay: float, seed: int,
                   tail_plies: int = 8, rollouts: int = 20,
                   progress=None, device: str = "cpu",
                   collect_workers: int = 1) -> float:
    """Train the value head (and the shared trunk) on rollout labels.

    The policy head is frozen at its random init; self-play learns it later.
    `progress`, if given, is called after every batch. Returns the final MSE.

    `device` is where the training loop runs (net must already live there;
    samples are moved once). `collect_workers` > 1 collects rollouts in
    parallel processes (distribution-equal sharding).
    """
    if collect_workers > 1:
        states, zs = collect_rollout_samples_parallel(
            games, seed, collect_workers, tail_plies, rollouts
        )
    else:
        states, zs = collect_rollout_samples(games, seed, tail_plies, rollouts)
    n = len(states)
    if n == 0:
        raise ValueError("rollout collection produced no samples")
    S = torch.stack(states).to(device)
    Z = torch.tensor(zs, dtype=torch.float32, device=device).unsqueeze(1)
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
        order = torch.randperm(n, device=device)
        for b in range(n_batches):
            idx = order[b * batch_size:(b + 1) * batch_size]
            z = Z[idx]
            s = S[idx].cpu()
            # D4 augmentation (value invariant, policy untouched) — CPU-side
            # per-sample transform, then one upload per batch.
            for i in range(s.shape[0]):
                s[i] = apply_d4(s[i], perms[int(torch.randint(8, (1,)).item())])
            s = s.to(device)
            _logits, value = net(s)
            loss = ((value - z) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final = loss.item()
            if progress is not None:
                progress()
    return final
