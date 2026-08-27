#!/usr/bin/env python3
"""Generate an opening book of contested board states by greedy self-play.

A noisy self-play model produces "dumb" states where one side has already
decided the game, which collapses arena results to ~0.5. This tool instead
self-plays a strong checkpoint greedily — MCTS with `--sims` simulations per
move, NO dirichlet root noise and temperature 0 (visit-count ties still
break randomly, seeded per game) — and harvests only states that are still
open: at least `--m` plies away from the game end.

Each iteration plays 25 games, White's first move forced to a different one
of the 25 board columns per game (fixed order). From every game the ply-1
state is dropped and only states at plies 2..L-m survive an L-move game.
States are deduplicated exactly (bitboards + side to move) and merged
across iterations; the book is written to model/openbook.json after every
iteration (atomic tmp + rename), sorted by ply depth, shallowest first.
Generation stops at `--target` unique states or when an iteration adds no
new state.

Example:
    python tools/make_openbook.py --checkpoint checkpoints/best1.pt
"""

import argparse
import contextlib
import json
import multiprocessing
import os
import queue as queue_mod
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODEL_DIR))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from smartfour.config import NetworkConfig, load_config  # noqa: E402
from smartfour.encode import action_to_xyz  # noqa: E402
from smartfour.game import apply_move, initial_state, is_terminal  # noqa: E402
from smartfour.mcts import MCTS  # noqa: E402
from smartfour.network import ResNet  # noqa: E402
from smartfour.openbook import (  # noqa: E402
    book_key,
    entry_to_state,
    state_to_entry,
)
from smartfour.selfplay import ignore_sigint, worker_num_threads  # noqa: E402
OPENBOOK_PATH = MODEL_DIR / "openbook.json"
BOARD_COLUMNS = 25  # 5x5 drop columns; game g opens on column g


def state_ply(state) -> int:
    """Ply depth of a state: pieces placed since the initial board."""
    return bin(state.white | state.black).count("1")


def play_game_states(net, mcts_cfg, first_move, torch_seed: int):
    """Greedy self-play one game: no root noise, temperature 0 (random
    visit-tie break, seeded by `torch_seed`). White's first move is forced
    to `first_move` = (x, z). Returns every NON-terminal state in move
    order; index i is the position after move i+1 (index 0 = after the
    forced opening move)."""
    torch.manual_seed(torch_seed)
    x, z = first_move
    state = apply_move(initial_state(), x, z)
    mcts = MCTS(net, mcts_cfg)
    states = [state]  # ply-1 state; harvest_states drops it
    while not is_terminal(state):
        _pi, chosen, _root = mcts.root_policy(
            state, root_noise=False, temperature=0.0)
        mx, mz, _my = action_to_xyz(chosen)
        state = apply_move(state, mx, mz)
        if not is_terminal(state):
            states.append(state)
    return states


def harvest_states(states, m: int):
    """Drop the ply-1 state and keep only states still open: at least `m`
    plies from the game end. An L-move game keeps plies 2..L-m (a 13-move
    game with m=8 keeps plies 2..5)."""
    total_moves = len(states) + 1  # + terminal move
    return [s for i, s in enumerate(states)
            if i + 1 >= 2 and total_moves - (i + 1) >= m]


def _load_net_state(checkpoint: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net_state = payload.get("net_state")
    saved_network = payload.get("network")
    if net_state is None or saved_network is None:
        raise SystemExit(f"ERROR: {checkpoint} has no net_state/network payload")
    return net_state, NetworkConfig(**saved_network), payload.get("iteration", "?")


def gen_worker(net_state, net_cfg, mcts_cfg, m, num_threads, in_q, out_q):
    """Process entry point: rebuild the net, then answer round requests
    ((iteration, game_idx, torch_seed) tuples) from in_q with harvested
    states until the sentinel None arrives. Errors ship in-band, never kill
    the parent."""
    ignore_sigint()
    try:
        if num_threads:
            torch.set_num_threads(max(1, int(num_threads)))
        net = ResNet(net_cfg)
        net.load_state_dict(net_state)
        net.eval()
        while True:
            req = in_q.get()
            if req is None:
                return
            _iteration, game_idx, torch_seed = req
            states = play_game_states(net, mcts_cfg,
                                      (game_idx % 5, game_idx // 5), torch_seed)
            out_q.put(("ok", harvest_states(states, m)))
    except Exception as exc:  # noqa: BLE001 — report, never crash the parent
        try:
            out_q.put(("err", f"{type(exc).__name__}: {exc}"))
        except Exception:  # noqa: BLE001
            pass


def write_book(unique: dict, target: int) -> None:
    """Render the book sorted by ply depth (shallowest first, book_key as a
    deterministic tie-break), capped at `target` entries, and write it
    atomically after a round-trip check: what we write must load back."""
    ordered = sorted(unique.values(),
                     key=lambda s: (state_ply(s), book_key(s)))[:target]
    rendered = [state_to_entry(s) for s in ordered]
    for i, entry in enumerate(rendered):
        entry_to_state(entry, index=i)
    tmp = OPENBOOK_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(rendered, f, separators=(",", ":"))
    os.replace(tmp, OPENBOOK_PATH)


def plies_hist(states) -> str:
    counts: dict = {}
    for s in states:
        counts[state_ply(s)] = counts.get(state_ply(s), 0) + 1
    return " ".join(f"p{k}:{v}" for k, v in sorted(counts.items()))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate an opening book by greedy self-play of a checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",
                        default=str(MODEL_DIR / "checkpoints" / "best1.pt"),
                        help="model snapshot to self-play")
    parser.add_argument("--config", default=str(MODEL_DIR / "config.toml"))
    parser.add_argument("--target", type=int, default=250,
                        help="number of unique book states to collect")
    parser.add_argument("--m", type=int, default=8,
                        help="keep only states with at least this many plies "
                             "left to the game end")
    parser.add_argument("--sims", type=int, default=500,
                        help="MCTS simulations per move (no noise, tau=0)")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel game producers (default: min(8, cpus))")
    parser.add_argument("--seed", type=int, default=None,
                        help="base RNG seed for move tie-breaks "
                             "(default: [training].seed)")
    args = parser.parse_args(argv)

    if args.target < 1 or args.m < 1 or args.sims < 1:
        parser.error("--target, --m and --sims must all be >= 1")

    cfg = load_config(args.config)
    mcts_cfg = type(cfg.mcts)(**{**cfg.mcts.__dict__, "simulations": args.sims})
    cpus = os.cpu_count() or 1
    workers = args.workers if args.workers is not None else min(8, cpus)
    workers = max(1, min(workers, cpus))
    seed = args.seed if args.seed is not None else cfg.training.seed

    net_state, net_cfg, best_iteration = _load_net_state(args.checkpoint)

    print("=" * 62)
    print("opening book generation (greedy self-play)")
    print("-" * 62)
    print(f"  network    : blocks={net_cfg.blocks} channels={net_cfg.base_channels}")
    print(f"  search     : {mcts_cfg.simulations} sims/move, "
          "no root noise, temperature 0 (random tie-break)")
    print(f"  per iter   : {BOARD_COLUMNS} games, forced 1st ply at each of "
          f"the 25 columns")
    print(f"  filter     : drop ply 1, keep states >= {args.m} plies from end")
    print(f"  target     : {args.target} unique states   seed: {seed}")
    print(f"  workers    : {workers}   output: {OPENBOOK_PATH}")
    print("=" * 62, flush=True)

    unique: dict = {}          # book key -> GameState
    t0 = time.perf_counter()

    def absorb(states) -> int:
        new = 0
        for s in states:
            if book_key(s) not in unique:
                unique[book_key(s)] = s
                new += 1
        return new

    def play_round(iteration: int) -> int:
        """Play the 25 forced-opening games of one iteration; return the
        number of newly collected unique states."""
        seeds = [seed + iteration * 1000 + g for g in range(BOARD_COLUMNS)]
        if workers == 1:
            torch.set_num_threads(cpus)
            net = ResNet(net_cfg)
            net.load_state_dict(net_state)
            net.eval()
            harvested = []
            with tqdm(total=BOARD_COLUMNS, desc=f"iter {iteration}", unit="game") as bar:
                for g in range(BOARD_COLUMNS):
                    harvested.append(harvest_states(
                        play_game_states(net, mcts_cfg, (g % 5, g // 5), seeds[g]),
                        args.m))
                    bar.set_postfix(unique=len(unique))
                    bar.update(1)
        else:
            ctx = multiprocessing.get_context("spawn")
            in_q = ctx.Queue()
            out_q = ctx.Queue()
            num_threads = worker_num_threads(workers)
            procs = []
            try:
                for _ in range(workers):
                    p = ctx.Process(
                        target=gen_worker,
                        args=(net_state, net_cfg, mcts_cfg, args.m,
                              num_threads, in_q, out_q),
                        daemon=True,
                    )
                    p.start()
                    procs.append(p)
                for g in range(BOARD_COLUMNS):
                    in_q.put((iteration, g, seeds[g]))
                harvested = []
                got, errored = 0, None
                with tqdm(total=BOARD_COLUMNS, desc=f"iter {iteration}",
                          unit="game") as bar:
                    while got < BOARD_COLUMNS and errored is None:
                        try:
                            status, payload = out_q.get(timeout=300)
                        except queue_mod.Empty:
                            if all(not p.is_alive() for p in procs):
                                raise RuntimeError(
                                    f"generator workers died after {got}/"
                                    f"{BOARD_COLUMNS} games of this round")
                            continue
                        if status == "err":
                            errored = payload
                            break
                        harvested.append(payload)
                        got += 1
                        bar.set_postfix(unique=len(unique))
                        bar.update(1)
                if errored is not None:
                    raise RuntimeError(f"generator worker failed: {errored}")
            finally:
                with contextlib.suppress(Exception):
                    for _ in procs:
                        in_q.put(None)
                for p in procs:
                    p.join(timeout=5)
                    if p.is_alive():
                        p.terminate()
        new = sum(absorb(states) for states in harvested)
        write_book(unique, args.target)
        return new

    iteration = 0
    while len(unique) < args.target:
        new = play_round(iteration)
        unique_states = list(unique.values())
        print(f"iter {iteration}: +{new} new, book {len(unique)}/{args.target} "
              f"[{plies_hist(unique_states)}]  "
              f"elapsed {time.perf_counter() - t0:.0f}s", flush=True)
        if new == 0:
            print("no progress this iteration; stopping")
            break
        iteration += 1

    dt = time.perf_counter() - t0
    if len(unique) < args.target:
        raise SystemExit(
            f"ERROR: only {len(unique)} unique states after {iteration + 1} "
            f"iterations ({dt / 60:.1f} min); greedy play repeats itself. "
            "Lower --target or raise --sims.")
    print(f"done: {len(unique)} unique states in {dt / 60:.1f} min "
          f"-> {OPENBOOK_PATH}")


if __name__ == "__main__":
    main()
