#!/usr/bin/env python3
"""Generate an opening book of solid board states by self-playing a model.

The deterministic arena (no dirichlet noise, temperature 0) plays the same
game from the initial board twice — once per color — so head-to-head results
are nearly binary. This tool instead self-plays a strong checkpoint WITH
root noise and the usual temperature schedule and harvests diverse, solid
positions: the first `--head` states of every game plus `--tail` random
later states. Collected states are deduplicated exactly (bitboards + side
to move) until `--target` unique states are found, then written to
model/openbook.json (atomically via a temp file + rename).

Example:
    python tools/make_openbook.py --checkpoint checkpoints/best1.pt
"""

import argparse
import contextlib
import json
import multiprocessing
import os
import queue as queue_mod
import random
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODEL_DIR))

import torch  # noqa: E402

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


def play_game_states(net, mcts_cfg):
    """Self-play one game with dirichlet root noise + the temperature
    schedule. Returns every NON-terminal state in move order; index i is the
    position after move i+1 (index 0 = after white's first move)."""
    state = initial_state()
    mcts = MCTS(net, mcts_cfg)
    states = []
    while not is_terminal(state):
        ply = len(states)
        temperature = 1.0 if ply < mcts_cfg.temperature_threshold else 0.0
        _pi, chosen, _root = mcts.root_policy(
            state, root_noise=True, temperature=temperature)
        x, z, _y = action_to_xyz(chosen)
        state = apply_move(state, x, z)
        if not is_terminal(state):
            states.append(state)
    return states


def sample_states(states, head: int, tail: int, rng: random.Random):
    """Pick the `head` consecutive opening states plus `tail` uniformly
    random later non-terminal states. Returns [] when the game ended too
    early to offer enough tail candidates."""
    if len(states) < head or len(states) - head < tail:
        return []
    picks = list(range(head))
    picks += rng.sample(range(head, len(states)), tail)
    return [states[i] for i in picks]


def _load_net_state(checkpoint: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net_state = payload.get("net_state")
    saved_network = payload.get("network")
    if net_state is None or saved_network is None:
        raise SystemExit(f"ERROR: {checkpoint} has no net_state/network payload")
    return net_state, NetworkConfig(**saved_network), payload.get("iteration", "?")


def gen_worker(net_state, net_cfg, mcts_cfg, head, tail, num_threads, in_q, out_q):
    """Process entry point: rebuild the net, then answer round requests
    ((games, seed) tuples) from in_q with harvested GameStates until the
    sentinel None arrives. Errors ship in-band, never kill the parent."""
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
            n_games, worker_seed = req
            rng = random.Random(worker_seed)
            harvested = []
            for _ in range(n_games):
                states = play_game_states(net, mcts_cfg)
                harvested.extend(sample_states(states, head, tail, rng))
            out_q.put(("ok", harvested))
    except Exception as exc:  # noqa: BLE001 — report, never crash the parent
        try:
            out_q.put(("err", f"{type(exc).__name__}: {exc}"))
        except Exception:  # noqa: BLE001
            pass


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate an opening book by self-playing a checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="model snapshot to self-play (e.g. checkpoints/best1.pt)")
    parser.add_argument("--config", default=str(MODEL_DIR / "config.toml"))
    parser.add_argument("--target", type=int, default=1000,
                        help="number of unique book states to collect")
    parser.add_argument("--head", type=int, default=6,
                        help="consecutive post-move states kept per game (ply 1..head)")
    parser.add_argument("--tail", type=int, default=4,
                        help="random later non-terminal states sampled per game")
    parser.add_argument("--sims", type=int, default=None,
                        help="MCTS simulations per move (default: [mcts].simulations)")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel game producers (default: min(cpus, config workers))")
    parser.add_argument("--max-games", type=int, default=None,
                        help="hard cap on self-played games (default: 2x target)")
    parser.add_argument("--seed", type=int, default=None,
                        help="sampling seed (default: [training].seed)")
    args = parser.parse_args(argv)

    if min(args.target, args.head, args.tail) < 1:
        parser.error("--target, --head and --tail must all be >= 1")

    cfg = load_config(args.config)
    if args.sims is not None:
        mcts_cfg = type(cfg.mcts)(**{**cfg.mcts.__dict__, "simulations": args.sims})
    else:
        mcts_cfg = cfg.mcts

    cpus = os.cpu_count() or 1
    workers = args.workers if args.workers is not None \
        else min(cpus, cfg.training.workers)
    workers = max(1, min(workers, cpus))
    max_games = args.max_games if args.max_games is not None else 2 * args.target
    seed = args.seed if args.seed is not None else cfg.training.seed

    net_state, net_cfg, best_iteration = _load_net_state(args.checkpoint)
    games_needed_est = -(-args.target // (args.head + args.tail))

    print("=" * 62)
    print("opening book generation")
    print("-" * 62)
    print(f"  checkpoint : {args.checkpoint} (iteration {best_iteration})")
    print(f"  network    : blocks={net_cfg.blocks} channels={net_cfg.base_channels}")
    print(f"  sims/move  : {mcts_cfg.simulations}"
          f"  (noise + tau=1 for first {mcts_cfg.temperature_threshold} plies)")
    print(f"  target     : {args.target} unique states"
          f"  (est. >= {games_needed_est} games at {args.head}+{args.tail}/game)")
    print(f"  sampling   : head={args.head} consecutive + tail={args.tail} random")
    print(f"  workers    : {workers}   cap: {max_games} games   seed: {seed}")
    print(f"  output     : {OPENBOOK_PATH}")
    print("=" * 62)

    ctx = multiprocessing.get_context("spawn") if workers > 1 else None
    unique: dict = {}          # book key -> GameState
    games_done = 0
    t0 = time.perf_counter()
    stop = False

    def absorb(states):
        for s in states:
            unique.setdefault(book_key(s), s)

    def progress():
        print(
            f"\r  games {games_done:>5}  unique {len(unique):>5}/{args.target}"
            f"  elapsed {time.perf_counter() - t0:6.0f}s",
            end="", flush=True)

    if workers == 1:
        torch.set_num_threads(cpus)
        net = ResNet(net_cfg)
        net.load_state_dict(net_state)
        net.eval()
        rng = random.Random(seed)
        while not stop:
            absorb(sample_states(play_game_states(net, mcts_cfg),
                                 args.head, args.tail, rng))
            games_done += 1
            progress()
            stop = len(unique) >= args.target or games_done >= max_games
    else:
        in_q = ctx.Queue()
        out_q = ctx.Queue()
        num_threads = worker_num_threads(workers)
        procs = []
        try:
            for _ in range(workers):
                p = ctx.Process(
                    target=gen_worker,
                    args=(net_state, net_cfg, mcts_cfg, args.head, args.tail,
                          num_threads, in_q, out_q),
                    daemon=True,
                )
                p.start()
                procs.append(p)
            dispatched = 0
            while not stop:
                want = min(len(procs), max_games - dispatched)
                if want <= 0:
                    break
                for _ in range(want):
                    dispatched += 1
                    in_q.put((1, seed + dispatched))
                outstanding = want
                got = 0
                errored = None
                while got < outstanding and errored is None:
                    try:
                        status, payload = out_q.get(timeout=120)
                    except queue_mod.Empty:
                        if all(not p.is_alive() for p in procs):
                            raise RuntimeError(
                                f"generator workers died after {got}/{outstanding}"
                                " results of this round")
                        continue
                    if status == "err":
                        errored = payload
                        break
                    absorb(payload)
                    got += 1
                    games_done += 1
                    progress()
                if errored is not None:
                    raise RuntimeError(f"generator worker failed: {errored}")
                stop = len(unique) >= args.target or games_done >= max_games
        finally:
            with contextlib.suppress(Exception):
                for _ in procs:
                    in_q.put(None)
            for p in procs:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()

    print()
    if len(unique) < args.target:
        raise SystemExit(
            f"ERROR: only {len(unique)} unique states after {games_done} games "
            f"(cap {max_games}); the model may be producing near-identical "
            "openings. Raise --max-games or lower --target.")

    by_key = {book_key(s): s for s in unique.values()}
    rendered = [state_to_entry(by_key[k]) for k in sorted(by_key)]
    # Round-trip check before writing anything: what we write must load back.
    for i, entry in enumerate(rendered):
        entry_to_state(entry, index=i)

    tmp = OPENBOOK_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(rendered, f, separators=(",", ":"))
    os.replace(tmp, OPENBOOK_PATH)

    dt = time.perf_counter() - t0
    print(f"done: {len(rendered)} unique states from {games_done} games "
          f"in {dt / 60:.1f} min -> {OPENBOOK_PATH}")


if __name__ == "__main__":
    main()
