"""Per-phase performance benchmarks for the smart-four trainer.

Measures, for a given config and device:
  net        raw forward throughput (states/s) at several batch sizes
  selfplay   games/s via the real worker topology (baseline CPU per-worker
             nets vs central-server), with per-component timing inside MCTS
  optimize   batches/s of the real training step
  arena      games/s via the real arena worker topology

Usage:
  python tools/bench.py --config config.toml --device mps
  python tools/bench.py --config config.toml --device cpu --phases selfplay,optimize
"""

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from smartfour.config import load_config
from smartfour.device import resolve_device
from smartfour.network import ResNet


# ------------------------------------------------------------------ helpers

def _fmt_row(label, value, unit=""):
    return f"  {label:<34s} {value:>12s}{unit}"


def bench_net(cfg, device):
    """Raw forward throughput at several batch sizes."""
    net = ResNet(cfg.network).to(device).eval()
    rows = []
    for b in (32, 128, 512, 1024):
        x = torch.randn(b, 16, 5, 5, device=device)
        with torch.no_grad():
            for _ in range(3):
                net(x)
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            iters = 30
            for _ in range(iters):
                net(x)
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
            dt = (time.perf_counter() - t0) / iters
        rows.append((b, dt * 1000, b / dt))
    return rows


def bench_optimize(cfg, device, n_batches=40):
    """The real training step: sample -> forward -> backward -> AdamW."""
    from smartfour.encode import encode
    from smartfour.game import apply_move, initial_state
    from smartfour.network import loss_components

    torch.manual_seed(0)
    net = ResNet(cfg.network).to(device)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.training.learning_rate,
                            weight_decay=cfg.training.weight_decay)
    # Synthetic replay buffer of encoded states
    states, pis = [], []
    import random
    rng = random.Random(7)
    s = initial_state()
    legal = 25
    for _ in range(cfg.training.batch_size * 4):
        states.append(encode(s))
        pi = torch.zeros(125)
        idx = rng.randrange(25)
        pi[idx] = 1.0
        pis.append(pi)
        x, z = rng.randrange(5), rng.randrange(5)
        from smartfour.game import legal_moves
        mv = legal_moves(s)
        if not mv or s.winner is not None:
            s = initial_state()
            continue
        x, z = rng.choice(mv)
        s = apply_move(s, x, z)
    S = torch.stack(states)
    PI = torch.stack(pis)
    Z = torch.randn(len(S), 1)
    t0 = time.perf_counter()
    for i in range(n_batches):
        j = i * cfg.training.batch_size % (len(S) - cfg.training.batch_size)
        s = S[j:j + cfg.training.batch_size].to(device)
        pi = PI[j:j + cfg.training.batch_size].to(device)
        z = Z[j:j + cfg.training.batch_size].to(device)
        logits, value = net(s)
        _pol, _val, loss = loss_components(logits, value, pi, z)
        opt.zero_grad()
        loss.backward()
        opt.step()
    dt = time.perf_counter() - t0
    return n_batches / dt, dt / n_batches * 1000


def bench_pretrain(cfg, device, games, workers, epochs=2):
    """The real pretrain pass (collect + value-head training) at reduced
    epochs. Reports collection seconds, training ms/batch, total samples."""
    from smartfour.pretrain import (
        collect_rollout_samples,
        collect_rollout_samples_parallel,
        pretrain_value,
    )

    t0 = time.perf_counter()
    seq_states, seq_zs = collect_rollout_samples(games, seed=991)
    seq_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    par_states, par_zs = collect_rollout_samples_parallel(games, 991, workers)
    par_s = time.perf_counter() - t0

    net = ResNet(cfg.network).to(device)
    net.train()
    t0 = time.perf_counter()
    mse = pretrain_value(
        net, games, epochs, cfg.training.batch_size, cfg.training.pretrain_lr,
        cfg.training.weight_decay, seed=991, device=device,
        collect_workers=workers,
    )
    train_s = time.perf_counter() - t0
    n = len(par_states)
    n_batches = max(1, n // cfg.training.batch_size) * epochs
    return {
        "games": games,
        "samples": n,
        "collect_sequential_s": seq_s,
        "collect_parallel_s": par_s,
        "train_s": train_s,
        "train_ms_per_batch": train_s / n_batches * 1000,
        "final_mse": mse,
    }


# ------------------------------------------------------------------ self-play

def _bench_selfplay_worker(net_state, net_cfg, mcts_cfg, tt, games, seed,
                           num_threads, out_q, server_addr):
    """Play `games` with real play_game; ship per-game wall time."""
    from smartfour.selfplay import play_game
    ignore = None
    torch.manual_seed(seed)
    if num_threads:
        torch.set_num_threads(max(1, int(num_threads)))
    from smartfour.network import ResNet
    net = ResNet(net_cfg)
    net.load_state_dict(net_state)
    net.eval()
    evaluator = None
    if server_addr is not None:
        from smartfour.inference_server import RemoteEvaluator
        evaluator = RemoteEvaluator(server_addr, slot=0)
    t0 = time.perf_counter()
    plies = 0
    try:
        for _ in range(games):
            samples, _w = play_game(net, mcts_cfg, tt, evaluator=evaluator)
            plies += len(samples)
        out_q.put(("ok", plies, time.perf_counter() - t0))
    except Exception as exc:  # noqa: BLE001
        out_q.put(("err", f"{type(exc).__name__}: {exc}", 0))
    finally:
        if evaluator is not None:
            evaluator.close()


def bench_selfplay(cfg, device, games, workers, server):
    """Spawn the real worker topology; returns games/s and total plies."""
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    net = ResNet(cfg.network)
    net.eval()
    net_state = {k: v.cpu() for k, v in net.state_dict().items()}
    from smartfour.selfplay import worker_num_threads
    nt = worker_num_threads(workers)
    addr = server.address if server else None
    counts = []
    per = games // workers
    for i in range(workers):
        counts.append(per + (1 if i < games % workers else 0))
    procs = []
    t0 = time.perf_counter()
    try:
        for i, n in enumerate(counts):
            if n == 0:
                continue
            p = ctx.Process(
                target=_bench_selfplay_worker,
                args=(net_state, cfg.network, cfg.mcts,
                      cfg.mcts.temperature_threshold, n, 1000 + i, nt,
                      out_q, addr),
                daemon=True,
            )
            p.start()
            procs.append(p)
        total_plies = 0
        errors = []
        got = 0
        expect = len([c for c in counts if c > 0])
        while got < expect:
            msg = out_q.get(timeout=1800)
            got += 1
            if msg[0] == "err":
                errors.append(msg[1])
            else:
                total_plies += msg[1]
        if errors:
            raise RuntimeError(f"bench worker failed: {errors[0]}")
        dt = time.perf_counter() - t0
        return games / dt, total_plies, dt
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=10)


# ------------------------------------------------------------------ arena

def _bench_arena_worker(net_a_state, net_b_state, net_cfg, mcts_cfg, games,
                        start, seed, num_threads, out_q, server_addr):
    from smartfour.arena import _play_two
    from smartfour.network import ResNet
    torch.manual_seed(seed)
    if num_threads:
        torch.set_num_threads(max(1, int(num_threads)))
    net_a = ResNet(net_cfg); net_a.load_state_dict(net_a_state); net_a.eval()
    net_b = ResNet(net_cfg); net_b.load_state_dict(net_b_state); net_b.eval()
    ev_a = ev_b = None
    if server_addr is not None:
        from smartfour.inference_server import RemoteEvaluator
        ev_a = RemoteEvaluator(server_addr, slot=0)
        ev_b = RemoteEvaluator(server_addr, slot=1)
    t0 = time.perf_counter()
    plies = 0
    try:
        for j in range(games):
            a_is_white = (start + j) % 2 == 0
            if a_is_white:
                _r, p = _play_two(net_a, net_b, mcts_cfg, ev_a, ev_b)
            else:
                _r, p = _play_two(net_b, net_a, mcts_cfg, ev_b, ev_a)
            plies += p
        out_q.put(("ok", plies, time.perf_counter() - t0))
    except Exception as exc:  # noqa: BLE001
        out_q.put(("err", f"{type(exc).__name__}: {exc}", 0))
    finally:
        if ev_a is not None:
            ev_a.close()
        if ev_b is not None:
            ev_b.close()


def bench_arena(cfg, device, games, workers, server):
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    net = ResNet(cfg.network).eval()
    net_state = {k: v.cpu() for k, v in net.state_dict().items()}
    from smartfour.selfplay import worker_num_threads
    nt = worker_num_threads(workers)
    addr = server.address if server else None
    counts = []
    per = games // workers
    for i in range(workers):
        counts.append(per + (1 if i < games % workers else 0))
    procs = []
    t0 = time.perf_counter()
    try:
        start = 0
        for i, n in enumerate(counts):
            if n == 0:
                continue
            p = ctx.Process(
                target=_bench_arena_worker,
                args=(net_state, net_state, cfg.network, cfg.mcts, n, start,
                      2000 + i, nt, out_q, addr),
                daemon=True,
            )
            p.start()
            procs.append(p)
            start += n
        total_plies = 0
        got = 0
        expect = len([c for c in counts if c > 0])
        while got < expect:
            msg = out_q.get(timeout=1800)
            got += 1
            if msg[0] == "err":
                raise RuntimeError(f"arena bench worker failed: {msg[1]}")
            total_plies += msg[1]
        dt = time.perf_counter() - t0
        return games / dt, total_plies, dt
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=10)


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--device", default=None,
                    help="cpu | mps | cuda | auto (default: from config)")
    ap.add_argument("--phases", default="net,selfplay,optimize,arena,pretrain")
    ap.add_argument("--selfplay-games", type=int, default=None)
    ap.add_argument("--arena-games", type=int, default=None)
    ap.add_argument("--pretrain-games", type=int, default=200,
                    help="games for the pretrain collection benchmark")
    ap.add_argument("--workers", type=int, default=None,
                    help="override training.workers for self-play/arena")
    ap.add_argument("--json", action="store_true", help="emit JSON results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device if args.device else cfg.device.name)
    workers = args.workers or cfg.training.workers
    sp_games = args.selfplay_games or max(6, workers)  # a few per worker
    ar_games = args.arena_games or max(6, workers)
    phases = args.phases.split(",")

    results = {"device": device, "workers": workers, "phases": {}}
    print(f"config {args.config}  device {device}  workers {workers}")

    server = None
    if "selfplay" in phases or "arena" in phases:
        if device in ("mps", "cuda"):
            from smartfour.inference_server import InferenceServerHandle
            net = ResNet(cfg.network).eval()
            st = {k: v.cpu() for k, v in net.state_dict().items()}
            server = InferenceServerHandle(cfg.network, device, slots=2).start(
                initial_states=[st, st]
            )

    try:
        if "net" in phases:
            rows = bench_net(cfg, device)
            print("[net forward]")
            for b, ms, tput in rows:
                print(_fmt_row(f"batch {b:>4d}", f"{ms:8.2f}", " ms")
                      + f"   {tput:9.0f} states/s")
            results["phases"]["net"] = [
                {"batch": b, "ms": ms, "states_per_s": tput} for b, ms, tput in rows
            ]
        if "selfplay" in phases:
            gps, plies, dt = bench_selfplay(cfg, device, sp_games, workers, server)
            print("[self-play]")
            print(_fmt_row("throughput", f"{gps:8.3f}", " games/s")
                  + f"   {plies:6d} plies in {dt:.1f}s")
            results["phases"]["selfplay"] = {
                "games_per_s": gps, "plies": plies, "seconds": dt,
                "games": sp_games,
            }
        if "optimize" in phases:
            bps, ms = bench_optimize(cfg, device)
            print("[optimize]")
            print(_fmt_row("throughput", f"{bps:8.2f}", " batches/s")
                  + f"   {ms:.1f} ms/batch (batch {cfg.training.batch_size})")
            results["phases"]["optimize"] = {"batches_per_s": bps, "ms_per_batch": ms}
        if "arena" in phases:
            gps, plies, dt = bench_arena(cfg, device, ar_games, workers, server)
            print("[arena]")
            print(_fmt_row("throughput", f"{gps:8.3f}", " games/s")
                  + f"   {plies:6d} plies in {dt:.1f}s")
            results["phases"]["arena"] = {
                "games_per_s": gps, "plies": plies, "seconds": dt,
                "games": ar_games,
            }
        if "pretrain" in phases:
            pt = bench_pretrain(cfg, device, args.pretrain_games, workers)
            print("[pretrain]")
            print(_fmt_row("collect sequential", f"{pt['collect_sequential_s']:8.1f}", " s")
                  + f"   {pt['samples']} samples / {pt['games']} games")
            print(_fmt_row("collect parallel", f"{pt['collect_parallel_s']:8.1f}", " s")
                  + f"   {pt['collect_sequential_s'] / max(pt['collect_parallel_s'], 1e-9):.1f}x")
            print(_fmt_row("train", f"{pt['train_ms_per_batch']:8.1f}", " ms/batch")
                  + f"   total {pt['train_s']:.1f}s, final MSE {pt['final_mse']:.4f}")
            results["phases"]["pretrain"] = pt
    finally:
        if server is not None:
            server.shutdown()

    if args.json:
        print(json.dumps(results))


if __name__ == "__main__":
    main()
