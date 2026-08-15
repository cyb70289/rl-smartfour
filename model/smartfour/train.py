"""AlphaZero training loop: self-play -> replay buffer -> optimize -> arena.

Checkpoints (checkpoint_dir/):
  latest.pt    full state (net, optimizer, best_net, replay buffer) — the
               only per-iteration checkpoint; written atomically after every
               completed iteration, so it always holds the last completed
               iteration.
  best.pt      slim inference snapshot of the arena-best net (weights +
               iteration the best was set). Never used for resume.
All writes are atomic (temp file + os.replace), so an interrupt or crash
mid-save never corrupts an existing checkpoint. Older versions also wrote
iter_NNNN.pt snapshots per iteration; those are no longer created, and stale
ones can be deleted manually or wiped with --restart.

Resume is the default and needs no flag: latest.pt -> fresh start. A
corrupt/unreadable checkpoint or a network-config mismatch is a hard error
(never a silent fallback); --restart wipes the checkpoint dir
(after confirmation) to start over. --iterations N is a target: train until
iteration N, exit immediately when already there; without it, train forever
until SIGINT/SIGTERM. An interrupt discards the in-flight iteration (no
save), so latest.pt always holds the last completed iteration.
"""

import argparse
import json
import multiprocessing
import os
import queue
import signal
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from tqdm import tqdm

from .arena import arena_worker, play_arena
from .diagnostics import aggregate_games, buffer_stats, format_lines
from .device import resolve_device, VALID_DEVICES
from .config import Config, load_config
from .encode import apply_d4, apply_d4_policy, d4_perms
from .game import DRAW, WHITE
from .network import ResNet, loss_components
from .pretrain import pretrain_value
from .selfplay import (
    play_game,
    samples_from_ipc,
    selfplay_worker,
    split_games,
    worker_num_threads,
)


def _tqdm(*args, **kwargs):
    """Progress bar that renders even when the terminal window size is unknown.

    tqdm >= 4.69 derives ncols/nrows from the terminal size minus one; a pty
    without a window size (CI, some tmux/ssh setups) then reports (-1, -1),
    which makes tqdm skip rendering the bar entirely. Normalize degenerate
    sizes to None so the bar falls back to default formatting.
    """
    kwargs.setdefault("disable", None)  # render only on a TTY
    bar = tqdm(*args, **kwargs)
    ncols = getattr(bar, "ncols", None)
    nrows = getattr(bar, "nrows", None)
    if (ncols is not None and ncols < 0) or (nrows is not None and nrows < 0):
        bar.ncols = None
        bar.nrows = None
    return bar


def plys_postfix(plies: int, games: int) -> str:
    """Average plies per game over the current self-play phase, for the bar.

    One ply is one move by one player (a turn is two plies), so a game that
    stored `len(samples)` positions ran exactly that many plies.
    """
    return f"{plies / games:.0f} plys/game" if games else "0 plys/game"


def _cpu_optim_state(state: dict) -> dict:
    """Optimizer state with every tensor moved to CPU (portable checkpoints)."""
    out = {}
    for k, v in state.items():
        if k == "state":
            out[k] = {
                pid: {
                    pk: pv.cpu() if isinstance(pv, torch.Tensor) else pv
                    for pk, pv in pstate.items()
                }
                for pid, pstate in v.items()
            }
        elif k == "param_groups":
            out[k] = v
        else:
            out[k] = v
    return out


# ------------------------------------------------------------- checkpoint I/O

def _atomic_save(payload, path) -> None:
    """torch.save to a temp file in the same directory, then atomically rename.

    An interrupt or crash mid-write leaves the previous checkpoint intact
    instead of a truncated/corrupt file.
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _terminate_workers(procs) -> None:
    """Terminate and join workers, immune to further signals.

    Ctrl-C during teardown must not skip joins and orphan workers, so the
    signals stay blocked until every process is reaped.
    """
    signal.pthread_sigmask(signal.SIG_BLOCK, (signal.SIGINT, signal.SIGTERM))
    try:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=30)
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))


def _find_resume_checkpoint(checkpoint_dir) -> Path | None:
    """Resume anchor: latest.pt, else None (fresh start)."""
    checkpoint_dir = Path(checkpoint_dir)
    latest = checkpoint_dir / "latest.pt"
    return latest if latest.exists() else None


def _confirm_and_wipe(checkpoint_dir, force: bool) -> None:
    """Delete every file under checkpoint_dir, after confirmation unless --yes.

    A non-terminal stdin never wipes without an explicit --yes; a declined
    or empty answer aborts with exit code 1.
    """
    checkpoint_dir = Path(checkpoint_dir)
    files = (
        sorted(p for p in checkpoint_dir.iterdir() if p.is_file())
        if checkpoint_dir.exists()
        else []
    )
    total = sum(p.stat().st_size for p in files)
    if not files:
        print(f"Nothing to delete in {checkpoint_dir}/; training fresh")
        return
    if force:
        _delete_files(files, total)
        return
    if not sys.stdin.isatty():
        raise SystemExit(
            f"ERROR: --restart would delete {len(files)} file(s) "
            f"({total / 1e6:.1f} MB) in {checkpoint_dir}/,\n"
            "  but stdin is not a terminal. Re-run interactively or pass --yes."
        )
    answer = input(
        f"Delete {len(files)} file(s) ({total / 1e6:.1f} MB) in {checkpoint_dir}/? [y/N] "
    )
    if answer.strip().lower() not in ("y", "yes"):
        print("Aborted.")
        raise SystemExit(1)
    _delete_files(files, total)


def _delete_files(files, total: int) -> None:
    for f in files:
        f.unlink()
    print(f"Deleted {len(files)} file(s) ({total / 1e6:.1f} MB)")


class ReplayBuffer:
    """Stores (state_tensor, pi, z); samples batches with optional D4 augmentation."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._s = []
        self._pi = []
        self._z = []

    def push(self, samples) -> None:
        for s, pi, _player, z in samples:
            self._s.append(s)
            self._pi.append(pi)
            self._z.append(z)
        if len(self._s) > self.capacity:
            drop = len(self._s) - self.capacity
            del self._s[:drop]
            del self._pi[:drop]
            del self._z[:drop]

    def __len__(self) -> int:
        return len(self._s)

    def sample(self, batch_size: int, augment: bool = True):
        idx = torch.randint(len(self), (batch_size,))
        s = torch.stack([self._s[i] for i in idx])
        pi = torch.stack([self._pi[i] for i in idx])
        z = torch.tensor([[self._z[i]] for i in idx], dtype=torch.float32)
        if augment:
            perms = d4_perms()
            for i in range(batch_size):
                perm = perms[int(torch.randint(8, (1,)).item())]
                s[i] = apply_d4(s[i], perm)
                pi[i] = apply_d4_policy(pi[i], perm)
        return s, pi, z

    def state(self):
        return self._s, self._pi, self._z

    def load_state(self, state) -> None:
        self._s, self._pi, self._z = state


class Trainer:
    def __init__(self, cfg: Config, device="cpu"):
        self.cfg = cfg
        self.device = device
        torch.manual_seed(cfg.training.seed)
        self.net = ResNet(cfg.network).to(device)
        self.best_net = ResNet(cfg.network).to(device)
        self.best_net.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        self.buffer = ReplayBuffer(cfg.training.replay_capacity)
        self.iteration = 0          # last *completed* iteration
        self.best_iteration = 0     # iteration whose net is the current best
        self.checkpoint_dir = Path(cfg.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._buf_hashes: set = set()  # tensor hashes of states pushed this run
        self.server = None          # InferenceServerHandle when accelerator-backed

    # ------------------------------------------------------------- internals

    def _selfplay(self, net, games: int | None = None) -> list:
        """Play `games` self-play games; returns the per-game stats dicts."""
        games = games if games is not None else self.cfg.training.selfplay_games
        workers = self.cfg.training.workers
        if workers <= 1:
            game_stats = []
            with _tqdm(total=games, desc="self-play", unit="game", leave=False) as bar:
                plies = 0
                for i in range(games):
                    stats: dict = {}
                    samples, _winner = play_game(
                        net, self.cfg.mcts, self.cfg.mcts.temperature_threshold,
                        stats_out=stats,
                    )
                    self.buffer.push(samples)
                    game_stats.append(stats)
                    plies += len(samples)
                    bar.set_postfix_str(plys_postfix(plies, i + 1))
                    bar.update(1)
            return game_stats
        with _tqdm(total=games, desc="self-play", unit="game", leave=False) as bar:
            return self._selfplay_parallel(net, games, workers, bar)

    def _selfplay_parallel(self, net, games: int, workers: int, bar) -> None:
        """Spawn one process per worker; each plays its share of games with a
        fresh net copy and ships samples over a queue. Fails fast if any
        worker errors or dies before delivering its games. Workers are
        daemonic (a dying parent cannot orphan them) and ignore SIGINT (the
        parent alone decides when to stop).
        """
        ctx = multiprocessing.get_context("spawn")
        out_q = ctx.Queue()
        net_state = {k: v.cpu() for k, v in net.state_dict().items()}
        num_threads = worker_num_threads(workers)
        procs = []
        try:
            for i, n in enumerate(split_games(games, workers)):
                if n == 0:
                    continue
                p = ctx.Process(
                    target=selfplay_worker,
                    args=(
                        net_state, self.cfg.network, self.cfg.mcts,
                        self.cfg.mcts.temperature_threshold, n,
                        self.cfg.training.seed + i + 1, num_threads, out_q,
                        self.server.address if self.server else None,
                    ),
                    daemon=True,
                )
                p.start()
                procs.append(p)
            return self._collect_selfplay(games, procs, out_q, bar)
        finally:
            _terminate_workers(procs)

    def _collect_selfplay(self, games: int, procs, out_q, bar) -> list:
        """Consume worker results until `games` games are pushed to the buffer.

        Raises RuntimeError when a worker reports failure or dies early, so a
        broken worker can never hang training or silently shrink the batch;
        surviving workers are terminated before the error propagates.
        Returns the per-game stats dicts shipped with the samples.
        """
        received = 0
        plies = 0
        game_stats = []
        try:
            while received < games:
                try:
                    msg = out_q.get(timeout=0.5)
                except queue.Empty:
                    if all(not p.is_alive() for p in procs):
                        raise RuntimeError(
                            f"self-play workers exited early: {received}/{games} games collected"
                        )
                    continue
                if (
                    isinstance(msg, tuple) and len(msg) == 2
                    and msg[0] == "__worker_error__"
                ):
                    raise RuntimeError(f"self-play worker failed: {msg[1]}")
                if isinstance(msg, tuple) and len(msg) == 2:
                    samples = samples_from_ipc(msg[0])
                    gstats = msg[1] if isinstance(msg[1], dict) else None
                else:
                    samples = samples_from_ipc(msg)
                    gstats = None
                self.buffer.push(samples)
                game_stats.append(gstats)
                plies += len(samples)
                received += 1
                bar.set_postfix_str(plys_postfix(plies, received))
                bar.update(1)
        except BaseException:
            _terminate_workers(procs)
            raise
        self._finish_workers(procs, "self-play")
        return game_stats

    def _finish_workers(self, procs, label: str) -> None:
        """Join successfully-collected workers; reap stragglers and fail on a
        nonzero exit code. Shared by the self-play and arena collectors."""
        for p in procs:
            p.join(timeout=60)
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
        bad = [p.exitcode for p in procs if p.exitcode != 0]
        if bad:
            raise RuntimeError(f"{label} worker(s) exited with code {bad}")

    def _optimize(self) -> tuple:
        """Gradient steps on the replay buffer; returns (mean, pol, val) losses."""
        if len(self.buffer) < self.cfg.training.batch_size:
            tqdm.write(
                f"WARNING: replay buffer too small ({len(self.buffer)} < "
                f"batch_size {self.cfg.training.batch_size}); skipping optimize"
            )
            return float("nan"), float("nan"), float("nan")
        self.net.train()  # MCTS leaves the net in eval mode; BN must update
        losses = []
        pols = []
        vals = []
        n_batches = max(1, len(self.buffer) // self.cfg.training.batch_size)
        total = n_batches * self.cfg.training.train_epochs
        with _tqdm(total=total, desc="optimize", unit="batch", leave=False) as bar:
            for _ in range(self.cfg.training.train_epochs):
                for _ in range(n_batches):
                    s, pi, z = self.buffer.sample(
                        self.cfg.training.batch_size,
                        augment=self.cfg.training.symmetry_augment,
                    )
                    s = s.to(self.device)
                    pi = pi.to(self.device)
                    z = z.to(self.device)
                    logits, value = self.net(s)
                    pol, val, loss = loss_components(logits, value, pi, z)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())
                    pols.append(pol.item())
                    vals.append(val.item())
                    recent = losses[-50:]
                    bar.set_postfix(
                        loss=f"{sum(recent) / len(recent):.4f}",
                        pol=f"{sum(pols[-50:]) / len(pols[-50:]):.3f}",
                        val=f"{sum(vals[-50:]) / len(vals[-50:]):.3f}",
                    )
                    bar.update(1)
        return (
            sum(losses) / len(losses) if losses else float("nan"),
            sum(pols) / len(pols) if pols else float("nan"),
            sum(vals) / len(vals) if vals else float("nan"),
        )

    def _arena(self, net_a, net_b, games: int):
        workers = self.cfg.training.workers
        if workers <= 1:
            with _tqdm(total=games, desc="arena", unit="game", leave=False) as bar:
                plies_out = []
                w, l, d = play_arena(
                    net_a, net_b, self.cfg.mcts, games,
                    progress=bar.update, plies_out=plies_out,
                )
                return w, l, d, sum(plies_out)
        with _tqdm(total=games, desc="arena", unit="game", leave=False) as bar:
            return self._arena_parallel(net_a, net_b, games, workers, bar)

    def _arena_parallel(self, net_a, net_b, games: int, workers: int, bar):
        """Spawn one process per worker; each plays its share of games with
        fresh copies of both nets and ships per-game results over a queue.
        Fails fast if any worker errors or dies before delivering its games.
        Workers are daemonic (a dying parent cannot orphan them) and ignore
        SIGINT (the parent alone decides when to stop).
        """
        ctx = multiprocessing.get_context("spawn")
        out_q = ctx.Queue()
        net_a_state = {k: v.cpu() for k, v in net_a.state_dict().items()}
        net_b_state = {k: v.cpu() for k, v in net_b.state_dict().items()}
        num_threads = worker_num_threads(workers)
        procs = []
        start = 0
        try:
            for i, n in enumerate(split_games(games, workers)):
                if n == 0:
                    continue
                p = ctx.Process(
                    target=arena_worker,
                    args=(
                        net_a_state, net_b_state, self.cfg.network, self.cfg.mcts,
                        n, start, self.cfg.training.seed + i + 1, num_threads, out_q,
                        self.server.address if self.server else None,
                    ),
                    daemon=True,
                )
                p.start()
                procs.append(p)
                start += n
            return self._collect_arena(games, procs, out_q, bar)
        finally:
            _terminate_workers(procs)

    def _collect_arena(self, games: int, procs, out_q, bar) -> tuple:
        """Consume worker results until `games` games are counted.

        Raises RuntimeError when a worker reports failure or dies early, so a
        broken worker can never silently shrink the arena; surviving workers
        are terminated before the error propagates. Results arrive in net_a's
        frame, so counting is the same as a sequential run.
        """
        a_wins = b_wins = draws = 0
        plies = 0
        received = 0
        try:
            while received < games:
                try:
                    msg = out_q.get(timeout=0.5)
                except queue.Empty:
                    if all(not p.is_alive() for p in procs):
                        raise RuntimeError(
                            f"arena workers exited early: {received}/{games} games collected"
                        )
                    continue
                if (
                    isinstance(msg, tuple) and len(msg) == 2
                    and msg[0] == "__worker_error__"
                ):
                    raise RuntimeError(f"arena worker failed: {msg[1]}")
                if isinstance(msg, tuple) and len(msg) == 2:
                    result, game_plies = msg
                    plies += game_plies
                else:
                    result = msg  # legacy plain-result message
                if result == DRAW:
                    draws += 1
                elif result == WHITE:
                    a_wins += 1
                else:
                    b_wins += 1
                received += 1
                bar.update(1)
        except BaseException:
            _terminate_workers(procs)
            raise
        self._finish_workers(procs, "arena")
        return a_wins, b_wins, draws, plies

    def _maybe_update_best(self, current_iteration: int) -> dict:
        games = self.cfg.training.eval_games
        wins, losses, draws, plies = self._arena(self.net, self.best_net, games)
        total = wins + losses + draws
        # Draws count as half a win so a drawish but stronger candidate can
        # still clear the threshold instead of being drowned in the denominator.
        ratio = (wins + 0.5 * draws) / total if total else 0.0
        improved = ratio >= self.cfg.training.arena_win_ratio
        if improved:
            self.best_net.load_state_dict(self.net.state_dict())
            self.best_iteration = current_iteration
            self.save_best()
        return {
            "arena_wins": wins,
            "arena_losses": losses,
            "arena_draws": draws,
            "arena_ratio": ratio,
            "arena_plies": plies,
            "improved": improved,
        }

    # ------------------------------------------------------------- checkpointing

    def save_checkpoint(self, path) -> None:
        """Full training state (net, optimizer, best net, replay buffer)."""
        _atomic_save(self._payload(), path)

    def save_best(self) -> None:
        """Slim best.pt for inference: the arena-best weights + iteration."""
        _atomic_save(
            {
                "iteration": self.best_iteration,
                "network": asdict(self.cfg.network),
                "net_state": {k: v.cpu() for k, v in self.best_net.state_dict().items()},
            },
            self.checkpoint_dir / "best.pt",
        )

    def _payload(self) -> dict:
        """Checkpoint payload with device-normalized (CPU) tensors, so a
        checkpoint written on mps/cuda loads on any device."""
        return {
            "iteration": self.iteration,
            "network": asdict(self.cfg.network),
            "net_state": {k: v.cpu() for k, v in self.net.state_dict().items()},
            "optimizer_state": _cpu_optim_state(self.optimizer.state_dict()),
            "best_net_state": {k: v.cpu() for k, v in self.best_net.state_dict().items()},
            "best_iteration": self.best_iteration,
            "buffer": self.buffer.state(),
        }

    def load_checkpoint(self, path) -> None:
        """Load a training checkpoint, hard-failing on corruption, a missing
        replay buffer, or a network-config mismatch: resume must never
        silently continue from a different architecture."""
        try:
            payload = torch.load(path, weights_only=False)
        except Exception as exc:
            raise SystemExit(
                f"ERROR: checkpoint {path} is unreadable/corrupt: {exc}\n"
                "  Delete it manually, or run with --restart to wipe the checkpoint dir."
            ) from exc
        self._apply_payload(payload)

    def _apply_payload(self, payload) -> None:
        saved_network = payload.get("network")
        if saved_network is not None and saved_network != asdict(self.cfg.network):
            raise SystemExit(
                f"ERROR: checkpoint was trained with network {saved_network},\n"
                f"  but the config has {asdict(self.cfg.network)}.\n"
                "  Run with --restart to wipe the checkpoint dir and start fresh."
            )
        self.iteration = payload["iteration"]
        self.net.load_state_dict(payload["net_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if "buffer" not in payload:
            raise SystemExit(
                "ERROR: checkpoint has no replay buffer; expected a latest.pt "
                "written by this version of the trainer."
            )
        self.buffer.load_state(payload["buffer"])
        self.best_net.load_state_dict(payload["best_net_state"])
        self.best_iteration = payload["best_iteration"]

    def train_iteration(self) -> dict:
        current = self.iteration + 1
        if self.server is not None:
            # Self-play evaluates the current net (slot 0).
            self.server.set_weights(0, self.net.state_dict())
        game_stats = self._selfplay(self.net)
        self._selfplay_diag(current, game_stats)
        loss_mean, policy_loss, value_loss = self._optimize()
        if self.server is not None:
            # Arena: candidate (slot 0) vs best (slot 1). Workers are dead
            # between phases, so no in-flight request can observe the swap.
            self.server.set_weights(0, self.net.state_dict())
            self.server.set_weights(1, self.best_net.state_dict())
        arena = self._maybe_update_best(current)
        self.iteration = current  # only a fully completed iteration counts
        self.save_checkpoint(self.checkpoint_dir / "latest.pt")
        return {
            "iteration": self.iteration,
            "buffer_size": len(self.buffer),
            "loss_mean": loss_mean,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            **arena,
        }

    def _selfplay_diag(self, current: int, game_stats: list) -> None:
        """Aggregate self-play diagnostics, print them, and append to JSONL."""
        agg = aggregate_games(game_stats)
        novel_frac = self._novelty(game_stats)
        buf = buffer_stats(*self.buffer.state()[:2])
        diag = {
            "agg": agg,
            "buffer": buf,
            "novel_frac": novel_frac,
        }
        for line in format_lines(current, agg, buf, novel_frac, None):
            tqdm.write(line)
        try:
            with open(self.checkpoint_dir / "diagnostics.jsonl", "a") as f:
                f.write(json.dumps({"iteration": current, **diag}) + "\n")
        except OSError as exc:
            tqdm.write(f"WARNING: could not write diagnostics.jsonl: {exc}")

    def _novelty(self, game_stats: list) -> float:
        """Fraction of this iteration's stored samples already in the buffer."""
        new_hashes = []
        for g in game_stats:
            if g:
                new_hashes.extend(g.get("sample_hashes") or [])
        if not new_hashes:
            return 0.0
        novel = sum(1 for h in new_hashes if h not in self._buf_hashes)
        self._buf_hashes.update(new_hashes)
        return novel / len(new_hashes)

    def train(self, target: int | None) -> list:
        """Run iterations until `target` (inclusive), or forever when None.

        Starts the central inference server first when the resolved device
        is an accelerator (mps/cuda); it survives across iterations and
        receives fresh weights at each phase boundary. Returns per-iteration
        stats dicts.
        """
        stats = []
        if self.device in ("mps", "cuda"):
            from .inference_server import InferenceServerHandle
            self.server = InferenceServerHandle(
                self.cfg.network, self.device, slots=2,
                num_threads=worker_num_threads(2),
            ).start(
                initial_states=[
                    {k: v.cpu() for k, v in self.net.state_dict().items()},
                    {k: v.cpu() for k, v in self.best_net.state_dict().items()},
                ]
            )
        try:
            remaining = None if target is None else target - self.iteration
            with _tqdm(total=remaining, desc="iterations", unit="iter") as bar:
                while target is None or self.iteration < target:
                    t0 = time.perf_counter()
                    stats.append(self.train_iteration())
                    s = stats[-1]
                    s["elapsed"] = time.perf_counter() - t0
                    bar.update(1)
                    tqdm.write(
                        f"iter {s['iteration']:3d}  loss {s['loss_mean']:.4f}"
                        f" (pol {s['policy_loss']:.3f} val {s['value_loss']:.3f})  "
                        f"buffer {s['buffer_size']}  "
                        f"arena {s['arena_wins']}W/{s['arena_losses']}L/{s['arena_draws']}D"
                        f" ({s['arena_ratio']:.2f}) "
                        f"{s['arena_plies'] // max(1, s['arena_wins'] + s['arena_losses'] + s['arena_draws'])}ply/g  "
                        f"{'BEST ' if s['improved'] else ''}"
                        f"[{s['elapsed']:.1f}s]"
                    )
        finally:
            if self.server is not None:
                self.server.shutdown()
                self.server = None
        return stats


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the smart-four AlphaZero model")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--iterations", type=int, default=None,
                        help="train until this iteration (inclusive); omit to train forever")
    parser.add_argument("--restart", action="store_true",
                        help="delete all checkpoints (after confirmation) and train fresh")
    parser.add_argument("--yes", action="store_true",
                        help="with --restart: skip the confirmation prompt")
    parser.add_argument("--device", default=None,
                        choices=VALID_DEVICES,
                        help="device override: auto (default, from config), cpu, mps, cuda")
    args = parser.parse_args(argv)
    if args.iterations is not None and args.iterations < 0:
        parser.error("--iterations must be >= 0")

    cfg = load_config(args.config)
    # Clamp worker counts to the machine's CPU count: spawning more processes
    # than cores only adds context-switch overhead.
    cpus = os.cpu_count() or 1
    if cfg.training.workers > cpus:
        cfg = replace(
            cfg,
            training=replace(cfg.training, workers=min(cfg.training.workers, cpus)),
        )

    device = resolve_device(args.device if args.device else cfg.device.name)

    if args.restart:
        _confirm_and_wipe(Path(cfg.training.checkpoint_dir), args.yes)

    try:
        trainer = Trainer(cfg, device=device)
    except KeyboardInterrupt:
        # The first optimizer/network init lazily imports torch internals,
        # which can take seconds; nothing exists to save yet.
        print("\nInterrupted during setup — nothing to save.", file=sys.stderr)
        raise SystemExit(130)

    resume = _find_resume_checkpoint(trainer.checkpoint_dir)
    if resume is None:
        print("No checkpoint found; starting fresh")
    else:
        trainer.load_checkpoint(resume)
        print(f"Resumed from {resume.name} (iteration {trainer.iteration})")

    target = args.iterations
    if target is not None and trainer.iteration >= target:
        print(f"Already at iteration {trainer.iteration} (target {target}); nothing to train.")
        raise SystemExit(0)

    n_params = sum(p.numel() for p in trainer.net.parameters())
    print("Smart-four AlphaZero training")
    print(f"  config      {args.config}")
    print(f"  device      {trainer.device}")
    print(f"  network     {cfg.network.blocks} blocks x {cfg.network.base_channels} ch"
          f" ({n_params:,} params)")
    print(f"  cpus        {cpus} (worker counts capped at this)")
    print(f"  inference   central server on {device}" if device in ("mps", "cuda")
          else f"  inference   per-worker CPU nets")
    print(f"  self-play   {cfg.training.selfplay_games} games"
          f" x {cfg.training.workers} worker(s),"
          f" {cfg.mcts.simulations} sims/move")
    print(f"  optimize    {cfg.training.train_epochs} epochs, batch {cfg.training.batch_size}")
    print(f"  arena       {cfg.training.eval_games} games vs best"
          f" x {cfg.training.workers} worker(s),"
          f" {cfg.mcts.simulations} sims/move")
    print(f"  checkpoint  {trainer.checkpoint_dir}/")
    if resume is None and cfg.training.pretrain_games > 0:
        print(f"  pretrain    {cfg.training.pretrain_games} random games"
              f" x {cfg.training.pretrain_epochs} epochs (value head only)")
    if target is None:
        print(f"  iterations  training indefinitely (start iteration "
              f"{trainer.iteration + 1}, until Ctrl-C/SIGTERM)")
    else:
        print(f"  iterations  start iteration {trainer.iteration + 1}, "
              f"{target - trainer.iteration} left (target {target})")

    if resume is None and cfg.training.pretrain_games > 0:
        tqdm.write("Pretraining value head on random-rollout outcomes...")
        t0 = time.perf_counter()
        with _tqdm(desc="pretrain value", unit="batch", leave=False) as bar:
            # Pretrain runs on a CPU copy (one-time, cheap); weights copy back.
            cpu_net = ResNet(cfg.network)
            cpu_net.load_state_dict({k: v.cpu() for k, v in trainer.net.state_dict().items()})
            cpu_net.train()
            mse = pretrain_value(
                cpu_net,
                cfg.training.pretrain_games,
                cfg.training.pretrain_epochs,
                cfg.training.batch_size,
                cfg.training.pretrain_lr,
                cfg.training.weight_decay,
                cfg.training.seed + 12345,
                progress=bar.update,
            )
            trainer.net.load_state_dict({k: v.to(trainer.device) for k, v in cpu_net.state_dict().items()})
            trainer.net.eval()
        trainer.best_net.load_state_dict(trainer.net.state_dict())
        print(f"  pretrain    done in {time.perf_counter() - t0:.0f}s,"
              f" final value MSE {mse:.4f}")

    def _on_sigterm(signum, frame):
        # Exit cleanly so atexit reaps the daemonic workers, but discard the
        # in-flight iteration: latest.pt already holds the last completed one.
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _on_sigterm)

    t_start = time.perf_counter()
    try:
        stats = trainer.train(target)
    except KeyboardInterrupt:
        # Ctrl-C discards the in-flight iteration; latest.pt is untouched.
        raise SystemExit(130)
    total = time.perf_counter() - t_start

    best_iter = max((s["iteration"] for s in stats if s["improved"]), default=None)
    n = max(1, len(stats))
    print(f"\nTraining complete: {len(stats)} iteration(s) in {total:.1f}s"
          f" ({total / n:.1f}s/iter)")
    if best_iter:
        print(f"Best model updated at iteration {best_iter} ->"
              f" {trainer.checkpoint_dir}/best.pt")
    else:
        print(f"Best model unchanged -> {trainer.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
