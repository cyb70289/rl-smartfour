"""AlphaZero training loop: self-play -> replay buffer -> optimize -> arena.

Checkpoints (checkpoint_dir/):
  latest.pt    full state (net, optimizer, best_net, replay buffer) — the
               exact resume anchor; written after every completed iteration
               and on interrupt/crash.
  iter_NNNN.pt net + optimizer + best_net only (no buffer) — one light
               snapshot per completed iteration, for history and manual
               pruning.
  best.pt      slim inference snapshot of the arena-best net (weights +
               iteration the best was set). Never used for resume.
All writes are atomic (temp file + os.replace), so an interrupt or crash
mid-save never corrupts an existing checkpoint.

Resume is the default and needs no flag: latest.pt -> newest iter_NNNN.pt ->
fresh start. A corrupt/unreadable checkpoint or a network-config mismatch is
a hard error (never a silent fallback); --restart wipes the checkpoint dir
(after confirmation) to start over. --iterations N is a target: train until
iteration N, exit immediately when already there; without it, train forever
until SIGINT/SIGTERM, which save latest.pt before exiting.
"""

import argparse
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
from .config import Config, load_config
from .encode import apply_d4, apply_d4_policy, d4_perms
from .game import DRAW, WHITE
from .network import ResNet, loss_fn
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
    """Resume anchor: latest.pt, else the newest iter_NNNN.pt, else None."""
    checkpoint_dir = Path(checkpoint_dir)
    latest = checkpoint_dir / "latest.pt"
    if latest.exists():
        return latest
    iters = sorted(checkpoint_dir.glob("iter_*.pt"))
    return iters[-1] if iters else None


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

    # ------------------------------------------------------------- internals

    def _selfplay(self, net, games: int | None = None) -> None:
        games = games if games is not None else self.cfg.training.selfplay_games
        workers = self.cfg.training.workers
        if workers <= 1:
            with _tqdm(total=games, desc="self-play", unit="game", leave=False) as bar:
                for _ in range(games):
                    samples, _winner = play_game(
                        net, self.cfg.mcts, self.cfg.mcts.temperature_threshold
                    )
                    self.buffer.push(samples)
                    bar.set_postfix(buffer=len(self.buffer))
                    bar.update(1)
            return
        with _tqdm(total=games, desc="self-play", unit="game", leave=False) as bar:
            self._selfplay_parallel(net, games, workers, bar)

    def _selfplay_parallel(self, net, games: int, workers: int, bar) -> None:
        """Spawn one process per worker; each plays its share of games with a
        fresh net copy and ships samples over a queue. Fails fast if any
        worker errors or dies before delivering its games. Workers are
        daemonic (a dying parent cannot orphan them) and ignore SIGINT (the
        parent alone decides when to stop).
        """
        ctx = multiprocessing.get_context("spawn")
        out_q = ctx.Queue()
        net_state = net.state_dict()
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
                    ),
                    daemon=True,
                )
                p.start()
                procs.append(p)
            self._collect_selfplay(games, procs, out_q, bar)
        finally:
            _terminate_workers(procs)

    def _collect_selfplay(self, games: int, procs, out_q, bar) -> None:
        """Consume worker results until `games` games are pushed to the buffer.

        Raises RuntimeError when a worker reports failure or dies early, so a
        broken worker can never hang training or silently shrink the batch;
        surviving workers are terminated before the error propagates.
        """
        received = 0
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
                self.buffer.push(samples_from_ipc(msg))
                received += 1
                bar.set_postfix(buffer=len(self.buffer))
                bar.update(1)
        except BaseException:
            _terminate_workers(procs)
            raise
        self._finish_workers(procs, "self-play")

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

    def _optimize(self) -> float:
        if len(self.buffer) < self.cfg.training.batch_size:
            tqdm.write(
                f"WARNING: replay buffer too small ({len(self.buffer)} < "
                f"batch_size {self.cfg.training.batch_size}); skipping optimize"
            )
            return float("nan")
        self.net.train()  # MCTS leaves the net in eval mode; BN must update
        losses = []
        n_batches = max(1, len(self.buffer) // self.cfg.training.batch_size)
        total = n_batches * self.cfg.training.train_epochs
        with _tqdm(total=total, desc="optimize", unit="batch", leave=False) as bar:
            for _ in range(self.cfg.training.train_epochs):
                for _ in range(n_batches):
                    s, pi, z = self.buffer.sample(
                        self.cfg.training.batch_size,
                        augment=self.cfg.training.symmetry_augment,
                    )
                    logits, value = self.net(s)
                    loss = loss_fn(logits, value, pi, z)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())
                    recent = losses[-50:]
                    bar.set_postfix(loss=f"{sum(recent) / len(recent):.4f}")
                    bar.update(1)
        return sum(losses) / len(losses) if losses else float("nan")

    def _arena(self, net_a, net_b, games: int):
        workers = self.cfg.training.workers
        if workers <= 1:
            with _tqdm(total=games, desc="arena", unit="game", leave=False) as bar:
                return play_arena(net_a, net_b, self.cfg.mcts, games, progress=bar.update)
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
        net_a_state = net_a.state_dict()
        net_b_state = net_b.state_dict()
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
                if msg == DRAW:
                    draws += 1
                elif msg == WHITE:
                    a_wins += 1
                else:
                    b_wins += 1
                received += 1
                bar.update(1)
        except BaseException:
            _terminate_workers(procs)
            raise
        self._finish_workers(procs, "arena")
        return a_wins, b_wins, draws

    def _maybe_update_best(self, current_iteration: int) -> dict:
        games = self.cfg.training.eval_games
        wins, losses, draws = self._arena(self.net, self.best_net, games)
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
            "improved": improved,
        }

    # ------------------------------------------------------------- checkpointing

    def save_checkpoint(self, path, include_buffer: bool = False) -> None:
        _atomic_save(self._payload(include_buffer), path)

    def save_best(self) -> None:
        """Slim best.pt for inference: the arena-best weights + iteration."""
        _atomic_save(
            {
                "iteration": self.best_iteration,
                "network": asdict(self.cfg.network),
                "net_state": self.best_net.state_dict(),
            },
            self.checkpoint_dir / "best.pt",
        )

    def _payload(self, include_buffer: bool) -> dict:
        payload = {
            "iteration": self.iteration,
            "network": asdict(self.cfg.network),
            "net_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_net_state": self.best_net.state_dict(),
            "best_iteration": self.best_iteration,
        }
        if include_buffer:
            payload["buffer"] = self.buffer.state()
        return payload

    def load_checkpoint(self, path) -> None:
        """Load a training checkpoint, hard-failing on corruption or a
        network-config mismatch: resume must never silently continue from a
        different architecture."""
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
        if "buffer" in payload:
            self.buffer.load_state(payload["buffer"])
        else:
            tqdm.write(
                f"WARNING: checkpoint has no replay buffer (likely an iter_*.pt "
                f"fallback after latest.pt was removed); the buffer is empty, "
                f"so the next optimize is skipped until self-play refills it."
            )
        if "best_net_state" in payload:
            self.best_net.load_state_dict(payload["best_net_state"])
            self.best_iteration = payload.get("best_iteration", self.iteration)

    # ------------------------------------------------------------- driving

    def train_iteration(self) -> dict:
        current = self.iteration + 1
        self._selfplay(self.net)
        loss_mean = self._optimize()
        arena = self._maybe_update_best(current)
        self.iteration = current  # only a fully completed iteration counts
        self.save_checkpoint(self.checkpoint_dir / f"iter_{current:04d}.pt")
        self.save_checkpoint(self.checkpoint_dir / "latest.pt", include_buffer=True)
        return {
            "iteration": self.iteration,
            "buffer_size": len(self.buffer),
            "loss_mean": loss_mean,
            **arena,
        }

    def train(self, target: int | None) -> list:
        """Run iterations until `target` (inclusive), or forever when None.

        Returns per-iteration stats dicts.
        """
        stats = []
        remaining = None if target is None else target - self.iteration
        with _tqdm(total=remaining, desc="iterations", unit="iter") as bar:
            while target is None or self.iteration < target:
                t0 = time.perf_counter()
                stats.append(self.train_iteration())
                s = stats[-1]
                s["elapsed"] = time.perf_counter() - t0
                bar.update(1)
                tqdm.write(
                    f"iter {s['iteration']:3d}  loss {s['loss_mean']:.4f}  "
                    f"buffer {s['buffer_size']}  "
                    f"arena {s['arena_wins']}W/{s['arena_losses']}L/{s['arena_draws']}D"
                    f" ({s['arena_ratio']:.2f})  "
                    f"{'BEST ' if s['improved'] else ''}"
                    f"[{s['elapsed']:.1f}s]"
                )
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

    if args.restart:
        _confirm_and_wipe(Path(cfg.training.checkpoint_dir), args.yes)

    try:
        trainer = Trainer(cfg)
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
    print(f"  self-play   {cfg.training.selfplay_games} games"
          f" x {cfg.training.workers} worker(s),"
          f" {cfg.mcts.simulations} sims/move")
    print(f"  optimize    {cfg.training.train_epochs} epochs, batch {cfg.training.batch_size}")
    print(f"  arena       {cfg.training.eval_games} games vs best"
          f" x {cfg.training.workers} worker(s),"
          f" {cfg.mcts.simulations} sims/move")
    print(f"  checkpoint  {trainer.checkpoint_dir}/")
    if target is None:
        print(f"  iterations  training indefinitely (start iteration "
              f"{trainer.iteration + 1}, until Ctrl-C/SIGTERM)")
    else:
        print(f"  iterations  start iteration {trainer.iteration + 1}, "
              f"{target - trainer.iteration} left (target {target})")

    sigterm_seen = {"flag": False}

    def _on_sigterm(signum, frame):
        sigterm_seen["flag"] = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_sigterm)

    t_start = time.perf_counter()
    try:
        stats = trainer.train(target)
    except KeyboardInterrupt:
        code = 143 if sigterm_seen["flag"] else 130
        tqdm.write("\nTraining interrupted, saving current state...")
        # Block further signals so the save cannot be interrupted; the
        # process is exiting, so the mask is never lifted.
        signal.pthread_sigmask(signal.SIG_BLOCK, (signal.SIGINT, signal.SIGTERM))
        try:
            trainer.save_checkpoint(trainer.checkpoint_dir / "latest.pt", include_buffer=True)
        finally:
            pass
        tqdm.write(f"Saved {trainer.checkpoint_dir}/latest.pt (iteration {trainer.iteration})")
        raise SystemExit(code)
    except Exception:
        # A crash must not lose the training state either.
        tqdm.write("\nTraining failed, saving current state...")
        trainer.save_checkpoint(trainer.checkpoint_dir / "latest.pt", include_buffer=True)
        tqdm.write(f"Saved {trainer.checkpoint_dir}/latest.pt (iteration {trainer.iteration})")
        raise
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
