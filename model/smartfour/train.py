"""AlphaZero training loop: self-play -> replay buffer -> optimize -> arena.

Checkpoints (checkpoint_dir/): latest.pt every iteration, best.pt plus a
best_iter_XXXX.pt copy whenever the candidate beats the current best in the
arena. Checkpoints carry net + optimizer + buffer state for resume.
"""

import argparse
import multiprocessing
import os
import queue
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm

from .arena import play_arena
from .config import Config, load_config
from .encode import apply_d4, apply_d4_policy, d4_perms
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
        self.iteration = 0
        self.checkpoint_dir = Path(cfg.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- internals

    def _selfplay(self, net, games: int | None = None) -> None:
        games = games if games is not None else self.cfg.training.selfplay_games
        workers = self.cfg.training.selfplay_workers
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
        worker errors or dies before delivering its games.
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
                )
                p.start()
                procs.append(p)
            self._collect_selfplay(games, procs, out_q, bar)
        finally:
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join(timeout=30)

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
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join(timeout=30)
            raise
        for p in procs:
            p.join(timeout=60)
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
        bad = [p.exitcode for p in procs if p.exitcode != 0]
        if bad:
            raise RuntimeError(f"self-play worker(s) exited with code {bad}")

    def _optimize(self) -> float:
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

    def _arena(self, net_a, net_b, games: int, eval_simulations: int):
        with _tqdm(total=games, desc="arena", unit="game", leave=False) as bar:
            return play_arena(
                net_a, net_b, self.cfg.mcts, games, eval_simulations, progress=bar.update
            )

    def _maybe_update_best(self) -> dict:
        games = self.cfg.training.eval_games
        wins, losses, draws = self._arena(
            self.net, self.best_net, games, self.cfg.training.eval_simulations
        )
        total = wins + losses + draws
        ratio = wins / total if total else 0.0
        improved = ratio >= self.cfg.training.arena_win_ratio
        if improved:
            self.save_checkpoint(self.checkpoint_dir / "best.pt", is_best=True)
        return {
            "arena_wins": wins,
            "arena_losses": losses,
            "arena_draws": draws,
            "arena_ratio": ratio,
            "improved": improved,
        }

    # ------------------------------------------------------------- checkpointing

    def save_checkpoint(self, path, is_best: bool = False) -> None:
        payload = {
            "iteration": self.iteration,
            "network": asdict(self.cfg.network),
            "net_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "buffer": self.buffer.state(),
        }
        torch.save(payload, path)
        if is_best:
            torch.save(payload, self.checkpoint_dir / f"best_iter_{self.iteration:04d}.pt")

    def load_checkpoint(self, path) -> None:
        payload = torch.load(path, weights_only=False)
        self.iteration = payload["iteration"]
        self.net.load_state_dict(payload["net_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.buffer.load_state(payload["buffer"])

    # ------------------------------------------------------------- driving

    def train_iteration(self) -> dict:
        self.iteration += 1
        self._selfplay(self.net)
        loss_mean = self._optimize()
        arena = self._maybe_update_best()
        self.save_checkpoint(self.checkpoint_dir / "latest.pt")
        return {
            "iteration": self.iteration,
            "buffer_size": len(self.buffer),
            "loss_mean": loss_mean,
            **arena,
        }

    def train(self, iterations: int) -> list:
        stats = []
        with _tqdm(total=iterations, desc="iterations", unit="iter") as bar:
            for _ in range(iterations):
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
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="resume from latest.pt")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    try:
        trainer = Trainer(cfg)
    except KeyboardInterrupt:
        # The first optimizer/network init lazily imports torch internals,
        # which can take seconds; nothing exists to save yet.
        print("\nInterrupted during setup — nothing to save.", file=sys.stderr)
        raise SystemExit(130)
    if args.resume:
        latest = trainer.checkpoint_dir / "latest.pt"
        if latest.exists():
            trainer.load_checkpoint(latest)
            print(f"Resumed from iteration {trainer.iteration}")
        else:
            print("No checkpoint to resume; starting fresh")

    n_params = sum(p.numel() for p in trainer.net.parameters())
    print("Smart-four AlphaZero training")
    print(f"  config      {args.config}")
    print(f"  device      {trainer.device}")
    print(f"  network     {cfg.network.blocks} blocks x {cfg.network.base_channels} ch"
          f" ({n_params:,} params)")
    print(f"  self-play   {cfg.training.selfplay_games} games"
          f" x {cfg.training.selfplay_workers} worker(s),"
          f" {cfg.mcts.simulations} sims/move")
    print(f"  optimize    {cfg.training.train_epochs} epochs, batch {cfg.training.batch_size}")
    print(f"  arena       {cfg.training.eval_games} games vs best,"
          f" {cfg.training.eval_simulations} sims/move")
    print(f"  checkpoint  {trainer.checkpoint_dir}/")

    t_start = time.perf_counter()
    try:
        stats = trainer.train(args.iterations)
    except KeyboardInterrupt:
        tqdm.write("\nTraining interrupted, saving current state...")
        trainer.save_checkpoint(trainer.checkpoint_dir / "latest.pt")
        tqdm.write(f"Saved {trainer.checkpoint_dir}/latest.pt, resume with --resume")
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
