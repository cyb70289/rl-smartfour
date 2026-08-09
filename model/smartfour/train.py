"""AlphaZero training loop: self-play -> replay buffer -> optimize -> arena.

Checkpoints (checkpoint_dir/): latest.pt every iteration, best.pt plus a
best_iter_XXXX.pt copy whenever the candidate beats the current best in the
arena. Checkpoints carry net + optimizer + buffer state for resume.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from .arena import play_arena
from .config import Config, load_config
from .encode import apply_d4, apply_d4_policy, d4_perms
from .network import ResNet, loss_fn
from .selfplay import play_game


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
        for _ in range(games):
            samples, _winner = play_game(
                net, self.cfg.mcts, self.cfg.mcts.temperature_threshold
            )
            self.buffer.push(samples)

    def _optimize(self) -> float:
        losses = []
        n_batches = max(1, len(self.buffer) // self.cfg.training.batch_size)
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
        return sum(losses) / len(losses) if losses else float("nan")

    def _arena(self, net_a, net_b, games: int, eval_simulations: int):
        return play_arena(net_a, net_b, self.cfg.mcts, games, eval_simulations)

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
        for _ in range(iterations):
            stats.append(self.train_iteration())
        return stats


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the smart-four AlphaZero model")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="resume from latest.pt")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    trainer = Trainer(cfg)
    if args.resume:
        latest = trainer.checkpoint_dir / "latest.pt"
        if latest.exists():
            trainer.load_checkpoint(latest)
            print(f"Resumed from iteration {trainer.iteration}")
        else:
            print("No checkpoint to resume; starting fresh")
    for stats in trainer.train(args.iterations):
        print(
            f"iter {stats['iteration']:3d}  loss {stats['loss_mean']:.4f}  "
            f"buffer {stats['buffer_size']}  "
            f"arena {stats['arena_wins']}W/{stats['arena_losses']}L/{stats['arena_draws']}D"
            f" ({stats['arena_ratio']:.2f})  "
            f"{'BEST' if stats['improved'] else ''}"
        )


if __name__ == "__main__":
    main()
