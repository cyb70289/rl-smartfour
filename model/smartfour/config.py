"""Configuration dataclasses, loaded from a TOML file (see config.toml)."""

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class NetworkConfig:
    input_channels: int = 15
    blocks: int = 5
    base_channels: int = 64
    policy_channels: int = 32
    value_channels: int = 16
    value_fc: int = 64


@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 200
    c_puct: float = 2.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_threshold: int = 12  # plies with tau=1; later plies tau=0
    batch_eval_size: int = 32


@dataclass(frozen=True)
class TrainingConfig:
    selfplay_games: int = 100
    workers: int = 8  # parallel self-play and arena processes; clamped to
                      # the CPU count at startup (1 = sequential)
    train_epochs: int = 5
    batch_size: int = 128
    replay_capacity_games: int = 2_000  # replay buffer window, in whole games
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    symmetry_augment: bool = True
    eval_games: int = 40
    arena_win_ratio: float = 0.55
    checkpoint_dir: str = "checkpoints"
    seed: int = 0
    # Value-head bootstrap on random-rollout outcomes, run once for fresh
    # starts: a random value head leaves PUCT with no q-signal, so the search
    # fills breadth-first and freezes at ~3 plies at any sim budget.
    pretrain_games: int = 0       # 0 = disabled
    pretrain_epochs: int = 8
    pretrain_lr: float = 0.001

@dataclass(frozen=True)
class DeviceConfig:
    name: str = "auto"  # auto | cpu | mps | cuda (see smartfour.device)


@dataclass(frozen=True)
class Config:
    network: NetworkConfig
    mcts: MCTSConfig
    training: TrainingConfig
    device: DeviceConfig = DeviceConfig()



def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    training = TrainingConfig(**raw.get("training", {}))
    if training.workers < 1:
        raise ValueError(
            f"training.workers must be >= 1, got {training.workers}"
        )
    return Config(
        network=NetworkConfig(**raw.get("network", {})),
        mcts=MCTSConfig(**raw.get("mcts", {})),
        training=training,
        device=DeviceConfig(**raw.get("device", {})),
    )
