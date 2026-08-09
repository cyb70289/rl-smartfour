"""Configuration dataclasses, loaded from a TOML file (see config.toml)."""

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class NetworkConfig:
    input_channels: int = 16
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
    train_epochs: int = 5
    batch_size: int = 128
    replay_capacity: int = 100_000
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    symmetry_augment: bool = True
    eval_games: int = 40
    eval_simulations: int = 100
    arena_win_ratio: float = 0.55
    checkpoint_dir: str = "checkpoints"
    seed: int = 0


@dataclass(frozen=True)
class Config:
    network: NetworkConfig
    mcts: MCTSConfig
    training: TrainingConfig


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Config(
        network=NetworkConfig(**raw.get("network", {})),
        mcts=MCTSConfig(**raw.get("mcts", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )
