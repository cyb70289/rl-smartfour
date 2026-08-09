"""ResNet with policy and value heads (AlphaZero-style).

Input: (B, input_channels, 5, 5). Output: (B, 125) logits, (B, 1) value in
(-1, 1). Everything is from the current player's perspective (see encode).
"""

import torch
from torch import nn

from .config import NetworkConfig

POLICY_SIZE = 125  # 5 x 5 x 5 planes


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(x + out)


class ResNet(nn.Module):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.input_channels, cfg.base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.base_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([_ResBlock(cfg.base_channels) for _ in range(cfg.blocks)])

        # Policy head: 1x1 convs to 5 height planes of 5x5 = 125 logits.
        # Logit index y*25 + x*5 + z matches encode.xyz_to_action.
        self.policy_head = nn.Sequential(
            nn.Conv2d(cfg.base_channels, cfg.policy_channels, 1, bias=False),
            nn.BatchNorm2d(cfg.policy_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.policy_channels, 5, 1),
        )

        # Value head: 1x1 conv, flatten, MLP to a scalar.
        self.value_head = nn.Sequential(
            nn.Conv2d(cfg.base_channels, cfg.value_channels, 1, bias=False),
            nn.BatchNorm2d(cfg.value_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(cfg.value_channels * 25, cfg.value_fc),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.value_fc, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        logits = self.policy_head(h).flatten(1)
        value = self.value_head(h)
        return logits, value


def loss_fn(logits: torch.Tensor, value: torch.Tensor, pi: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """AlphaZero loss: policy cross-entropy + value MSE.

    logits (B, 125) raw logits; pi (B, 125) target distribution (zeros on
    illegal actions); value/z (B, 1). L2 regularization is applied through the
    optimizer's weight decay.
    """
    policy_loss = -torch.sum(pi * torch.log_softmax(logits, dim=1), dim=1)
    value_loss = (value - z) ** 2
    return (policy_loss + value_loss.squeeze(1)).mean()
