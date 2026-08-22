"""Inference: load a checkpoint and choose moves for a state.

The agent accepts either a smartfour GameState or a JSON state in the UI
interchange format (see game.state_to_json / state_from_json):

    {
      "grid": [[[0|1|null, ...], ...], ...],   # grid[x][z][y]
      "pieces_left": {"white": int, "black": int},
      "current": "white" | "black",
      "winner": null | "white" | "black" | "draw"
    }

CLI:  python -m smartfour.infer --checkpoint checkpoints/best{n}.pt --sims 200 --state state.json
Prints the chosen move as {"x": .., "z": ..} (or {"move": null} when terminal).
"""

import argparse
import json
import os
import sys

import torch

from .config import MCTSConfig, NetworkConfig
from .device import resolve_device
from .encode import action_mask, action_to_xyz, encode
from .game import is_terminal, state_from_json
from .mcts import MCTS
from .network import ResNet

NEG_INF = float("-inf")


class SmartFourAgent:
    def __init__(self, checkpoint_path: str, device=None):
        device = device or os.environ.get("SMARTFOUR_DEVICE", "auto")
        self.device = resolve_device(device)
        payload = torch.load(checkpoint_path, weights_only=True)
        net_cfg = NetworkConfig(**payload.get("network", NetworkConfig().__dict__))
        self.net = ResNet(net_cfg).to(self.device)
        self.net.load_state_dict(payload["net_state"])
        self.net.eval()
        self.iteration = payload.get("iteration", 0)

    def _encode(self, state):
        return encode(state)

    def _mask(self, state):
        return action_mask(state)

    def choose_move(self, state, simulations: int = 200, batch_eval_size: int = 128):
        """Return (x, z) for the best move, or None if the game is over.

        simulations=0 is policy-only (no search): argmax of the policy head.
        simulations>0 runs the virtual-loss batched search (fills
        device-sized leaf batches inside one search) with
        `batch_eval_size` as the per-pass target.
        """
        if isinstance(state, dict):
            state = state_from_json(state)
        if is_terminal(state):
            return None
        if simulations == 0:
            with torch.no_grad():
                x = self._encode(state).unsqueeze(0).to(self.device)
                logits, _ = self.net(x)
                logits = logits[0].cpu()
            mask = self._mask(state)
            masked = torch.where(mask.bool(), logits, torch.full_like(logits, NEG_INF))
            x, z, _y = action_to_xyz(int(torch.argmax(masked)))
            return (x, z)
        mcts = MCTS(
            self.net,
            MCTSConfig(simulations=simulations, batch_eval_size=batch_eval_size),
            device=self.device,
        )
        _pi, chosen, _root = mcts.root_policy(state, root_noise=False, temperature=0.0)
        if chosen is None:
            return None
        x, z, _y = action_to_xyz(chosen)
        return (x, z)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Choose a move with a trained model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations (0 = policy only)")
    parser.add_argument("--state", help="JSON state file; read stdin when omitted")
    parser.add_argument("--device", default=None,
                        choices=("auto", "cpu", "mps", "cuda"),
                        help="inference device (default: $SMARTFOUR_DEVICE or auto)")
    parser.add_argument("--batch-eval-size", type=int, default=128,
                        help="virtual-loss batch target per pass (search mode)")
    args = parser.parse_args(argv)

    if args.state:
        with open(args.state) as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    agent = SmartFourAgent(args.checkpoint, device=args.device)
    move = agent.choose_move(
        data, simulations=args.sims, batch_eval_size=args.batch_eval_size
    )
    if move is None:
        print(json.dumps({"move": None}))
    else:
        print(json.dumps({"x": move[0], "z": move[1]}))


if __name__ == "__main__":
    main()
