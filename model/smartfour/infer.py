"""Inference: load a checkpoint and choose moves for a state.

The agent accepts either a smartfour GameState or a JSON state in the UI
interchange format (see game.state_to_json / state_from_json):

    {
      "grid": [[[0|1|null, ...], ...], ...],   # grid[x][z][y]
      "pieces_left": {"white": int, "black": int},
      "current": "white" | "black",
      "winner": null | "white" | "black" | "draw"
    }

CLI:  python -m smartfour.infer --checkpoint checkpoints/best.pt --sims 200 --state state.json
Prints the chosen move as {"x": .., "z": ..} (or {"move": null} when terminal).
"""

import argparse
import json
import sys

import torch

from .config import MCTSConfig, NetworkConfig
from .encode import action_mask, action_to_xyz, encode
from .game import is_terminal, state_from_json
from .mcts import MCTS
from .network import ResNet

NEG_INF = float("-inf")


class SmartFourAgent:
    def __init__(self, checkpoint_path: str, device="cpu"):
        payload = torch.load(checkpoint_path, weights_only=False)
        net_cfg = NetworkConfig(**payload.get("network", NetworkConfig().__dict__))
        self.net = ResNet(net_cfg).to(device)
        self.net.load_state_dict(payload["net_state"])
        self.net.eval()
        self.iteration = payload.get("iteration", 0)

    def _encode(self, state):
        return encode(state)

    def _mask(self, state):
        return action_mask(state)

    def choose_move(self, state, simulations: int = 200):
        """Return (x, z) for the best move, or None if the game is over.

        simulations=0 is policy-only (no search): argmax of the policy head.
        """
        if isinstance(state, dict):
            state = state_from_json(state)
        if is_terminal(state):
            return None
        if simulations == 0:
            with torch.no_grad():
                logits, _ = self.net(self._encode(state).unsqueeze(0))
            mask = self._mask(state)
            masked = torch.where(mask.bool(), logits[0], torch.full_like(logits[0], NEG_INF))
            x, z, _y = action_to_xyz(int(torch.argmax(masked)))
            return (x, z)
        mcts = MCTS(self.net, MCTSConfig(simulations=simulations))
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
    args = parser.parse_args(argv)

    if args.state:
        with open(args.state) as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    agent = SmartFourAgent(args.checkpoint)
    move = agent.choose_move(data, simulations=args.sims)
    if move is None:
        print(json.dumps({"move": None}))
    else:
        print(json.dumps({"x": move[0], "z": move[1]}))


if __name__ == "__main__":
    main()
