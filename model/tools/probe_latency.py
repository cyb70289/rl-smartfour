"""Latency table: full game played in turn through the UI search path.

Machine plays WHITE every other ply (the human side is a fast random move),
through the same MCTS call the worker's choose_move makes. Reports mean ms
per machine move for:
  old  = sequential searcher, CPU (the pre-change UI path)
  newC = virtual-loss searcher, CPU
  newM = virtual-loss searcher, MPS
"""
import random
import sys
import time

sys.path.insert(0, ".")

import torch

from smartfour.config import MCTSConfig, NetworkConfig
from smartfour.encode import action_to_xyz
from smartfour.game import apply_move, initial_state, is_terminal, legal_moves
from smartfour.mcts import MCTS
from smartfour.network import ResNet

CKPT = "checkpoints/best.pt"


def load_net(device):
    payload = torch.load(CKPT, weights_only=True)
    net = ResNet(NetworkConfig(**payload["network"])).to(device)
    net.load_state_dict(payload["net_state"])
    net.eval()
    return net


def machine_move(net, state, sims, batched, device):
    cfg = MCTSConfig(
        simulations=sims,
        batch_eval_size=128 if batched else 32,
    )
    m = MCTS(net, cfg, device=device, batched=batched)
    _pi, chosen, _r = m.root_policy(state, root_noise=False, temperature=0.0)
    x, z, _y = action_to_xyz(chosen)
    return x, z


def play_game(net, sims, batched, device, rng):
    state = initial_state()
    machine_ms = 0.0
    machine_moves = 0
    while not is_terminal(state):
        if state.current == 0:  # machine is white
            t0 = time.perf_counter()
            x, z = machine_move(net, state, sims, batched, device)
            machine_ms += time.perf_counter() - t0
            machine_moves += 1
        else:
            x, z = rng.choice(legal_moves(state))
        state = apply_move(state, x, z)
    return machine_ms * 1000 / max(machine_moves, 1), machine_moves


if __name__ == "__main__":
    rng = random.Random(7)
    nets = {"cpu": load_net("cpu")}
    if torch.backends.mps.is_available():
        nets["mps"] = load_net("mps")
        _ = machine_move(nets["mps"], initial_state(), 8, True, "mps")  # warm
    print(f"{'effort':>7} {'old-CPU':>10} {'new-CPU':>10} {'new-MPS':>10}")
    for sims in (100, 400, 800, 2000):
        ms_old, _ = play_game(nets["cpu"], sims, False, "cpu", rng)
        ms_new, _ = play_game(nets["cpu"], sims, True, "cpu", rng)
        if "mps" in nets:
            ms_mps, _ = play_game(nets["mps"], sims, True, "mps", rng)
            mps = f"{ms_mps:>9.0f}m"
        else:
            mps = f"{'-':>10}"
        print(f"{sims:>7} {ms_old:>9.0f}m {ms_new:>9.0f}m {mps}")
