"""Arena parity: virtual-loss batched searcher vs sequential searcher.

Same net (checkpoints/best.pt), same sims (400), alternating colors.
Batched runs with batch_eval_size=128 (the UI default). Pass criterion:
batched win ratio >= 0.45 in net_a's frame (counted for the batched side).
"""
import multiprocessing as mp
import sys

sys.path.insert(0, ".")

import torch

from smartfour.arena import _result_in_a_frame
from smartfour.config import MCTSConfig
from smartfour.encode import action_to_xyz
from smartfour.game import BLACK, DRAW, WHITE, apply_move, initial_state, is_terminal
from smartfour.mcts import MCTS
from smartfour.network import ResNet

SIMS = 400
BES = 128
GAMES = 100
NET = "checkpoints/best.pt"


def load_net():
    payload = torch.load(NET, weights_only=True)
    from smartfour.config import NetworkConfig
    net = ResNet(NetworkConfig(**payload["network"]))
    net.load_state_dict(payload["net_state"])
    net.eval()
    return net


def _play_two(net, mcts_cfg, batched):
    white = MCTS(net, mcts_cfg, batched=batched)
    black = MCTS(net, mcts_cfg, batched=batched)
    state = initial_state()
    plies = 0
    while not is_terminal(state):
        m = white if state.current == WHITE else black
        _pi, chosen, _r = m.root_policy(state, root_noise=False, temperature=0.0)
        x, z, _y = action_to_xyz(chosen)
        state = apply_move(state, x, z)
        plies += 1
    if state.winner == DRAW:
        return DRAW, plies
    return (WHITE if state.winner == WHITE else BLACK), plies


def worker(games, start, seed, out_q):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    net = load_net()
    # net_a = BATCHED; net_b = sequential
    cfg_a = MCTSConfig(simulations=SIMS, batch_eval_size=BES)
    cfg_b = MCTSConfig(simulations=SIMS, batch_eval_size=32)
    for j in range(games):
        a_is_white = (start + j) % 2 == 0
        if a_is_white:
            r, p = _play_two(net, cfg_a, True)
        else:
            r, p = _play_two(net, cfg_b, False)
        out_q.put((_result_in_a_frame(r, a_is_white), p))


def run():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    workers = 8
    counts = [GAMES // workers + (1 if i < GAMES % workers else 0) for i in range(workers)]
    procs = []
    start = 0
    import time
    t0 = time.perf_counter()
    for i, n in enumerate(counts):
        p = ctx.Process(target=worker, args=(n, start, 100 + i, q), daemon=True)
        p.start()
        procs.append(p)
        start += n
    results = [q.get(timeout=3600) for _ in range(GAMES)]
    for p in procs:
        p.join(timeout=60)
    a_wins = sum(1 for r, _ in results if r == WHITE)
    b_wins = sum(1 for r, _ in results if r == BLACK)
    draws = sum(1 for r, _ in results if r == DRAW)
    ratio = (a_wins + 0.5 * draws) / GAMES
    print(f"batched bes={BES} vs sequential, {SIMS} sims, {GAMES} games:")
    print(f"  batched {a_wins}W  sequential {b_wins}W  draws {draws}"
          f"  -> ratio {ratio:.3f} (par >= 0.45)")
    print(f"  {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    run()
