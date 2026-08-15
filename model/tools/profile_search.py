"""Profile one server-backed search: where does the time go?"""
import cProfile
import io
import pstats
import sys
import time

sys.path.insert(0, ".")

import torch

from smartfour.config import load_config
from smartfour.inference_server import InferenceServerHandle, RemoteEvaluator
from smartfour.network import ResNet


def run():
    cfg = load_config("config.toml")
    net = ResNet(cfg.network).eval()
    st = {k: v.cpu() for k, v in net.state_dict().items()}
    server = InferenceServerHandle(cfg.network, "mps", slots=1).start(initial_states=[st])
    ev = RemoteEvaluator(server.address, slot=0)

    from smartfour.game import initial_state
    from smartfour.mcts import MCTS
    m = MCTS(None, cfg.mcts, evaluator=ev)
    # warmup
    m.root_policy(initial_state(), root_noise=True, temperature=1.0)

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    for _ in range(5):
        m.root_policy(initial_state(), root_noise=True, temperature=1.0)
    pr.disable()
    dt = time.perf_counter() - t0
    s = m.last_stats
    print(f"per move: {dt / 5 * 1000:.0f}ms  sims={s['sims_done']}"
          f"  forwards={s['net_forwards']}  batch_mean={s['batch_size_mean']:.0f}"
          f"  depth={s['depth_mean']:.1f}")
    out = io.StringIO()
    pstats.Stats(pr, stream=out).sort_stats("tottime").print_stats(14)
    print(out.getvalue())
    ev.close()
    server.shutdown()


if __name__ == "__main__":
    run()
