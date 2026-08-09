"""Persistent JSON-lines worker for the UI bridge.

Reads one request per line on stdin, writes one response per line on stdout.
The checkpoint is loaded once at startup, then each request is answered with
the same loaded agent (no per-move process or model-load cost):

    ->  {"id": 1, "state": {state_to_json...}, "simulations": 100}
    <-  {"id": 1, "move": {"x": 2, "z": 2}}        # or {"move": null} when terminal
    <-  {"id": 1, "error": "ValueError: ..."}      # in-band, loop keeps serving

A single {"ready": true, "iteration": n, "device": "<cuda|mps|cpu>"} line is
printed once the checkpoint is loaded so the bridge can wait for readiness
before sending requests. The device is auto-detected (CUDA > MPS > CPU).
`simulations=0` means policy-only (no MCTS search), matching
`SmartFourAgent.choose_move`.

CLI:  python -m smartfour.worker --checkpoint checkpoints/best.pt
"""

import argparse
import json
import sys

from .device import device_name
from .infer import SmartFourAgent


class Worker:
    """Pure request handler: dict in, dict out. Errors never raise."""

    def __init__(self, agent):
        self.agent = agent

    def handle(self, req: dict) -> dict:
        try:
            rid = req.get("id")
            if not isinstance(rid, int):
                raise ValueError("request 'id' must be an int")
            simulations = req.get("simulations", 0)
            if not isinstance(simulations, int) or simulations < 0:
                raise ValueError("'simulations' must be an int >= 0")
            state = req.get("state")
            if state is None:
                raise ValueError("missing 'state'")
            move = self.agent.choose_move(state, simulations=simulations)
            return {
                "id": rid,
                "move": None if move is None else {"x": move[0], "z": move[1]},
            }
        except Exception as exc:
            rid = req.get("id") if isinstance(req.get("id"), int) else None
            return {"id": rid, "error": f"{type(exc).__name__}: {exc}"}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Serve smart-four moves over JSON lines")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args(argv)

    agent = SmartFourAgent(args.checkpoint)
    print(
        json.dumps({
            "ready": True,
            "iteration": agent.iteration,
            "device": device_name(agent.device),
        }),
        flush=True,
    )

    worker = Worker(agent)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"id": None, "error": f"JSONDecodeError: {exc}"}), flush=True)
            continue
        print(json.dumps(worker.handle(req)), flush=True)


if __name__ == "__main__":
    main()
