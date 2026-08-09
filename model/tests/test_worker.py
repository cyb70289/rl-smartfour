"""Tests for smartfour.worker — persistent JSON-lines worker (UI bridge).

The worker reads one request per line on stdin and writes one response per
line on stdout. Each response echoes the request id; errors are reported
in-band and the loop keeps serving. A single {"ready": true} line is printed
once the checkpoint is loaded, so the bridge can wait for readiness.
"""

import json
import subprocess
import sys
from pathlib import Path

from smartfour.game import initial_state, state_to_json
from smartfour.worker import Worker


class FakeAgent:
    """Records choose_move calls and returns a canned move."""

    def __init__(self, move=(2, 3)):
        self.move = move
        self.calls = []

    def choose_move(self, state, simulations=200):
        self.calls.append((state, simulations))
        return self.move


def req(**kw):
    base = {"id": 1, "state": state_to_json(initial_state()), "simulations": 100}
    base.update(kw)
    return base


# ---------------------------------------------------------------- Worker.handle

def test_handle_returns_move_with_matching_id():
    out = Worker(FakeAgent(move=(2, 3))).handle(req(id=7))
    assert out == {"id": 7, "move": {"x": 2, "z": 3}}


def test_handle_passes_state_and_simulations_through():
    agent = FakeAgent()
    data = req(simulations=42)
    Worker(agent).handle(data)
    assert agent.calls == [(data["state"], 42)]


def test_handle_terminal_move_is_null():
    out = Worker(FakeAgent(move=None)).handle(req(id=1))
    assert out == {"id": 1, "move": None}


def test_handle_missing_state_is_an_error_that_keeps_the_id():
    out = Worker(FakeAgent()).handle({"id": 5, "simulations": 10})
    assert out["id"] == 5
    assert "error" in out


def test_handle_bad_simulations_is_an_error():
    for bad in [-1, 1.5, "100", None]:
        out = Worker(FakeAgent()).handle(req(simulations=bad))
        assert out["id"] == 1, bad
        assert "error" in out, bad


def test_handle_bad_id_reports_null_id():
    out = Worker(FakeAgent()).handle(req(id="nope"))
    assert out["id"] is None
    assert "error" in out


def test_handle_agent_exception_is_reported_in_band():
    class Boom:
        def choose_move(self, state, simulations=0):
            raise RuntimeError("net exploded")

    out = Worker(Boom()).handle(req(id=3))
    assert out["id"] == 3
    assert "RuntimeError" in out["error"]


# ---------------------------------------------------------------- CLI loop

def make_checkpoint(path):
    import torch
    from dataclasses import asdict

    from smartfour.config import NetworkConfig
    from smartfour.network import ResNet

    torch.manual_seed(0)
    net_cfg = NetworkConfig(
        input_channels=16, blocks=1, base_channels=8,
        policy_channels=4, value_channels=4, value_fc=8,
    )
    net = ResNet(net_cfg)
    torch.save(
        {"net_state": net.state_dict(), "iteration": 3, "network": asdict(net_cfg)},
        path,
    )


def spawn_worker(tmp_path):
    ckpt = tmp_path / "tiny.pt"
    make_checkpoint(ckpt)
    root = Path(__file__).resolve().parent.parent
    return subprocess.Popen(
        [sys.executable, "-m", "smartfour.worker", "--checkpoint", str(ckpt)],
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def read_line(proc):
    line = proc.stdout.readline()
    assert line, "worker closed stdout unexpectedly"
    return json.loads(line)


def test_cli_ready_then_serves_requests(tmp_path):
    proc = spawn_worker(tmp_path)
    try:
        ready = read_line(proc)
        assert ready["ready"] is True
        assert ready["iteration"] == 3

        # policy-only request (simulations=0) still yields a move
        proc.stdin.write(json.dumps(req(id=1, simulations=0)) + "\n")
        proc.stdin.flush()
        out = read_line(proc)
        assert out["id"] == 1
        assert out["move"] is not None
        assert 0 <= out["move"]["x"] < 5 and 0 <= out["move"]["z"] < 5

        # malformed JSON: error with null id, loop keeps serving
        proc.stdin.write("not json\n")
        proc.stdin.flush()
        out = read_line(proc)
        assert out["id"] is None
        assert "error" in out

        # bad request: error keeps the id, loop keeps serving
        proc.stdin.write(json.dumps({"id": 2, "state": "nope", "simulations": 5}) + "\n")
        proc.stdin.flush()
        out = read_line(proc)
        assert out["id"] == 2
        assert "error" in out

        # still alive: next request is answered normally
        proc.stdin.write(json.dumps(req(id=3)) + "\n")
        proc.stdin.flush()
        out = read_line(proc)
        assert out["id"] == 3
        assert out["move"] is not None
    finally:
        proc.kill()
        proc.wait(timeout=10)
