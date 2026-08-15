"""Central batched inference server for self-play and arena workers.

One process owns the accelerator (mps/cuda) nets; worker processes ship
encoded leaf batches to it over multiprocessing pipes and receive
(masked priors, values) back. The server greedily drains every queued
request into one net forward per slot, so the effective GPU batch scales
with the number of busy workers instead of one search's `batch_eval_size`.

Message framing: Connection.send_bytes/recv_bytes are length-prefixed, so
every logical message is exactly one bytes send. Requests are two messages
(JSON header, then raw float32 payload); replies likewise.

  client -> server   {"t":"eval","slot":s,"n":N}   + N*400 float32 (states)
  server -> client   {"t":"result","n":N}          + N*125+N float32 (priors|values)
  control           {"t":"set_weights","slot":s}   + pickled state_dict
                    {"t":"shutdown"} / {"t":"ping"} -> {"t":"ok"/"pong"}

Priors are masked-softmaxed over each state's legal actions on the server
(legality is recoverable from encoding channels 10-14: channel 10+h is set
at column (x,z) iff its stack height is h), so clients apply them directly.

Weights are only updated between phases (the trainer holds evals back), so
the server needs no versioning. Failures are loud: a dead server surfaces as
EOFError in the workers, which fail fast — never a silent CPU fallback.
"""

import json
import multiprocessing as mp
import pickle
import signal
import threading

import torch

from .config import NetworkConfig
from .device import resolve_device
from .network import ResNet

PLANES = 25
POLICY_FLOATS = 125
STATE_FLOATS = 16 * PLANES  # 400

NEG_INF = float("-inf")


# ------------------------------------------------------------------ messages

def _send_json(conn, header: dict) -> None:
    conn.send_bytes(json.dumps(header).encode())


def _recv_json(conn) -> dict:
    return json.loads(conn.recv_bytes())


# -------------------------------------------------------------------- server

def _server_main(net_cfg, device_name, slots, initial_states, ready_q,
                 authkey, num_threads):
    """Server process entry point. `initial_states` are CPU state_dicts."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        if num_threads:
            torch.set_num_threads(max(1, int(num_threads)))
        device = resolve_device(device_name)
        nets = []
        for s in range(slots):
            net = ResNet(net_cfg).to(device)
            if initial_states is not None and s < len(initial_states) and initial_states[s] is not None:
                net.load_state_dict(initial_states[s])
            net.eval()
            nets.append(net)
        listener = mp.connection.Listener(address=None, authkey=authkey, backlog=64)
        ready_q.put(listener.address)
        _serve(listener, nets, device)
    except Exception as exc:  # noqa: BLE001 — report, then die loudly
        try:
            ready_q.put(("__error__", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass


class _Request:
    __slots__ = ("conn", "slot", "n", "payload")

    def __init__(self, conn, slot, n):
        self.conn = conn
        self.slot = slot
        self.n = n
        self.payload = b""


def _serve(listener, nets, device):
    requests = []          # queued _Request objects
    cond = threading.Condition()
    conns = {}             # conn -> reader thread
    stop = threading.Event()

    def batcher():
        while not stop.is_set():
            with cond:
                while not requests and not stop.is_set():
                    cond.wait(timeout=0.1)
                if stop.is_set():
                    return
                batch = requests[:]
                requests.clear()
            _run_batch(batch, nets, device)

    t = threading.Thread(target=batcher, daemon=True)
    t.start()

    def reader(conn):
        try:
            while True:
                header = _recv_json(conn)
                kind = header.get("t")
                if kind == "eval":
                    payload = conn.recv_bytes()
                    req = _Request(conn, int(header["slot"]), int(header["n"]))
                    req.payload = payload
                    with cond:
                        requests.append(req)
                        cond.notify()
                elif kind == "set_weights":
                    state = pickle.loads(conn.recv_bytes())
                    nets[int(header["slot"])].load_state_dict(state)
                    _send_json(conn, {"t": "ok"})
                elif kind == "ping":
                    _send_json(conn, {"t": "pong"})
                elif kind == "shutdown":
                    _send_json(conn, {"t": "ok"})
                    stop.set()
                    with cond:
                        cond.notify_all()
                    return
                else:
                    _send_json(conn, {"t": "error", "msg": f"unknown type {kind!r}"})
        except (EOFError, OSError, ConnectionResetError):
            pass  # client died; drop the connection
    try:
        while not stop.is_set():
            try:
                conn = listener.accept()
            except EOFError:
                continue  # transient: a concurrent connection vanished
            except OSError as exc:
                if stop.is_set():
                    break
                # Transient accept errors (e.g. ECONNABORTED when a peer
                # dies mid-handshake) must not kill the server.
                import errno
                if exc.errno in (errno.ECONNABORTED, errno.EINTR, errno.EAGAIN):
                    continue
                break  # listener itself is gone
            conns[conn] = threading.Thread(target=reader, args=(conn,), daemon=True)
            conns[conn].start()
    finally:
        stop.set()
        with cond:
            cond.notify_all()
        try:
            listener.close()
        except Exception:
            pass


def _run_batch(batch, nets, device):
    """Evaluate every queued request: one forward per slot, then reply."""
    by_slot = {}
    for req in batch:
        by_slot.setdefault(req.slot, []).append(req)
    for slot, reqs in by_slot.items():
        net = nets[slot]
        try:
            n_total = sum(r.n for r in reqs)
            data = torch.frombuffer(
                bytearray(b"".join(r.payload for r in reqs)), dtype=torch.float32
            )
            states = data.reshape(n_total, 16, 5, 5)
            mask = (states[:, 10:15] > 0.5).reshape(n_total, POLICY_FLOATS)
            with torch.no_grad():
                x = states.to(device)
                logits, values = net(x)
                logits = logits.masked_fill(
                    ~mask.to(device), torch.tensor(NEG_INF, device=device)
                )
                priors = torch.softmax(logits, dim=1).cpu()
                vals = values.squeeze(1).cpu()
            out = torch.empty((n_total, POLICY_FLOATS + 1), dtype=torch.float32)
            out[:, :POLICY_FLOATS] = priors
            out[:, POLICY_FLOATS] = vals
            blob = out.numpy().tobytes()
            off = 0
            for r in reqs:
                nb = r.n * (POLICY_FLOATS + 1) * 4
                _send_json(r.conn, {"t": "result", "n": r.n})
                r.conn.send_bytes(blob[off:off + nb])
                off += nb
        except Exception as exc:  # noqa: BLE001 — report to the requesters
            for r in reqs:
                try:
                    _send_json(r.conn, {"t": "error", "msg": f"{type(exc).__name__}: {exc}"})
                except Exception:
                    pass


# ------------------------------------------------------------- trainer handle

class InferenceServerHandle:
    """Trainer-side lifecycle handle for the server process."""

    def __init__(self, net_cfg: NetworkConfig, device: str, slots: int = 2,
                 num_threads: int = 0, authkey: bytes = b"smartfour"):
        self.net_cfg = net_cfg
        self.device = device
        self.slots = slots
        self.num_threads = num_threads
        self.address = None
        self.proc = None
        self._ctrl = None

    def start(self, initial_states=None):
        ctx = mp.get_context("spawn")
        ready_q = ctx.Queue()
        self.proc = ctx.Process(
            target=_server_main,
            args=(
                self.net_cfg, self.device, self.slots, initial_states,
                ready_q, b"smartfour", self.num_threads,
            ),
            daemon=True,
        )
        self.proc.start()
        msg = ready_q.get(timeout=300)
        if isinstance(msg, tuple) and msg and msg[0] == "__error__":
            raise RuntimeError(f"inference server failed to start: {msg[1]}")
        self.address = msg
        self._ctrl = mp.connection.Client(self.address, authkey=b"smartfour")
        return self

    def set_weights(self, slot: int, net_state: dict) -> None:
        clean = {k: v.detach().cpu().clone() for k, v in net_state.items()}
        _send_json(self._ctrl, {"t": "set_weights", "slot": slot})
        self._ctrl.send_bytes(pickle.dumps(clean, protocol=pickle.HIGHEST_PROTOCOL))
        reply = _recv_json(self._ctrl)
        if reply.get("t") != "ok":
            raise RuntimeError(f"set_weights failed: {reply}")

    def ping(self) -> None:
        _send_json(self._ctrl, {"t": "ping"})
        reply = _recv_json(self._ctrl)
        if reply.get("t") != "pong":
            raise RuntimeError(f"ping failed: {reply}")

    def shutdown(self) -> None:
        if self._ctrl is not None:
            try:
                _send_json(self._ctrl, {"t": "shutdown"})
                _recv_json(self._ctrl)
            except Exception:
                pass
            self._ctrl = None
        if self.proc is not None:
            self.proc.join(timeout=10)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=5)
            self.proc = None


# ------------------------------------------------------------------- client

class RemoteEvaluator:
    """Worker-side evaluator: (states) -> (masked priors, values) via the
    central server. One instance owns one connection; not thread-safe."""

    def __init__(self, address, slot: int, authkey: bytes = b"smartfour"):
        self.conn = mp.connection.Client(address, authkey=authkey)
        self.slot = slot

    def __call__(self, states):
        from .encode import encode  # local import keeps module import light
        n = len(states)
        t = torch.stack([encode(s) for s in states])
        _send_json(self.conn, {"t": "eval", "slot": self.slot, "n": n})
        self.conn.send_bytes(t.numpy().tobytes())
        header = _recv_json(self.conn)
        if header.get("t") == "error":
            raise RuntimeError(f"inference server error: {header.get('msg')}")
        if header.get("t") != "result" or header.get("n") != n:
            raise RuntimeError(f"protocol error: unexpected header {header}")
        payload = self.conn.recv_bytes()
        expect = n * (POLICY_FLOATS + 1) * 4
        if len(payload) != expect:
            raise RuntimeError(
                f"protocol error: payload {len(payload)} bytes, expected {expect}"
            )
        data = torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(n, POLICY_FLOATS + 1)
        priors = data[:, :POLICY_FLOATS].clone()
        values = data[:, POLICY_FLOATS].clone()
        return priors, values

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
