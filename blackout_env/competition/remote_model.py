"""
BaseModel that forwards observations to a team's policy container over a Unix socket.

Each step has a deadline. If the container does not answer in time, answers with an
error, or sends a malformed response, the team's agents stand still (0, 0) for that
step and the failure is counted. Late responses are discarded by sequence number.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass

import numpy as np

from ..env.constants import unit_index
from ..model.base import BaseModel
from .protocol import encode_msg, try_parse

MAX_RESPONSE_BYTES = 1 << 16


@dataclass
class RemoteStats:
    steps: int = 0
    timeouts: int = 0
    errors: int = 0
    last_error: str | None = None
    max_latency_ms: float = 0.0


class RemoteModel(BaseModel):
    """
    Parameters
    ----------
    socket_path : str
        Unix socket created by the team's policy_server.
    step_timeout : float
        Seconds allowed per act() call.
    first_step_timeout : float
        Seconds allowed for the first call (CUDA warm-up, cudnn autotune, ...).
    """

    def __init__(
        self,
        socket_path: str,
        step_timeout: float = 0.1,
        first_step_timeout: float = 30.0,
        connect_timeout: float = 120.0,
        send_timeout: float = 5.0,
    ):
        self.step_timeout = step_timeout
        self.send_timeout = send_timeout
        self._broken = False
        self.first_step_timeout = first_step_timeout
        self.stats = RemoteStats()
        self._seq = 0
        self._buf = bytearray()
        self._sock = self._connect(socket_path, connect_timeout)

    @staticmethod
    def _connect(path: str, timeout: float) -> socket.socket:
        deadline = time.monotonic() + timeout
        while True:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(path)
                return s
            except (FileNotFoundError, ConnectionRefusedError):
                s.close()
                if time.monotonic() > deadline:
                    raise TimeoutError(f"policy server did not open {path} within {timeout}s")
                time.sleep(0.5)

    def act(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        agents = sorted(obs, key=unit_index)
        zeros = {a: np.zeros(2, dtype=np.float32) for a in agents}
        if not agents:
            return zeros
        if self._broken:
            self.stats.steps += 1
            self.stats.errors += 1
            return zeros

        vector = np.stack([obs[a]["vector"] for a in agents]).astype(np.float32)
        # graphic channels are one-hot masks: send the id map (H×W) instead of H×W×C floats
        graphic = np.stack([obs[a]["graphic"] for a in agents])
        ids = graphic.argmax(axis=-1).astype(np.uint8)
        ids[graphic.max(axis=-1) < 0.5] = 255

        self._seq += 1
        seq = self._seq
        header = {
            "seq": seq,
            "agents": agents,
            "vector_shape": list(vector.shape),
            "graphic_shape": list(graphic.shape),
        }
        timeout = self.first_step_timeout if self.stats.steps == 0 else self.step_timeout
        self.stats.steps += 1
        start = time.monotonic()

        try:
            # A partial send would corrupt the stream, so sending gets its own generous
            # timeout; if it still fails the connection is treated as dead.
            self._sock.settimeout(self.send_timeout)
            self._sock.sendall(encode_msg(header, vector.tobytes() + ids.tobytes()))
            resp = self._wait_for(seq, start + timeout)
        except (OSError, ValueError) as e:
            self._broken = True
            self.stats.errors += 1
            self.stats.last_error = f"connection lost: {type(e).__name__}: {e}"
            return zeros

        if resp is None:
            self.stats.timeouts += 1
            return zeros

        if self.stats.steps > 1:  # first step includes CUDA warm-up
            self.stats.max_latency_ms = max(self.stats.max_latency_ms, (time.monotonic() - start) * 1000)
        resp_header, body = resp
        if "error" in resp_header:
            self.stats.errors += 1
            self.stats.last_error = str(resp_header["error"])[:500]
            return zeros
        if len(body) != len(agents) * 2 * 4:
            self.stats.errors += 1
            self.stats.last_error = f"bad action payload size {len(body)}"
            return zeros

        actions = np.frombuffer(body, dtype=np.float32).reshape(len(agents), 2)
        actions = np.clip(np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
        return {a: actions[i].copy() for i, a in enumerate(agents)}

    def _wait_for(self, seq: int, deadline: float) -> tuple[dict, bytes] | None:
        """Read until the response for seq arrives; drop stale ones. None on timeout."""
        while True:
            msg = try_parse(self._buf, MAX_RESPONSE_BYTES)
            if msg is not None:
                if msg[0].get("seq") == seq:
                    return msg
                continue  # late answer to an earlier step
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self._sock.settimeout(remaining)
            try:
                chunk = self._sock.recv(1 << 16)
            except socket.timeout:
                return None
            if not chunk:
                raise ConnectionError("policy server closed the connection")
            self._buf += chunk

    def close(self) -> None:
        try:
            self._sock.settimeout(1.0)
            self._sock.sendall(encode_msg({"type": "close"}))
        except OSError:
            pass
        self._sock.close()
