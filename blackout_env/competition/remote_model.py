"""
BaseModel that forwards observations to a team's policy container over a Unix socket.

Each step has a deadline. If the container does not answer in time, answers with an
error, or sends a malformed response, the team's agents stand still (0, 0) for that
step and the failure is counted. Late responses are discarded by sequence number.

If the connection is lost (e.g. the server process crashed and the container is restarting
it), the step is held: the model waits up to reconnect_timeout for the server to come back
and sends the same request again, so the match does not move on while a team is down.
After max_reconnects such waits, or if the server does not come back in time, the agents
stand still and the model only retries the connection every RECONNECT_INTERVAL seconds.
A restarted server process starts with empty frame stacks and zero states.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass

import numpy as np

from ..env.constants import TEAM_SIZE, unit_index
from ..model.base import BaseModel
from .protocol import encode_msg, try_parse

MAX_RESPONSE_BYTES = 1 << 16
RECONNECT_INTERVAL = 0.5


@dataclass
class RemoteStats:
    steps: int = 0
    timeouts: int = 0
    errors: int = 0
    last_error: str | None = None
    reconnects: int = 0
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
        Seconds allowed for the first call (the server already ran one warm-up call).
    reconnect_timeout : float
        Seconds to hold a step while waiting for a lost server to come back.
    max_reconnects : int
        How many times the step may be held over the model's lifetime (one series).
    """

    def __init__(
        self,
        socket_path: str,
        step_timeout: float = 0.1,
        first_step_timeout: float = 30.0,
        connect_timeout: float = 120.0,
        send_timeout: float = 5.0,
        reconnect_timeout: float = 60.0,
        max_reconnects: int = 5,
    ):
        self.reconnect_timeout = reconnect_timeout
        self.max_reconnects = max_reconnects
        self.step_timeout = step_timeout
        self.send_timeout = send_timeout
        self._broken = False
        self.first_step_timeout = first_step_timeout
        self.stats = RemoteStats()
        self._seq = 0
        self._new_episode = True
        self._buf = bytearray()
        self._path = socket_path
        self._next_reconnect = 0.0
        self._sock = self._connect(socket_path, connect_timeout)

    @staticmethod
    def _connect(path: str, timeout: float) -> socket.socket:
        deadline = time.monotonic() + timeout
        while True:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(path)
                return s
            except (FileNotFoundError, ConnectionRefusedError) as e:
                s.close()
                if time.monotonic() > deadline:
                    raise TimeoutError(f"policy server did not open {path} within {timeout}s") from e
                time.sleep(0.5)

    def reset(self) -> None:
        # Sent with the next request so the server clears frame stacks and states.
        self._new_episode = True

    def act(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        agents = sorted(obs, key=unit_index)
        zeros = {a: np.zeros(2, dtype=np.float32) for a in agents}
        if not agents:
            return zeros
        if self._broken and not self._reconnect():
            self.stats.steps += 1
            self.stats.errors += 1
            return zeros

        vector = np.stack([obs[a]["vector"] for a in agents]).astype(np.float32)
        # Agents of one team share the same graphic array: send each distinct array once
        # (as uint8, the channels are 0/1 masks) plus a per-agent index into them.
        unique: dict[int, int] = {}
        graphics = []
        graphic_index = []
        for a in agents:
            g = obs[a]["graphic"]
            if id(g) not in unique:
                unique[id(g)] = len(graphics)
                graphics.append(g)
            graphic_index.append(unique[id(g)])
        graphic = np.stack(graphics).astype(np.uint8)

        self._seq += 1
        seq = self._seq
        header = {
            "seq": seq,
            "agents": agents,
            "vector_shape": list(vector.shape),
            "graphic_shape": list(graphic.shape),
            "graphic_index": graphic_index,
            "slots": [unit_index(a) % TEAM_SIZE for a in agents],
        }
        if self._new_episode:
            header["reset"] = True
            self._new_episode = False
        timeout = self.first_step_timeout if self.stats.steps == 0 else self.step_timeout
        self.stats.steps += 1
        start = time.monotonic()

        msg = encode_msg(header, vector.tobytes() + graphic.tobytes())
        held = False
        try:
            resp = self._send_and_wait(msg, seq, start + timeout)
        except (OSError, ValueError) as e:
            self.stats.last_error = f"connection lost: {type(e).__name__}: {e}"
            held = True
            resp = self._retry_after_reconnect(msg, seq)
            if resp is False:
                self._broken = True
                self.stats.errors += 1
                return zeros

        if resp is None:
            self.stats.timeouts += 1
            return zeros

        if self.stats.steps > 1 and not held:  # first step may include remaining warm-up
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

    def _send_and_wait(self, msg: bytes, seq: int, deadline: float) -> tuple[dict, bytes] | None:
        # A partial send would corrupt the stream, so sending gets its own generous
        # timeout; if it still fails the connection is treated as dead.
        self._sock.settimeout(self.send_timeout)
        self._sock.sendall(msg)
        return self._wait_for(seq, deadline)

    def _retry_after_reconnect(self, msg: bytes, seq: int) -> tuple[dict, bytes] | None | bool:
        """Hold the step until the server is back, then resend. False if it did not come back."""
        if self.stats.reconnects >= self.max_reconnects:
            return False
        deadline = time.monotonic() + self.reconnect_timeout
        while not self._reconnect(wait=True):
            if time.monotonic() > deadline:
                return False
            time.sleep(0.2)
        try:
            return self._send_and_wait(msg, seq, time.monotonic() + self.first_step_timeout)
        except (OSError, ValueError) as e:
            self.stats.last_error = f"connection lost again: {type(e).__name__}: {e}"
            return False

    def _reconnect(self, wait: bool = False) -> bool:
        """One connection attempt; without wait, at most every RECONNECT_INTERVAL seconds."""
        now = time.monotonic()
        if not wait and now < self._next_reconnect:
            return False
        self._next_reconnect = now + RECONNECT_INTERVAL
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(RECONNECT_INTERVAL)
            s.connect(self._path)
        except OSError:
            s.close()
            return False
        self._sock.close()
        self._sock = s
        self._buf = bytearray()
        self._broken = False
        self.stats.reconnects += 1
        return True

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
            except TimeoutError:
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
