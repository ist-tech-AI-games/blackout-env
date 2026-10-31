"""policy_server.serve with a stand-in policy, over a real Unix socket."""

import importlib.util
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from blackout_env.competition.protocol import encode_msg, recv_msg

pytestmark = pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="needs Unix sockets")

_path = Path(__file__).parent.parent / "blackout_env" / "competition" / "policy_server.py"
_spec = importlib.util.spec_from_file_location("policy_server", _path)
server_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server_mod)
sys.path.remove(str(_path.parent))  # policy_server puts its own folder on sys.path


class RecordingPolicy:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.append(("reset",))

    def observe(self, vector, graphic, slots):
        self.calls.append(("observe", int(vector[0, 0])))

    def act(self, vector, graphic, slots):
        self.calls.append(("act", int(vector[0, 0])))
        assert graphic.shape == (len(slots), 11, 2, 2)  # CHW, one row per agent
        return np.zeros((len(slots), 2), np.float32)


def request(seq, reset=False):
    vector = np.full((1, 96), seq, np.float32)
    graphic = np.zeros((1, 2, 2, 11), np.uint8)
    header = {"seq": seq, "agents": ["unit_0"], "vector_shape": [1, 96], "graphic_shape": [1, 2, 2, 11],
              "graphic_index": [0], "slots": [0]}
    if reset:
        header["reset"] = True
    return encode_msg(header, vector.tobytes() + graphic.tobytes())


def test_stale_requests_are_observed_and_only_the_newest_is_answered(tmp_path):
    sock_path = tmp_path / "p.sock"
    policy = RecordingPolicy()
    thread = threading.Thread(target=server_mod.serve, args=(policy, sock_path), daemon=True)
    thread.start()
    while not sock_path.exists():
        time.sleep(0.01)

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(sock_path))
    client.sendall(request(1, reset=True) + request(2) + request(3))  # queued before the server reads
    header, body = recv_msg(client, bytearray(), 1 << 16)
    client.sendall(encode_msg({"type": "close"}))
    thread.join(5)
    client.close()

    assert header == {"seq": 3} and len(body) == 8
    assert policy.calls == [("reset",), ("observe", 1), ("observe", 2), ("act", 3)]
