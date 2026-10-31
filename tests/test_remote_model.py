"""RemoteModel against a stand-in policy server on a real Unix socket."""

import socket
import threading
import time

import numpy as np
import pytest

from blackout_env.competition.protocol import recv_latest, send_msg
from blackout_env.competition.remote_model import RemoteModel

pytestmark = pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="needs Unix sockets")


def serve(path, handler):
    """Accept one connection; handler(header, body) -> (header, body) or None (no answer)."""
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(path))
    srv.listen(1)
    requests = []

    def loop():
        conn, _ = srv.accept()
        buf = bytearray()
        try:
            while True:
                header, body = recv_latest(conn, buf, 1 << 26)
                if header.get("type") == "close":
                    return
                requests.append(header)
                reply = handler(header, body)
                if reply is not None:
                    send_msg(conn, *reply)
        except ConnectionError:
            pass
        finally:
            conn.close()
            srv.close()

    threading.Thread(target=loop, daemon=True).start()
    return requests


def team_obs(units=(7, 5, 9, 6, 8)):
    graphic = np.zeros((96, 96, 11), np.float32)  # one array shared by the team, as the env does
    return {f"unit_{i}": {"vector": np.full(96, i, np.float32), "graphic": graphic} for i in units}


def echo_slots(header, body):
    slots = np.asarray(header["slots"], np.float32) / 10  # stay inside the [-1, 1] clip
    return {"seq": header["seq"]}, np.stack([slots, slots], 1).astype(np.float32).tobytes()


def test_rows_are_sent_in_unit_order_with_slots(tmp_path):
    requests = serve(tmp_path / "p.sock", echo_slots)
    model = RemoteModel(str(tmp_path / "p.sock"))
    actions = model.act(team_obs())
    model.close()

    req = requests[0]
    assert req["agents"] == ["unit_5", "unit_6", "unit_7", "unit_8", "unit_9"]
    assert req["slots"] == [0, 1, 2, 3, 4]
    assert req["graphic_shape"] == [1, 96, 96, 11] and req["graphic_index"] == [0] * 5
    np.testing.assert_allclose(actions["unit_8"], [0.3, 0.3])
    assert model.stats.errors == 0 and model.stats.timeouts == 0


def test_timeout_and_error_give_zero_actions(tmp_path):
    def handler(header, body):
        if header["seq"] == 2:
            time.sleep(0.3)
            return None                                       # too slow, never answers
        if header["seq"] == 3:
            return {"seq": header["seq"], "error": "boom"}, b""
        return echo_slots(header, body)

    serve(tmp_path / "p.sock", handler)
    model = RemoteModel(str(tmp_path / "p.sock"), step_timeout=0.1)
    model.act(team_obs())                                     # 1: ok
    slow = model.act(team_obs())                              # 2: timeout
    time.sleep(0.3)                                           # let the server finish step 2
    failed = model.act(team_obs())                            # 3: error
    model.close()

    assert all(v.tolist() == [0.0, 0.0] for v in slow.values())
    assert all(v.tolist() == [0.0, 0.0] for v in failed.values())
    assert model.stats.timeouts == 1 and model.stats.errors == 1 and model.stats.last_error == "boom"


def test_nan_is_zeroed_and_actions_clipped(tmp_path):
    def handler(header, body):
        out = np.array([[np.nan, 5.0]] * len(header["agents"]), np.float32)
        return {"seq": header["seq"]}, out.tobytes()

    serve(tmp_path / "p.sock", handler)
    model = RemoteModel(str(tmp_path / "p.sock"))
    actions = model.act(team_obs((0, 1)))
    model.close()
    assert actions["unit_0"].tolist() == [0.0, 1.0]
