"""
Policy server that runs inside a team's container.

Loads the team's ONNX submission (see onnx_policy.py for the layout and IO) and answers
action requests from the match runner over a Unix socket. The container has no network;
the socket file in a shared volume is its only channel to the outside.

This file is copied into the policy image next to protocol.py and onnx_policy.py and
run as a script, so it must not import blackout_env.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onnx_policy import Policy, load_policy  # noqa: E402
from protocol import recv_latest, send_msg  # noqa: E402

MAX_REQUEST_BYTES = 64 << 20


def log(msg: str) -> None:
    print(f"[policy_server] {msg}", flush=True)


def serve(policy: Policy, sock_path: Path) -> None:
    if sock_path.exists():
        sock_path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(1)
    log(f"listening on {sock_path}")

    while True:
        conn, _ = server.accept()
        buf = bytearray()
        try:
            while True:
                header, body = recv_latest(conn, buf, MAX_REQUEST_BYTES)
                if header.get("type") == "close":
                    log("close requested")
                    return
                seq = header["seq"]
                try:
                    b, n = header["vector_shape"]
                    g_shape = tuple(header["graphic_shape"])
                    v_bytes = b * n * 4
                    vector = np.frombuffer(body[:v_bytes], dtype=np.float32).reshape(b, n)
                    graphic = np.frombuffer(body[v_bytes:], dtype=np.uint8).reshape(g_shape)
                    # (U, H, W, C) uint8 -> (U, C, H, W) float32, then one row per agent
                    graphic = graphic.transpose(0, 3, 1, 2).astype(np.float32)
                    graphic = graphic[np.asarray(header["graphic_index"], dtype=np.int64)]
                    slots = np.asarray(header["slots"], dtype=np.int64)
                    action = policy.act(vector, graphic, slots)
                    send_msg(conn, {"seq": seq}, action.tobytes())
                except Exception as e:  # report per-step failures, keep serving
                    send_msg(conn, {"seq": seq, "error": f"{type(e).__name__}: {e}"})
        except (ConnectionError, ValueError, KeyError) as e:
            log(f"connection dropped ({type(e).__name__}: {e}); waiting for a new connection")
        finally:
            conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="/submission")
    parser.add_argument("--socket", default="/sock/policy.sock")
    args = parser.parse_args()

    log(f"mps_thread_pct={os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')}, "
        f"mps_mem_limit={os.environ.get('CUDA_MPS_PINNED_DEVICE_MEM_LIMIT')}")
    try:
        policy = load_policy(Path(args.submission))
        policy.warm_up()
    except Exception:
        traceback.print_exc()
        sys.exit(2)
    log(f"loaded {policy.mode} mode, {policy.n_models} model(s), providers={policy.providers}")
    serve(policy, Path(args.socket))


if __name__ == "__main__":
    main()
