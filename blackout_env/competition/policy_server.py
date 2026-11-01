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
import faulthandler
import os
import socket
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onnx_policy import Policy, load_policy  # noqa: E402
from protocol import recv_pending, send_msg  # noqa: E402

MAX_REQUEST_BYTES = 64 << 20


def log(msg: str) -> None:
    print(f"[policy_server] {msg}", flush=True)


def decode_request(header: dict, body: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Request -> (vector (B, 96), graphic (B, 11, 96, 96) float32, slots (B,) int64)."""
    b, n = header["vector_shape"]
    g_shape = tuple(header["graphic_shape"])
    v_bytes = b * n * 4
    vector = np.frombuffer(body[:v_bytes], dtype=np.float32).reshape(b, n)
    graphic = np.frombuffer(body[v_bytes:], dtype=np.uint8).reshape(g_shape)
    # (U, H, W, C) uint8 -> (U, C, H, W) float32, then one row per agent
    graphic = graphic.transpose(0, 3, 1, 2).astype(np.float32)
    graphic = graphic[np.asarray(header["graphic_index"], dtype=np.int64)]
    return vector, graphic, np.asarray(header["slots"], dtype=np.int64)


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
                *stale, (header, body) = recv_pending(conn, buf, MAX_REQUEST_BYTES)
                if any(h.get("type") == "close" for h, _ in [*stale, (header, body)]):
                    log("close requested")
                    return
                # Steps the runner already gave up on are not answered, but their
                # observations still go into the frame stacks.
                for old_header, old_body in stale:
                    if old_header.get("reset"):
                        policy.reset()
                    try:
                        policy.observe(*decode_request(old_header, old_body))
                    except Exception as e:
                        log(f"skipped step {old_header.get('seq')} failed ({type(e).__name__}: {e})")
                seq = header["seq"]
                if header.get("reset"):
                    policy.reset()
                try:
                    action = policy.act(*decode_request(header, body))
                    send_msg(conn, {"seq": seq}, action.tobytes())
                except Exception as e:  # report per-step failures, keep serving
                    send_msg(conn, {"seq": seq, "error": f"{type(e).__name__}: {e}"})
        except ConnectionError as e:
            log(f"connection dropped ({type(e).__name__}: {e}); waiting for a new connection")
        except Exception:  # a malformed request or a bug here: drop this connection, keep serving
            log("closing the connection after an error; waiting for a new connection\n" + traceback.format_exc())
        finally:
            conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="/submission")
    parser.add_argument("--socket", default="/sock/policy.sock")
    args = parser.parse_args()
    faulthandler.enable()  # print the Python stack if onnxruntime crashes the process

    log(f"mps_thread_pct={os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')}, "
        f"mps_mem_limit={os.environ.get('CUDA_MPS_PINNED_DEVICE_MEM_LIMIT')}")
    try:
        policy = load_policy(Path(args.submission))
        policy.warm_up()
    except Exception:
        traceback.print_exc()
        sys.exit(2)
    log(f"loaded {policy.mode} mode, {policy.n_models} model(s), frame_stack={policy.frame_stack}, "
        f"states={sorted(policy.state_inputs)}, providers={policy.providers}")
    serve(policy, Path(args.socket))


if __name__ == "__main__":
    main()
