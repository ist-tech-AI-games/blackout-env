"""
Policy server that runs inside a team's container.

Loads the team's submission (policy.py + checkpoint.pt [+ meta.json]) and answers
action requests from the match runner over a Unix socket. The container has no network;
the socket file in a shared volume is its only channel to the outside.

This file is copied into the policy image next to protocol.py and run as a script,
so it must not import blackout_env.

Submission layout:
    /submission/policy.py       nn.Module subclass: forward(vector, graphic) -> action
    /submission/checkpoint.pt   state dict, raw or under "policy_state"
    /submission/meta.json       optional: {"class_name": "MyPolicy",
                                           "kwargs": {"vector_size": 96, "n_channels": 11},
                                           "state_dict_key": "policy_state"}
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import socket
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import recv_latest, send_msg  # noqa: E402

MAX_REQUEST_BYTES = 64 << 20


def log(msg: str) -> None:
    print(f"[policy_server] {msg}", flush=True)


def load_policy(submission: Path, device):
    import torch
    import torch.nn as nn

    meta_path = submission / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    spec = importlib.util.spec_from_file_location("policy", submission / "policy.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(submission))
    spec.loader.exec_module(module)

    class_name = meta.get("class_name")
    if class_name is None:
        candidates = [
            obj for obj in vars(module).values()
            if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj.__module__ == "policy"
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"policy.py defines {len(candidates)} nn.Module classes; set class_name in meta.json"
            )
        cls = candidates[0]
    else:
        cls = getattr(module, class_name)

    net = cls(**meta.get("kwargs", {}))
    raw = torch.load(submission / "checkpoint.pt", map_location=device, weights_only=True)
    key = meta.get("state_dict_key", "policy_state")
    state_dict = raw[key] if key is not None and isinstance(raw, dict) and key in raw else raw
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    log(f"loaded {cls.__name__} on {device}")
    return net


def serve(net, device, sock_path: Path) -> None:
    import torch

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
                    ids = np.frombuffer(body[v_bytes:], dtype=np.uint8).reshape(g_shape[:3])
                    graphic = ids[..., None] == np.arange(g_shape[3], dtype=np.uint8)
                    with torch.no_grad():
                        v = torch.from_numpy(vector.copy()).to(device)
                        g = torch.from_numpy(graphic).to(device).float().permute(0, 3, 1, 2)
                        out = net(v, g)
                    action = out.detach().float().clamp(-1, 1).cpu().numpy().astype(np.float32)
                    if action.shape != (b, 2):
                        raise ValueError(f"forward returned shape {tuple(action.shape)}, expected ({b}, 2)")
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

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Create the CUDA context before running submission code, so the MPS
        # limits from the environment are applied to it.
        torch.zeros(1, device=device)
        log(f"cuda: {torch.cuda.get_device_name(0)}, "
            f"mps_thread_pct={os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')}, "
            f"mps_mem_limit={os.environ.get('CUDA_MPS_PINNED_DEVICE_MEM_LIMIT')}")

    try:
        net = load_policy(Path(args.submission), device)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
    serve(net, device, Path(args.socket))


if __name__ == "__main__":
    main()
