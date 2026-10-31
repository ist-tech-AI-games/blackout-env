"""
Check an ONNX submission before sending it in: layout, input names/shapes, output shape,
and time per step on this machine.

Runs as a plain script without importing blackout_env (whose mlagents dependency pins an old
protobuf that clashes with onnx), so it works in a separate export environment:

    pip install numpy onnxruntime        # or onnxruntime-gpu
    python blackout_env/competition/check_submission.py my_team/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Import onnx_policy as a sibling file, not through the blackout_env package: the package
# pulls in mlagents-envs, whose old protobuf pin breaks the onnx export environment.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from onnx_policy import GRAPHIC_SHAPE, TEAM_SIZE, VECTOR_SIZE, SubmissionError, load_policy  # noqa: E402

STEP_LIMIT_MS = 100.0


def main() -> None:
    p = argparse.ArgumentParser(description="Validate a BlackOut ONNX submission")
    p.add_argument("submission", type=Path)
    p.add_argument("--cpu", action="store_true", help="run on CPU even if CUDA is available")
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()

    try:
        policy = load_policy(args.submission, ["CPUExecutionProvider"] if args.cpu else None)
        policy.warm_up()
    except SubmissionError as e:
        sys.exit(f"FAIL: {e}")
    print(f"mode={policy.mode}  models={policy.n_models}  providers={policy.providers}")
    print(f"inputs: {sorted(policy.inputs)}")

    rng = np.random.default_rng(0)
    slots = np.arange(TEAM_SIZE, dtype=np.int64)
    times = []
    for _ in range(args.steps):
        vector = rng.uniform(-1, 1, (TEAM_SIZE, VECTOR_SIZE)).astype(np.float32)
        graphic = np.repeat((rng.random((1, *GRAPHIC_SHAPE)) < 0.1).astype(np.float32), TEAM_SIZE, 0)
        t0 = time.perf_counter()
        try:
            action = policy.act(vector, graphic, slots)
        except Exception as e:
            sys.exit(f"FAIL: {type(e).__name__}: {e}")
        times.append((time.perf_counter() - t0) * 1000)
        if not np.isfinite(action).all():
            print("WARN: non-finite actions (they are replaced with 0 in matches)")
            break

    t = np.array(times)
    print(f"time per step: mean {t.mean():.2f} ms, p99 {np.percentile(t, 99):.2f} ms, max {t.max():.2f} ms "
          f"(limit {STEP_LIMIT_MS:.0f} ms incl. transfer, on a 50% share of an RTX 3090)")
    if np.percentile(t, 99) > STEP_LIMIT_MS * 0.5:
        print("WARN: p99 is over half the limit; leave headroom for the slower evaluation GPU share")
    print("OK")


if __name__ == "__main__":
    main()
