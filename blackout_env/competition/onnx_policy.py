"""
Load and run a team's ONNX submission.

Copied into the policy image next to policy_server.py, so it must not import blackout_env.
check_submission.py uses the same code to validate a submission before it is sent in.

Submission layout:
    meta.json              {"mode": "team"} or {"mode": "unit"}
    model.onnx             team mode, or unit mode with one model shared by all 5 units
    unit_0.onnx ~ unit_4.onnx
                           unit mode only: unit i (slot i) uses unit_i.onnx

ONNX inputs (by name; a model declares only the ones it uses, at least one):
    vector   float32 (B, 96)
    graphic  float32 (B, 11, 96, 96)    0/1 channel masks, CHW
    slot     int64   (B,)               unit's index within its team (0~4)
Output: the first output, float32 (B, 2) = (dx, dy) per row.

team mode: one call per step, B = 5, row i = slot i (unit-number order).
unit mode: one call per unit per step, B = 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TEAM_SIZE = 5
VECTOR_SIZE = 96
GRAPHIC_SHAPE = (11, 96, 96)
# Exporters drop inputs the graph does not use, so none of these is required.
KNOWN_INPUTS = ("vector", "graphic", "slot")
INPUT_DTYPES = {"vector": "tensor(float)", "graphic": "tensor(float)", "slot": "tensor(int64)"}


class SubmissionError(Exception):
    pass


def default_providers() -> list[str]:
    import onnxruntime as ort
    available = ort.get_available_providers()
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]


def _check_dims(name: str, declared: list, expected: tuple) -> None:
    if len(declared) != len(expected):
        raise SubmissionError(f"input {name!r} has rank {len(declared)}, expected shape {expected}")
    for d, e in zip(declared, expected):
        if isinstance(d, int) and d != e:  # symbolic (str/None) dims accept any size
            raise SubmissionError(f"input {name!r} has shape {declared}, expected {expected}")


class _Session:
    def __init__(self, path: Path, batch: int, providers: list[str]):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        except Exception as e:
            raise SubmissionError(f"{path.name}: cannot load ({type(e).__name__}: {e})")

        inputs = {i.name: i for i in self.sess.get_inputs()}
        unknown = set(inputs) - set(KNOWN_INPUTS)
        if unknown or not inputs:
            raise SubmissionError(f"{path.name}: inputs must be named from {list(KNOWN_INPUTS)}, "
                                  f"got {sorted(inputs)}")
        expected = {"vector": (batch, VECTOR_SIZE), "graphic": (batch, *GRAPHIC_SHAPE), "slot": (batch,)}
        for name, inp in inputs.items():
            if inp.type != INPUT_DTYPES[name]:
                raise SubmissionError(f"{path.name}: input {name!r} is {inp.type}, expected {INPUT_DTYPES[name]}")
            _check_dims(name, inp.shape, expected[name])
        self.inputs = set(inputs)
        self.use_slot = "slot" in inputs
        self.output = self.sess.get_outputs()[0].name
        self.batch = batch
        self.name = path.name

    def run(self, vector: np.ndarray, graphic: np.ndarray, slot: np.ndarray) -> np.ndarray:
        feed = {k: v for k, v in (("vector", vector), ("graphic", graphic), ("slot", slot)) if k in self.inputs}
        out = np.asarray(self.sess.run([self.output], feed)[0])
        if out.shape != (self.batch, 2):
            raise ValueError(f"{self.name} returned shape {out.shape}, expected ({self.batch}, 2)")
        return out.astype(np.float32, copy=False)


class OnnxPolicy:
    def __init__(self, submission: Path, providers: list[str] | None = None):
        providers = providers or default_providers()
        meta_path = submission / "meta.json"
        if not meta_path.exists():
            raise SubmissionError('meta.json is missing (expected {"mode": "team"} or {"mode": "unit"})')
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as e:
            raise SubmissionError(f"meta.json is not valid JSON: {e}")
        self.mode = meta.get("mode")
        if self.mode not in ("team", "unit"):
            raise SubmissionError(f'meta.json "mode" must be "team" or "unit", got {self.mode!r}')

        shared = submission / "model.onnx"
        per_unit = [submission / f"unit_{i}.onnx" for i in range(TEAM_SIZE)]
        if self.mode == "team":
            if not shared.exists():
                raise SubmissionError("team mode needs model.onnx")
            self.sessions = [_Session(shared, TEAM_SIZE, providers)]
        elif shared.exists():
            self.sessions = [_Session(shared, 1, providers)] * TEAM_SIZE
        else:
            missing = [p.name for p in per_unit if not p.exists()]
            if missing:
                raise SubmissionError(f"unit mode needs model.onnx or all of unit_0~4.onnx (missing {missing})")
            self.sessions = [_Session(p, 1, providers) for p in per_unit]
        self.providers = self.sessions[0].sess.get_providers()

    def act(self, vector: np.ndarray, graphic: np.ndarray, slots: np.ndarray) -> np.ndarray:
        """
        vector (N, 96) float32, graphic (N, 11, 96, 96) float32, slots (N,) int64 in 0~4,
        rows in slot order. Returns (N, 2) float32.
        """
        n = len(slots)
        if self.mode == "team":
            if n == TEAM_SIZE:
                return self.sessions[0].run(vector, graphic, slots)
            # Fewer units than the model's batch: pad the missing slots with zeros.
            v = np.zeros((TEAM_SIZE, VECTOR_SIZE), np.float32)
            g = np.zeros((TEAM_SIZE, *GRAPHIC_SHAPE), np.float32)
            v[slots], g[slots] = vector, graphic
            return self.sessions[0].run(v, g, np.arange(TEAM_SIZE, dtype=np.int64))[slots]
        out = np.empty((n, 2), np.float32)
        for row, s in enumerate(slots):
            out[row] = self.sessions[s].run(vector[row:row + 1], graphic[row:row + 1], slots[row:row + 1])[0]
        return out

    def warm_up(self) -> None:
        self.act(np.zeros((TEAM_SIZE, VECTOR_SIZE), np.float32),
                 np.zeros((TEAM_SIZE, *GRAPHIC_SHAPE), np.float32),
                 np.arange(TEAM_SIZE, dtype=np.int64))
