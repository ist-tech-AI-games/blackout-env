"""
Load and run a team's ONNX submission.

Copied into the policy image next to policy_server.py, so it must not import blackout_env.
check_submission.py uses the same code to validate a submission before it is sent in.

Submission layout:
    meta.json              {"mode": "team"} or {"mode": "unit"}, optionally "frame_stack": K
    model.onnx             team mode, or unit mode with one model shared by all 5 units
    unit_0.onnx ~ unit_4.onnx
                           unit mode only: unit i (slot i) uses unit_i.onnx

ONNX inputs (by name; a model declares only the ones it uses, at least one):
    vector   float32 (B, 96)               K = 1 (default)
             float32 (B, K, 96)            K > 1, frames oldest first, [:, -1] is the current one
    graphic  float32 (B, 11, 96, 96)       0/1 channel masks, CHW
             float32 (B, K, 11, 96, 96)    K > 1
    slot     int64   (B,)                  unit's index within its team (0~4)
    state_in*  float32 (B, ...)            recurrent state; needs a matching state_out* output
Outputs:
    action     the first output not named state_out*, float32 (B, 2) = (dx, dy) per row
    state_out* same shape as its state_in*; fed back as state_in* on the unit's next step

Frame stacks and states are kept per slot and cleared at the start of every episode. The
first frame of an episode fills all K frames, and states start at zero.

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
MAX_FRAME_STACK = 16
STATE_IN, STATE_OUT = "state_in", "state_out"
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
    for d, e in zip(declared, expected, strict=True):
        if isinstance(d, int) and d != e:  # symbolic (str/None) dims accept any size
            raise SubmissionError(f"input {name!r} has shape {declared}, expected {expected}")


class FrameStack:
    """
    Last k observations per key (a slot here; any hashable in training code), oldest first.
    The first observation of a key after reset() fills all k frames.

    Training code can import this class (blackout_env.competition.onnx_policy) so its
    stacked inputs match what the competition feeds the model. Pass graphic as CHW.
    """

    def __init__(self, k: int):
        self.k = k
        self.reset()

    def reset(self) -> None:
        self._frames: dict = {}  # key -> (vector ring (k, ...), graphic ring (k, ...), next write index)

    def push(self, keys, vector: np.ndarray, graphic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Add one frame per row and return (N, k, ...) stacks for those rows."""
        out_v = np.empty((len(keys), self.k, *vector.shape[1:]), vector.dtype)
        out_g = np.empty((len(keys), self.k, *graphic.shape[1:]), graphic.dtype)
        for row, key in enumerate(keys):
            if key not in self._frames:
                ring_v = np.repeat(vector[row:row + 1], self.k, axis=0)
                ring_g = np.repeat(graphic[row:row + 1], self.k, axis=0)
                pos = 0
            else:
                ring_v, ring_g, pos = self._frames[key]
                ring_v[pos], ring_g[pos] = vector[row], graphic[row]
                pos = (pos + 1) % self.k
            self._frames[key] = (ring_v, ring_g, pos)
            order = (pos + np.arange(self.k)) % self.k  # pos is the oldest frame
            out_v[row], out_g[row] = ring_v[order], ring_g[order]
        return out_v, out_g


class _Session:
    """One .onnx file, checked against the input spec for a given batch size and frame stack."""

    def __init__(self, path: Path, batch: int, providers: list[str], frame_stack: int = 1):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        except Exception as e:
            raise SubmissionError(f"{path.name}: cannot load ({type(e).__name__}: {e})") from e
        self.batch = batch
        self.name = path.name

        inputs = {i.name: i for i in self.sess.get_inputs()}
        outputs = {o.name: o for o in self.sess.get_outputs()}
        states = {n for n in inputs if n.startswith(STATE_IN)}
        unknown = set(inputs) - set(KNOWN_INPUTS) - states
        if unknown or not set(inputs) - states:
            raise SubmissionError(f"{path.name}: inputs must be named from {list(KNOWN_INPUTS)} "
                                  f"(plus optional {STATE_IN}*), got {sorted(inputs)}")
        stack = () if frame_stack == 1 else (frame_stack,)
        expected = {"vector": (batch, *stack, VECTOR_SIZE),
                    "graphic": (batch, *stack, *GRAPHIC_SHAPE),
                    "slot": (batch,)}
        for name, inp in inputs.items():
            if name in states:
                continue
            if inp.type != INPUT_DTYPES[name]:
                raise SubmissionError(f"{path.name}: input {name!r} is {inp.type}, expected {INPUT_DTYPES[name]}")
            _check_dims(name, inp.shape, expected[name])
        self.inputs = set(inputs) - states

        # state_in<suffix> -> state_out<suffix>, shape without the batch dim
        self.state_shapes: dict[str, tuple[int, ...]] = {}
        self.state_outputs: dict[str, str] = {}
        for name in sorted(states):
            inp = inputs[name]
            out_name = STATE_OUT + name[len(STATE_IN):]
            if out_name not in outputs:
                raise SubmissionError(f"{path.name}: input {name!r} needs an output named {out_name!r}")
            dims = inp.shape[1:]
            if inp.type != "tensor(float)" or len(inp.shape) < 2 or not all(isinstance(d, int) for d in dims):
                raise SubmissionError(f"{path.name}: state input {name!r} must be float32 (B, ...) with fixed "
                                      f"sizes after the batch dim, got {inp.type} {inp.shape}")
            _check_dims(name, inp.shape, (batch, *dims))
            self.state_shapes[name] = tuple(dims)
            self.state_outputs[name] = out_name

        actions = [n for n in outputs if not n.startswith(STATE_OUT)]
        if not actions:
            raise SubmissionError(f"{path.name}: no action output (every output is named {STATE_OUT}*)")
        self.output = actions[0]

    def run(self, vector: np.ndarray, graphic: np.ndarray, slot: np.ndarray,
            states: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Returns the action (batch, 2) and the next value of each state input."""
        feed = {k: v for k, v in (("vector", vector), ("graphic", graphic), ("slot", slot)) if k in self.inputs}
        feed.update(states)
        state_names = list(self.state_shapes)
        result = self.sess.run([self.output, *(self.state_outputs[n] for n in state_names)], feed)
        out = np.asarray(result[0])
        if out.shape != (self.batch, 2):
            raise ValueError(f"{self.name} returned shape {out.shape}, expected ({self.batch}, 2)")
        new_states = {}
        for name, value in zip(state_names, result[1:], strict=True):
            value = np.asarray(value, dtype=np.float32)
            if value.shape != (self.batch, *self.state_shapes[name]):
                raise ValueError(f"{self.name}: {self.state_outputs[name]} has shape {value.shape}, "
                                 f"expected {(self.batch, *self.state_shapes[name])}")
            new_states[name] = value
        return out.astype(np.float32, copy=False), new_states


class Policy:
    """
    A loaded submission. act() takes one team's rows for the current step in slot order:
    vector (N, 96) float32, graphic (N, 11, 96, 96) float32, slots (N,) int64 in 0~4,
    and returns (N, 2) float32. Frame stacking and recurrent state are handled here.
    """

    mode: str
    sessions: list[_Session]

    def __init__(self, frame_stack: int = 1):
        self.frame_stack = frame_stack
        self._frames = FrameStack(frame_stack) if frame_stack > 1 else None
        self._states: dict[int, dict[str, np.ndarray]] = {}

    @property
    def providers(self) -> list[str]:
        return self.sessions[0].sess.get_providers()

    @property
    def n_models(self) -> int:
        return len({id(s) for s in self.sessions})

    @property
    def inputs(self) -> set[str]:
        return set().union(*(s.inputs for s in self.sessions))

    @property
    def state_inputs(self) -> set[str]:
        return set().union(*(s.state_shapes for s in self.sessions))

    def reset(self) -> None:
        """Start of an episode: drop stacked frames and zero the states."""
        if self._frames is not None:
            self._frames.reset()
        self._states.clear()

    def observe(self, vector: np.ndarray, graphic: np.ndarray, slots: np.ndarray) -> None:
        """Add a step to the frame stacks without running the model (a request that was skipped)."""
        if self._frames is not None:
            self._frames.push(slots.tolist(), vector, graphic)

    def act(self, vector: np.ndarray, graphic: np.ndarray, slots: np.ndarray) -> np.ndarray:
        if self._frames is not None:
            vector, graphic = self._frames.push(slots.tolist(), vector, graphic)
        return self._run(vector, graphic, slots)

    def _run(self, vector: np.ndarray, graphic: np.ndarray, slots: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _session(self, slot: int) -> _Session:
        raise NotImplementedError

    def _state(self, slot: int) -> dict[str, np.ndarray]:
        if slot not in self._states:
            shapes = self._session(slot).state_shapes
            self._states[slot] = {n: np.zeros(s, np.float32) for n, s in shapes.items()}
        return self._states[slot]

    def warm_up(self) -> None:
        self.act(np.zeros((TEAM_SIZE, VECTOR_SIZE), np.float32),
                 np.zeros((TEAM_SIZE, *GRAPHIC_SHAPE), np.float32),
                 np.arange(TEAM_SIZE, dtype=np.int64))
        self.reset()


class TeamPolicy(Policy):
    """One model, called once per step with the whole team as a batch of 5."""

    mode = "team"

    def __init__(self, submission: Path, providers: list[str], frame_stack: int = 1):
        super().__init__(frame_stack)
        model = submission / "model.onnx"
        if not model.exists():
            raise SubmissionError("team mode needs model.onnx")
        self.sessions = [_Session(model, TEAM_SIZE, providers, frame_stack)]

    def _session(self, slot: int) -> _Session:
        return self.sessions[0]

    def _run(self, vector: np.ndarray, graphic: np.ndarray, slots: np.ndarray) -> np.ndarray:
        if len(slots) != TEAM_SIZE:
            # Fewer units than the model's batch: pad the missing slots with zeros.
            v = np.zeros((TEAM_SIZE, *vector.shape[1:]), np.float32)
            g = np.zeros((TEAM_SIZE, *graphic.shape[1:]), np.float32)
            v[slots], g[slots] = vector, graphic
            vector, graphic = v, g
        all_slots = np.arange(TEAM_SIZE, dtype=np.int64)
        states = {n: np.stack([self._state(s)[n] for s in range(TEAM_SIZE)])
                  for n in self.sessions[0].state_shapes}
        action, new_states = self.sessions[0].run(vector, graphic, all_slots, states)
        for s in slots.tolist():  # missing units keep their state
            for n, value in new_states.items():
                self._state(s)[n] = value[s]
        return action[slots]


class UnitPolicy(Policy):
    """One call per unit with batch 1; a shared model.onnx, or unit_0~4.onnx picked by slot."""

    mode = "unit"

    def __init__(self, submission: Path, providers: list[str], frame_stack: int = 1):
        super().__init__(frame_stack)
        shared = submission / "model.onnx"
        if shared.exists():
            self.sessions = [_Session(shared, 1, providers, frame_stack)] * TEAM_SIZE
            return
        per_unit = [submission / f"unit_{i}.onnx" for i in range(TEAM_SIZE)]
        missing = [p.name for p in per_unit if not p.exists()]
        if missing:
            raise SubmissionError(f"unit mode needs model.onnx or all of unit_0~4.onnx (missing {missing})")
        self.sessions = [_Session(p, 1, providers, frame_stack) for p in per_unit]

    def _session(self, slot: int) -> _Session:
        return self.sessions[slot]

    def _run(self, vector: np.ndarray, graphic: np.ndarray, slots: np.ndarray) -> np.ndarray:
        out = np.empty((len(slots), 2), np.float32)
        for row, s in enumerate(slots.tolist()):
            state = self._state(s)
            action, new_states = self.sessions[s].run(
                vector[row:row + 1], graphic[row:row + 1], slots[row:row + 1],
                {n: v[None] for n, v in state.items()})
            out[row] = action[0]
            for n, value in new_states.items():
                state[n] = value[0]
        return out


POLICY_MODES: dict[str, type[Policy]] = {"team": TeamPolicy, "unit": UnitPolicy}


def load_policy(submission: Path, providers: list[str] | None = None) -> Policy:
    meta_path = submission / "meta.json"
    if not meta_path.exists():
        raise SubmissionError('meta.json is missing (expected {"mode": "team"} or {"mode": "unit"})')
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as e:
        raise SubmissionError(f"meta.json is not valid JSON: {e}") from e
    if not isinstance(meta, dict):
        raise SubmissionError("meta.json must be a JSON object")
    mode = meta.get("mode")
    if mode not in POLICY_MODES:
        raise SubmissionError(f'meta.json "mode" must be one of {list(POLICY_MODES)}, got {mode!r}')
    frame_stack = meta.get("frame_stack", 1)
    if type(frame_stack) is not int or not 1 <= frame_stack <= MAX_FRAME_STACK:
        raise SubmissionError(f'meta.json "frame_stack" must be an integer in 1~{MAX_FRAME_STACK}, '
                              f"got {frame_stack!r}")
    return POLICY_MODES[mode](submission, providers or default_providers(), frame_stack)
