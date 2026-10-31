"""
onnx_policy is loaded as a plain file (as in the policy container and check_submission),
so these tests run in an export environment with onnx + onnxruntime and no mlagents.
"""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx", exc_type=ImportError)  # skipped in the mlagents env (protobuf pin)
pytest.importorskip("onnxruntime", exc_type=ImportError)
from onnx import TensorProto, helper  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "onnx_policy", Path(__file__).parent.parent / "blackout_env" / "competition" / "onnx_policy.py")
op = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(op)

CPU = ["CPUExecutionProvider"]


def _save(path: Path, nodes, inputs, out_dim: int, initializers=()):
    graph = helper.make_graph(nodes, "g", inputs,
                              [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["B", out_dim])],
                              initializer=list(initializers))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def _int64(name, values):
    return helper.make_tensor(name, TensorProto.INT64, [len(values)], values)


def vector_model(path: Path, batch, add: float = 0.0, width: int = 2, extra_inputs=("graphic",),
                 vector_name: str = "vector"):
    """action = vector[:, :width] + add."""
    inputs = [helper.make_tensor_value_info(vector_name, TensorProto.FLOAT, [batch, 96])]
    if "graphic" in extra_inputs:
        inputs.append(helper.make_tensor_value_info("graphic", TensorProto.FLOAT, [batch, 11, 96, 96]))
    nodes = [helper.make_node("Slice", [vector_name, "starts", "ends", "axes"], ["head"]),
             helper.make_node("Add", ["head", "c"], ["action"])]
    inits = [_int64("starts", [0]), _int64("ends", [width]), _int64("axes", [1]),
             helper.make_tensor("c", TensorProto.FLOAT, [], [add])]
    _save(path, nodes, inputs, width, inits)


def slot_model(path: Path, batch):
    """action = [slot, slot] as float."""
    inputs = [helper.make_tensor_value_info("slot", TensorProto.INT64, [batch])]
    nodes = [helper.make_node("Cast", ["slot"], ["f"], to=TensorProto.FLOAT),
             helper.make_node("Unsqueeze", ["f", "ax"], ["col"]),
             helper.make_node("Concat", ["col", "col"], ["action"], axis=1)]
    _save(path, nodes, inputs, 2, [_int64("ax", [1])])


def stacked_model(path: Path, batch, k):
    """vector (B, k, 96) -> action = (vector[:, 0, 0], vector[:, k-1, 0]): oldest and newest frame."""
    inputs = [helper.make_tensor_value_info("vector", TensorProto.FLOAT, [batch, k, 96]),
              helper.make_tensor_value_info("graphic", TensorProto.FLOAT, [batch, k, 11, 96, 96])]
    nodes = [helper.make_node("Slice", ["vector", "starts", "ends", "axes"], ["first"]),
             helper.make_node("Squeeze", ["first", "axes"], ["frames"]),
             helper.make_node("Gather", ["frames", "idx"], ["action"], axis=1)]
    inits = [_int64("starts", [0]), _int64("ends", [1]), _int64("axes", [2]), _int64("idx", [0, k - 1])]
    _save(path, nodes, inputs, 2, inits)


def counter_model(path: Path, batch, with_output=True):
    """state_in (B, 1) counts calls: action = [state, state], state_out = state + 1."""
    inputs = [helper.make_tensor_value_info("vector", TensorProto.FLOAT, [batch, 96]),
              helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [batch, 1])]
    nodes = [helper.make_node("Concat", ["state_in", "state_in"], ["action"], axis=1),
             helper.make_node("Add", ["state_in", "one"], ["state_out"])]
    outputs = [helper.make_tensor_value_info("action", TensorProto.FLOAT, [batch, 2])]
    if with_output:
        outputs.append(helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [batch, 1]))
    else:
        nodes.pop()
    graph = helper.make_graph(nodes, "g", inputs, outputs,
                              initializer=[helper.make_tensor("one", TensorProto.FLOAT, [], [1.0])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def submission(tmp_path: Path, mode, **meta) -> Path:
    (tmp_path / "meta.json").write_text(json.dumps({"mode": mode, **meta}))
    return tmp_path


def team_inputs(n=5):
    vector = np.zeros((n, 96), np.float32)
    vector[:, 0] = np.arange(n)
    vector[:, 1] = 10 + np.arange(n)
    return vector, np.zeros((n, 11, 96, 96), np.float32)


def test_team_mode_runs_whole_team(tmp_path):
    sub = submission(tmp_path, "team")
    vector_model(sub / "model.onnx", 5)
    policy = op.load_policy(sub, CPU)
    assert isinstance(policy, op.TeamPolicy) and policy.inputs == {"vector", "graphic"}
    vector, graphic = team_inputs()
    out = policy.act(vector, graphic, np.arange(5))
    np.testing.assert_array_equal(out, vector[:, :2])


def test_team_mode_pads_missing_units(tmp_path):
    sub = submission(tmp_path, "team")
    vector_model(sub / "model.onnx", 5)
    policy = op.load_policy(sub, CPU)
    vector, graphic = team_inputs(3)
    out = policy.act(vector, graphic, np.array([0, 2, 4]))
    np.testing.assert_array_equal(out, vector[:, :2])


def test_unit_mode_shared_model_gets_each_slot(tmp_path):
    sub = submission(tmp_path, "unit")
    slot_model(sub / "model.onnx", 1)
    policy = op.load_policy(sub, CPU)
    assert isinstance(policy, op.UnitPolicy) and policy.n_models == 1
    vector, graphic = team_inputs(2)
    out = policy.act(vector, graphic, np.array([3, 1]))
    np.testing.assert_array_equal(out, [[3, 3], [1, 1]])


def test_unit_mode_picks_model_file_by_slot(tmp_path):
    sub = submission(tmp_path, "unit")
    for i in range(5):
        vector_model(sub / f"unit_{i}.onnx", 1, add=100.0 * i, extra_inputs=())
    policy = op.load_policy(sub, CPU)
    assert policy.n_models == 5
    vector, graphic = team_inputs(2)
    out = policy.act(vector, graphic, np.array([4, 2]))
    np.testing.assert_array_equal(out, vector[:, :2] + [[400], [200]])


def test_warm_up_runs(tmp_path):
    sub = submission(tmp_path, "unit")
    slot_model(sub / "model.onnx", "B")  # symbolic batch is accepted
    op.load_policy(sub, CPU).warm_up()


@pytest.mark.parametrize("setup, message", [
    (lambda d: vector_model(d / "model.onnx", 5), "meta.json is missing"),
    (lambda d: (submission(d, "both"), vector_model(d / "model.onnx", 5)), "must be one of"),
    (lambda d: submission(d, "team"), "team mode needs model.onnx"),
    (lambda d: (submission(d, "team"), vector_model(d / "model.onnx", 1)), "expected (5, 96)"),
    (lambda d: (submission(d, "team"), vector_model(d / "model.onnx", 5, vector_name="obs")), "inputs must be named"),
    (lambda d: (submission(d, "unit"), vector_model(d / "unit_0.onnx", 1)), "missing ['unit_1.onnx'"),
    (lambda d: (submission(d, "team", frame_stack=3), vector_model(d / "model.onnx", 5)), "expected shape (5, 3, 96)"),
    (lambda d: (submission(d, "team"), stacked_model(d / "model.onnx", 5, 3)), "expected shape (5, 96)"),
    (lambda d: (submission(d, "team", frame_stack=0), vector_model(d / "model.onnx", 5)), '"frame_stack" must be'),
    (lambda d: (submission(d, "team", frame_stack=17), vector_model(d / "model.onnx", 5)), '"frame_stack" must be'),
    (lambda d: (submission(d, "team", frame_stack="4"), vector_model(d / "model.onnx", 5)), '"frame_stack" must be'),
    (lambda d: (submission(d, "team"), counter_model(d / "model.onnx", 5, with_output=False)),
     "needs an output named 'state_out'"),
])
def test_bad_submissions_are_rejected(tmp_path, setup, message):
    setup(tmp_path)
    with pytest.raises(op.SubmissionError, match=re.escape(message)):
        op.load_policy(tmp_path, CPU)


def test_wrong_output_shape_fails_at_run_time(tmp_path):
    sub = submission(tmp_path, "team")
    vector_model(sub / "model.onnx", 5, width=3)
    policy = op.load_policy(sub, CPU)
    with pytest.raises(ValueError, match="expected"):
        policy.act(*team_inputs(), np.arange(5))


def step(policy, values, slots=None):
    """One step where unit row i has vector[:, 0] = values[i]."""
    slots = np.arange(len(values)) if slots is None else np.asarray(slots)
    vector = np.zeros((len(values), 96), np.float32)
    vector[:, 0] = values
    return policy.act(vector, np.zeros((len(values), 11, 96, 96), np.float32), slots)


def test_frame_stack_feeds_oldest_to_newest(tmp_path):
    sub = submission(tmp_path, "team", frame_stack=3)
    stacked_model(sub / "model.onnx", 5, 3)
    policy = op.load_policy(sub, CPU)
    policy.warm_up()
    assert policy.frame_stack == 3
    np.testing.assert_array_equal(step(policy, [1] * 5)[0], [1, 1])  # first frame fills the stack
    step(policy, [2] * 5)
    np.testing.assert_array_equal(step(policy, [3] * 5)[0], [1, 3])
    np.testing.assert_array_equal(step(policy, [4] * 5)[0], [2, 4])
    policy.reset()
    np.testing.assert_array_equal(step(policy, [9] * 5)[0], [9, 9])


def test_observe_adds_frames_without_running(tmp_path):
    sub = submission(tmp_path, "unit", frame_stack=2)
    stacked_model(sub / "model.onnx", 1, 2)
    policy = op.load_policy(sub, CPU)
    step(policy, [1, 1])
    vector = np.full((2, 96), 5, np.float32)
    policy.observe(vector, np.zeros((2, 11, 96, 96), np.float32), np.arange(2))
    np.testing.assert_array_equal(step(policy, [6, 6]), [[5, 6], [5, 6]])


def test_team_state_is_kept_per_slot(tmp_path):
    sub = submission(tmp_path, "team")
    counter_model(sub / "model.onnx", 5)
    policy = op.load_policy(sub, CPU)
    policy.warm_up()  # warm-up must not leave state behind
    assert policy.state_inputs == {"state_in"} and policy.inputs == {"vector"}
    assert step(policy, [0] * 5)[:, 0].tolist() == [0] * 5
    assert step(policy, [0] * 5)[:, 0].tolist() == [1] * 5
    assert step(policy, [0, 0], slots=[1, 3])[:, 0].tolist() == [2, 2]  # missing units keep their state
    assert step(policy, [0] * 5)[:, 0].tolist() == [2, 3, 2, 3, 2]
    policy.reset()
    assert step(policy, [0] * 5)[:, 0].tolist() == [0] * 5


def test_unit_state_is_kept_per_slot(tmp_path):
    sub = submission(tmp_path, "unit")
    counter_model(sub / "model.onnx", 1)
    policy = op.load_policy(sub, CPU)
    step(policy, [0, 0], slots=[0, 4])
    step(policy, [0], slots=[4])
    assert step(policy, [0, 0], slots=[0, 4])[:, 0].tolist() == [1, 2]
