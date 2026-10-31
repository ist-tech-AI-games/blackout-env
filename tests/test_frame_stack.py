import numpy as np

from blackout_env.competition.onnx_policy import FrameStack


def frame(value, n=1):
    return np.full((n, 3), value, np.float32), np.full((n, 2, 2, 2), value, np.float32)


def test_first_frame_fills_the_stack():
    stack = FrameStack(3)
    v, g = stack.push([0], *frame(7))
    assert v.shape == (1, 3, 3) and g.shape == (1, 3, 2, 2, 2)
    assert v[0, :, 0].tolist() == [7, 7, 7]


def test_frames_are_oldest_first():
    stack = FrameStack(3)
    for value in (1, 2, 3, 4):
        v, g = stack.push([0], *frame(value))
    assert v[0, :, 0].tolist() == [2, 3, 4]
    assert g[0, :, 0, 0, 0].tolist() == [2, 3, 4]


def test_keys_are_independent_and_reset_clears():
    stack = FrameStack(2)
    stack.push([0, 1], np.array([[1] * 3, [10] * 3], np.float32), np.zeros((2, 1), np.float32))
    v, _ = stack.push([1], np.full((1, 3), 11, np.float32), np.zeros((1, 1), np.float32))
    assert v[0, :, 0].tolist() == [10, 11]
    v, _ = stack.push([0], np.full((1, 3), 2, np.float32), np.zeros((1, 1), np.float32))
    assert v[0, :, 0].tolist() == [1, 2]
    stack.reset()
    v, _ = stack.push([0], np.full((1, 3), 5, np.float32), np.zeros((1, 1), np.float32))
    assert v[0, :, 0].tolist() == [5, 5]


def test_returned_stacks_are_copies():
    stack = FrameStack(2)
    v, _ = stack.push([0], *frame(1))
    stack.push([0], *frame(2))
    assert v[0, :, 0].tolist() == [1, 1]
