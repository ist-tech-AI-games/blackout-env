import numpy as np

from blackout_env.competition.match import run_match
from blackout_env.competition.replay import ReplayLog, play

from .fakes import ConstModel, FakeEnv


def test_replay_feeds_the_same_actions_back(tmp_path):
    env = FakeEnv(steps=6)
    log = ReplayLog()
    result = run_match(env, ConstModel(0.5, -0.25), ConstModel(-1.0, 1.0), swap_teams=True, seed=3,
                       on_step=log.add)
    score_0, score_1 = result.team_scores
    path = tmp_path / "m1.npz"
    log.save(path, 3, {"final": {"score_0": score_0, "score_1": score_1}})

    original = env.received
    replay_env = FakeEnv(steps=6)
    r = play(replay_env, path)

    assert replay_env.reset_seeds == [3]
    assert len(replay_env.received) == len(original)
    for a, b in zip(original, replay_env.received, strict=True):
        assert a.keys() == b.keys()
        for agent in a:
            np.testing.assert_array_equal(a[agent], b[agent])
    assert r["replayed"]["score_0"] == r["logged"]["score_0"]
    assert r["replayed"]["score_1"] == r["logged"]["score_1"]


def test_unseeded_match_replays_unseeded(tmp_path):
    log = ReplayLog()
    run_match(FakeEnv(steps=2), ConstModel(1.0), ConstModel(1.0), on_step=log.add)
    log.save(tmp_path / "m.npz", None, {})
    env = FakeEnv(steps=2)
    play(env, tmp_path / "m.npz")
    assert env.reset_seeds == [None]
