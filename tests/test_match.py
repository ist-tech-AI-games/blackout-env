import pytest

from blackout_env.competition.match import run_match, run_series

from .fakes import ConstModel, FakeEnv

MOVER = ConstModel(1.0)   # positive x: its side scores every step
IDLE = ConstModel(-1.0)


def test_winner_is_decided_by_score_not_reward():
    env = FakeEnv(steps=4)
    result = run_match(env, MOVER, IDLE)
    assert result.winner == 0
    assert result.model_a_score == pytest.approx(0.2)   # 5 units x 4 steps x 0.01
    assert result.model_b_score == 0.0
    assert result.team_a_total_reward == result.team_b_total_reward  # rewards are equal, winner is not


def test_swap_puts_model_a_on_team_b():
    env = FakeEnv(steps=4)
    result = run_match(env, MOVER, IDLE, swap_teams=True)
    assert result.winner == 0
    assert result.model_a_score == pytest.approx(0.2)
    assert result.team_scores == (0.0, pytest.approx(0.2))  # by side: Team A, Team B
    last = env.received[-1]
    assert last["unit_5"][0] == 1.0 and last["unit_0"][0] == -1.0


def test_equal_scores_are_a_draw():
    result = run_match(FakeEnv(steps=3), IDLE, IDLE)
    assert result.winner is None and result.is_draw


def test_on_step_sees_every_step_actions():
    env = FakeEnv(steps=3)
    seen = []
    run_match(env, MOVER, IDLE, seed=7, on_step=seen.append)
    assert len(seen) == 3
    assert set(seen[0]) == set(env.possible_agents)
    assert env.reset_seeds == [7]


def test_series_plays_each_seed_from_both_sides():
    env = FakeEnv(steps=2)
    series = run_series(env, MOVER, IDLE, n_matches=4, seeds=[11, 22])
    assert env.reset_seeds == [11, 11, 22, 22]
    assert [m.swap_teams for m in series.matches] == [False, True, False, True]
    assert series.model_a_wins == 4 and series.series_winner == 0
