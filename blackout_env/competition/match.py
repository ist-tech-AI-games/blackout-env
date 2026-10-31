"""
1v1 match runner for BlackOut competition.

Team A (unit_0~4) uses model_a; Team B (unit_5~9) uses model_b.
Supports single match and best-of-N series with team swapping for fairness.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..env.blackout_env import BlackOutEnv
from ..env.constants import team_a_agents, team_b_agents
from ..model.base import BaseModel


@dataclass
class MatchResult:
    """Result of a single episode."""

    winner: int | None          # 0 = model_a, 1 = model_b, None = draw
    team_a_total_reward: float  # model_a's summed reward (training signal, not used for the winner)
    team_b_total_reward: float  # model_b's summed reward
    episode_steps: int
    model_a_score: float = 0.0  # final game score of model_a's team (score / target_score)
    model_b_score: float = 0.0
    seed: int | None = None
    swap_teams: bool = False    # True = model_a played Team B

    @property
    def is_draw(self) -> bool:
        return self.winner is None

    @property
    def team_scores(self) -> tuple[float, float]:
        """Final scores by side: (Team A = unit_0~4, Team B = unit_5~9)."""
        if self.swap_teams:
            return self.model_b_score, self.model_a_score
        return self.model_a_score, self.model_b_score


@dataclass
class SeriesResult:
    """Result of a best-of-N series."""

    matches: list[MatchResult] = field(default_factory=list)

    @property
    def model_a_wins(self) -> int:
        return sum(1 for m in self.matches if m.winner == 0)

    @property
    def model_b_wins(self) -> int:
        return sum(1 for m in self.matches if m.winner == 1)

    @property
    def draws(self) -> int:
        return sum(1 for m in self.matches if m.is_draw)

    @property
    def series_winner(self) -> int | None:
        """0 = model_a wins series, 1 = model_b wins series, None = draw."""
        if self.model_a_wins > self.model_b_wins:
            return 0
        if self.model_b_wins > self.model_a_wins:
            return 1
        return None


def run_match(
    env: BlackOutEnv,
    model_a: BaseModel,
    model_b: BaseModel,
    swap_teams: bool = False,
    seed: int | None = None,
    on_step: Callable[[dict[str, np.ndarray]], None] | None = None,
) -> MatchResult:
    """
    Run a single episode between two models.

    Parameters
    ----------
    env : BlackOutEnv
        A fully constructed environment (not yet reset).
    model_a : BaseModel
        Model controlling Team A (unit_0~4) by default.
    model_b : BaseModel
        Model controlling Team B (unit_5~9) by default.
    swap_teams : bool
        If True, model_a controls Team B and model_b controls Team A.
        Useful for fairness in a series — swaps starting sides between matches.
    seed : int | None
        Map/item seed passed to env.reset(). None = random.
    on_step : callable | None
        Called with each step's actions before they are sent (e.g. replay.ReplayLog.add).

    Returns
    -------
    MatchResult
        Winner is always reported from model_a / model_b perspective,
        regardless of which team they were assigned. It is decided by the final
        game score, not by the (shaped) training reward.
    """
    team_a_names = set(team_a_agents())
    team_b_names = set(team_b_agents())

    # Assign models to teams
    if not swap_teams:
        controller = {name: model_a for name in team_a_names}
        controller.update({name: model_b for name in team_b_names})
    else:
        controller = {name: model_b for name in team_a_names}
        controller.update({name: model_a for name in team_b_names})

    model_a.reset()
    model_b.reset()
    obs, _ = env.reset(seed=seed)

    total_rewards: dict[str, float] = {a: 0.0 for a in env.possible_agents}
    steps = 0
    last_info: dict = {}

    while env.agents:
        # Split obs by controller
        model_a_obs = {a: obs[a] for a in env.agents if controller[a] is model_a and a in obs}
        model_b_obs = {a: obs[a] for a in env.agents if controller[a] is model_b and a in obs}

        actions: dict[str, np.ndarray] = {}
        if model_a_obs:
            actions.update(model_a.act(model_a_obs))
        if model_b_obs:
            actions.update(model_b.act(model_b_obs))
        if on_step is not None:
            on_step(actions)

        obs, rewards, terminations, _, infos = env.step(actions)
        if infos:
            last_info = next(iter(infos.values()))

        for agent, reward in rewards.items():
            total_rewards[agent] += reward
        steps += 1

    # Aggregate rewards per team (physical team in env)
    team_a_reward = sum(total_rewards[a] for a in team_a_names)
    team_b_reward = sum(total_rewards[a] for a in team_b_names)

    # Map physical team reward back to model_a / model_b
    if not swap_teams:
        model_a_reward = team_a_reward
        model_b_reward = team_b_reward
    else:
        model_a_reward = team_b_reward
        model_b_reward = team_a_reward

    # Final game score per physical team (normalized by target_score)
    score_team_a = float(last_info.get("score_0", 0.0))
    score_team_b = float(last_info.get("score_1", 0.0))
    if not swap_teams:
        model_a_score, model_b_score = score_team_a, score_team_b
    else:
        model_a_score, model_b_score = score_team_b, score_team_a

    # Winner by game score: first to target_score, or higher score when time runs out
    if model_a_score > model_b_score:
        winner = 0
    elif model_b_score > model_a_score:
        winner = 1
    else:
        winner = None

    return MatchResult(
        winner=winner,
        team_a_total_reward=model_a_reward,
        team_b_total_reward=model_b_reward,
        episode_steps=steps,
        model_a_score=model_a_score,
        model_b_score=model_b_score,
        seed=seed,
        swap_teams=swap_teams,
    )


def run_series(
    env: BlackOutEnv,
    model_a: BaseModel,
    model_b: BaseModel,
    n_matches: int = 10,
    seeds: list[int] | None = None,
) -> SeriesResult:
    """
    Run multiple matches, swapping team sides every match for fairness.

    Parameters
    ----------
    env : BlackOutEnv
    model_a : BaseModel
    model_b : BaseModel
    n_matches : int
        Total number of matches to play.
    seeds : list[int] | None
        One seed per pair of matches. Matches 2k and 2k+1 use seeds[k] with sides
        swapped, so both models play the same map from both sides. None = random maps.

    Returns
    -------
    SeriesResult
    """
    result = SeriesResult()
    for i in range(n_matches):
        swap = (i % 2 == 1)
        seed = seeds[i // 2] if seeds is not None else None
        match_result = run_match(env, model_a, model_b, swap_teams=swap, seed=seed)
        result.matches.append(match_result)
    return result
