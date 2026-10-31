"""Stand-ins for the Unity env and models, so match logic can be tested without a build."""

from __future__ import annotations

import numpy as np

from blackout_env.env.constants import all_agents
from blackout_env.model.base import BaseModel


class FakeEnv:
    """
    Runs for `steps` steps. Each step adds 0.01 to a side's score for every unit of that
    side whose action x is positive, so the score shows which actions reached which side.
    """

    def __init__(self, steps: int = 5):
        self.possible_agents = all_agents()
        self.agents: list[str] = []
        self.steps = steps
        self.reset_seeds: list[int | None] = []
        self.received: list[dict[str, np.ndarray]] = []

    def _obs(self) -> dict:
        return {a: {"vector": np.zeros(96, np.float32), "graphic": np.zeros((96, 96, 11), np.float32)}
                for a in self.agents}

    def reset(self, seed=None, options=None):
        self.reset_seeds.append(seed)
        self.agents = list(self.possible_agents)
        self.t = 0
        self.scores = [0.0, 0.0]
        return self._obs(), {}

    def step(self, actions):
        self.received.append({a: np.asarray(v).copy() for a, v in actions.items()})
        for a, v in actions.items():
            if v[0] > 0:
                self.scores[0 if int(a.split("_")[1]) < 5 else 1] += 0.01
        self.t += 1
        done = self.t >= self.steps
        info = {"score_0": self.scores[0], "score_1": self.scores[1]}
        obs = self._obs()
        rewards = {a: 1.0 for a in self.agents}
        terms = {a: done for a in self.agents}
        infos = {a: dict(info) for a in self.agents}
        if done:
            self.agents = []
        return obs, rewards, terms, {a: False for a in terms}, infos


class ConstModel(BaseModel):
    """Every agent gets the same action."""

    def __init__(self, x: float, y: float = 0.0):
        self.action = np.array([x, y], np.float32)

    def act(self, obs):
        return {a: self.action.copy() for a in obs}
