import numpy as np
from blackout_env import BaseModel


class RandomPolicy(BaseModel):
    """Policy that samples uniformly random actions for each agent."""

    def act(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {agent: np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
                for agent in obs}
