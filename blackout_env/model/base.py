"""
Base interface that every competition model must implement.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """
    Stateless inference interface for a team of 5 agents.

    Implement this class and pass instances to run_match().

    Example
    -------
        class MyPolicy(nn.Module):
            ...

        class MyModel(BaseModel):
            def __init__(self, checkpoint_path: str):
                self._net = load_checkpoint(MyPolicy, checkpoint_path)

            def act(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
                actions = {}
                for agent, agent_obs in obs.items():
                    vec = torch.tensor(agent_obs["vector"]).unsqueeze(0)
                    graphic = torch.tensor(agent_obs["graphic"]).unsqueeze(0)
                    with torch.no_grad():
                        action = self._net(vec, graphic).squeeze(0).numpy()
                    actions[agent] = action
                return actions
    """

    @abstractmethod
    def act(
        self,
        obs: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """
        Decide actions for a team's agents.

        Parameters
        ----------
        obs : dict[agent_name, {"vector": float32[N], "graphic": float32[H, W, C]}]
            Preprocessed observations for this model's agents only (5 agents).

        Returns
        -------
        dict[agent_name, float32[2]]
            Continuous actions (dx, dy) in [-1, 1] per agent.
        """
