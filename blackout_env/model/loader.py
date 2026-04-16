"""
PyTorch checkpoint loader.

Loads a state dict into a user-provided nn.Module and wraps it as a BaseModel.

Checkpoint format
-----------------
Both formats are supported:

  # Raw state dict
  torch.save(model.state_dict(), "model.pt")

  # Dict with a named key (e.g. from a trainer that saves extra metadata)
  torch.save({"policy_state": model.state_dict(), "step": 1000}, "model.pt")
"""

from __future__ import annotations

from typing import Any, Type

import numpy as np
import torch
import torch.nn as nn

from .base import BaseModel


class CheckpointModel(BaseModel):
    """
    Wraps a loaded nn.Module as a BaseModel.
    Batches all agents' observations together for a single forward pass.
    """

    def __init__(self, net: nn.Module, device: torch.device):
        self._net = net
        self._device = device

    def act(
        self,
        obs: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        agents = list(obs.keys())

        vectors = torch.tensor(
            np.stack([obs[a]["vector"] for a in agents]),
            dtype=torch.float32,
            device=self._device,
        )
        graphics = torch.tensor(
            np.stack([obs[a]["graphic"] for a in agents]),
            dtype=torch.float32,
            device=self._device,
        )
        # graphics from env: [B, H, W, C] → model expects [B, C, H, W]
        graphics = graphics.permute(0, 3, 1, 2)

        with torch.no_grad():
            raw: torch.Tensor = self._net(vectors, graphics)

        actions = torch.clamp(raw, -1.0, 1.0).cpu().numpy()
        return {agent: actions[i] for i, agent in enumerate(agents)}


def load_checkpoint(
    model_class: Type[nn.Module],
    checkpoint_path: str,
    state_dict_key: str | None = "policy_state",
    device: str | torch.device = "cpu",
    **model_kwargs: Any,
) -> CheckpointModel:
    """
    Instantiate a model class and load weights from a checkpoint file.

    Parameters
    ----------
    model_class : Type[nn.Module]
        The nn.Module subclass to instantiate.
    checkpoint_path : str
        Path to the .pt checkpoint file.
    state_dict_key : str | None
        Key to extract the state dict from the checkpoint dict.
        Pass None if the checkpoint file is a raw state dict.
        Defaults to "policy_state".
    device : str | torch.device
        Device to load the model onto.
    **model_kwargs
        Keyword arguments forwarded to model_class().

    Returns
    -------
    CheckpointModel
        A BaseModel-compatible wrapper ready for inference.

    Examples
    --------
        # Checkpoint saved as raw state dict
        model = load_checkpoint(MyPolicy, "model.pt", state_dict_key=None, hidden=256)

        # Checkpoint saved as {"policy_state": state_dict, ...}
        model = load_checkpoint(MyPolicy, "model.pt", n_items=1, n_classes=3)
    """
    device = torch.device(device)
    net = model_class(**model_kwargs)

    raw = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if state_dict_key is not None:
        if not isinstance(raw, dict) or state_dict_key not in raw:
            raise KeyError(
                f"Key '{state_dict_key}' not found in checkpoint. "
                f"Available keys: {list(raw.keys()) if isinstance(raw, dict) else 'N/A'}. "
                f"Pass state_dict_key=None if the file is a raw state dict."
            )
        state_dict = raw[state_dict_key]
    else:
        state_dict = raw

    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    return CheckpointModel(net, device)
