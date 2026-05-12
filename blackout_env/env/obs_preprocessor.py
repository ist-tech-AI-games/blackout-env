"""
Observation preprocessor for BlackOut ML-Agents environment.

Raw obs from Unity:
  vector: float32[45]
    [0~39]  unit blocks × 10  (4 floats each: pos_x, pos_y, team_sign, holding_item_id)
    [40]    self class_id      (float-cast int)
    [41]    own_score          (own_score / target_score)
    [42]    opp_score          (opp_score / target_score)
    [43]    time_left          (1 - timer.Ratio)
    [44]    unit_index         (0~9, used for agent_id mapping, stripped from output)
  graphic: float32[H × W × 1]  grayscale semantic ID map (normalized [0,1] by ML-Agents)

Preprocessed obs:
  vector: float32[20 + 10 + 10*(N_ITEMS+1) + N_CLASSES + 3]
  graphic: float32[H × W × (ITEM_ID_OFFSET + N_ITEMS)]  binary channel masks
"""

import json
import numpy as np
from pathlib import Path


def load_semantic_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


class ObsPreprocessor:
    """
    Stateless preprocessor. Construct once, call preprocess_vector / preprocess_graphic per step.

    Parameters
    ----------
    semantic_config : dict
        Loaded from semantic_map_config.json.
    n_items : int
        Number of distinct item types (KnownItems array length).
    n_classes : int
        Number of unit class types (KnownClasses array length).
    """

    N_UNITS = 10
    UNIT_BLOCK_SIZE = 4          # pos_x, pos_y, team_sign, holding_item_id
    SCALAR_OBS_COUNT = 3         # own_score, opp_score, time_left

    # Raw vector layout constants (derived from above)
    # [0 ~ N_UNITS*UNIT_BLOCK_SIZE-1] unit blocks
    # [RAW_CLASS_SLOT]               self class_id
    # [RAW_SCALAR_START ~ RAW_SCALAR_START+SCALAR_OBS_COUNT-1] own_score, opp_score, time_left
    # [RAW_UNIT_INDEX_SLOT]          unit_index (routing only, stripped from output)
    RAW_CLASS_SLOT      = N_UNITS * UNIT_BLOCK_SIZE               # 40
    RAW_SCALAR_START    = RAW_CLASS_SLOT + 1                      # 41
    RAW_UNIT_INDEX_SLOT = RAW_SCALAR_START + SCALAR_OBS_COUNT    # 44
    RAW_VECTOR_SIZE     = RAW_UNIT_INDEX_SLOT + 1                 # 45

    def __init__(self, semantic_config: dict, n_items: int = 1, n_classes: int = 3):
        self.n_items = n_items
        self.n_classes = n_classes
        self.item_id_offset: int = semantic_config["item_id_offset"]
        self.ids: dict[str, int] = semantic_config["ids"]

        # Number of semantic channels = base IDs + one channel per item
        self.n_graphic_channels: int = self.item_id_offset + n_items

        # Precompute processed vector length
        pos_floats = self.N_UNITS * 2                         # pos_x, pos_y × 10
        team_floats = self.N_UNITS                            # team_sign × 10
        item_oh_floats = self.N_UNITS * (n_items + 1)        # one-hot holding_item × 10
        class_oh_floats = n_classes                           # one-hot class_id
        scalar_floats = self.SCALAR_OBS_COUNT
        self.vector_obs_size: int = (
            pos_floats + team_floats + item_oh_floats + class_oh_floats + scalar_floats
        )

        # Pre-allocated channel index array for vectorized graphic separation
        self._channel_ids = np.arange(self.n_graphic_channels, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Vector preprocessing
    # ------------------------------------------------------------------

    def preprocess_vector(self, raw: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        raw : float32[45]

        Returns
        -------
        float32[vector_obs_size]
        """
        assert raw.shape == (self.RAW_VECTOR_SIZE,), f"Expected float32[{self.RAW_VECTOR_SIZE}], got {raw.shape}"

        parts: list[np.ndarray] = []

        for i in range(self.N_UNITS):
            base = i * self.UNIT_BLOCK_SIZE
            pos_x = raw[base]
            pos_y = raw[base + 1]
            team_sign = raw[base + 2]
            item_id = int(round(raw[base + 3]))   # 0 = no item, 1~N = item index

            parts.append(np.array([pos_x, pos_y, team_sign], dtype=np.float32))
            parts.append(self._one_hot(item_id, self.n_items + 1))

        class_id = int(round(raw[self.RAW_CLASS_SLOT]))
        parts.append(self._one_hot(class_id, self.n_classes))

        parts.append(raw[self.RAW_SCALAR_START : self.RAW_UNIT_INDEX_SLOT])  # own_score, opp_score, time_left

        return np.concatenate(parts, dtype=np.float32)

    # ------------------------------------------------------------------
    # Visual preprocessing
    # ------------------------------------------------------------------

    def preprocess_graphic(self, raw: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        raw : float32[H × W × 1]  grayscale semantic ID map from RenderTextureSensor.
              Values are normalized to [0, 1] by ML-Agents (id / 255.0).

        Returns
        -------
        float32[H × W × n_graphic_channels]  binary channel masks
          ch 0             : empty
          ch 1             : wall
          ch 2             : ally_storage
          ch 3             : enemy_storage
          ch 4             : ally_unit
          ch 5             : enemy_unit
          ch 6 ~ 6+N_ITEMS : item type i  (id = item_id_offset + i)
        """
        assert raw.ndim == 3 and raw.shape[2] in (1, 3), (
            f"Expected float32[H×W×1 or H×W×3], got {raw.shape}"
        )
        # ML-Agents normalizes pixel values to [0, 1]; recover integer IDs.
        # RGB24 source has R=G=B=id, so channel 0 is sufficient.
        id_map = np.round(raw[:, :, 0] * 255).astype(np.uint8)  # H × W

        # Vectorized channel separation via broadcast comparison: (H, W, 1) == (C,)
        return (id_map[:, :, np.newaxis] == self._channel_ids).astype(np.float32)

    # ------------------------------------------------------------------
    # Team perspective flip
    # ------------------------------------------------------------------

    def flip_team_perspective(self, graphic: np.ndarray) -> np.ndarray:
        """
        Convert a TeamA graphic to a TeamB graphic by swapping ally/enemy channels.

        Parameters
        ----------
        graphic : float32[H × W × n_graphic_channels]  TeamA perspective

        Returns
        -------
        float32[H × W × n_graphic_channels]  TeamB perspective
        """
        result = graphic.copy()
        ally_s  = self.ids["ally_storage"]    # 2
        enemy_s = self.ids["enemy_storage"]   # 3
        ally_u  = self.ids["ally_unit"]       # 4
        enemy_u = self.ids["enemy_unit"]      # 5
        result[..., ally_s]  = graphic[..., enemy_s]
        result[..., enemy_s] = graphic[..., ally_s]
        result[..., ally_u]  = graphic[..., enemy_u]
        result[..., enemy_u] = graphic[..., ally_u]
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _one_hot(idx: int, size: int) -> np.ndarray:
        vec = np.zeros(size, dtype=np.float32)
        if 0 <= idx < size:
            vec[idx] = 1.0
        return vec
