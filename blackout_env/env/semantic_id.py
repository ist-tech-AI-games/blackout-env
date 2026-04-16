"""
Semantic ID definitions for the BlackOut graphic observation (semantic map).

The graphic obs from Unity is a grayscale RenderTexture where each pixel holds
an integer ID. ObsPreprocessor converts this into per-channel binary masks.
SemanticId maps those channel indices to human-readable names.

Channel index == semantic ID value == position in the float32[H × W × C] array.

Usage
-----
    from training.env.semantic_id import SemanticId

    # Index into the preprocessed graphic obs
    ally_channel = graphic[:, :, SemanticId.ALLY_UNIT]

    # Check whether a custom item ID is within the known range
    if SemanticId.is_item(some_id, n_items=2):
        item_index = SemanticId.item_index(some_id)

    # Visualize: assign a color per channel
    COLOR_MAP = {
        SemanticId.EMPTY:         (0,   0,   0),
        SemanticId.WALL:          (128, 128, 128),
        SemanticId.ALLY_STORAGE:  (0,   0,   255),
        SemanticId.ENEMY_STORAGE: (255, 0,   0),
        SemanticId.ALLY_UNIT:     (0,   128, 255),
        SemanticId.ENEMY_UNIT:    (255, 64,  0),
    }
"""

from __future__ import annotations


class SemanticId:
    """
    Channel index constants for the preprocessed graphic observation.

    These values are fixed by semantic_map_config.json (item_id_offset = 6).
    Item channels start at ITEM_ID_OFFSET and are indexed sequentially by
    KnownItems array order (same as BlackOutEpisodeCoordinator.KnownItems).
    """

    EMPTY:         int = 0
    WALL:          int = 1
    ALLY_STORAGE:  int = 2
    ENEMY_STORAGE: int = 3
    ALLY_UNIT:     int = 4
    ENEMY_UNIT:    int = 5
    ITEM_ID_OFFSET: int = 6

    # Convenience tuple of all base (non-item) channel indices
    BASE_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

    @classmethod
    def item_channel(cls, item_index: int) -> int:
        """
        Return the channel index for an item type.

        Parameters
        ----------
        item_index : int
            Zero-based index into KnownItems array (same order as coordinator).
        """
        if item_index < 0:
            raise ValueError(f"item_index must be >= 0, got {item_index}")
        return cls.ITEM_ID_OFFSET + item_index

    @classmethod
    def item_index(cls, channel: int) -> int:
        """
        Return the zero-based KnownItems index for an item channel.
        Raises ValueError if the channel is not an item channel.
        """
        if channel < cls.ITEM_ID_OFFSET:
            raise ValueError(f"Channel {channel} is not an item channel (offset={cls.ITEM_ID_OFFSET})")
        return channel - cls.ITEM_ID_OFFSET

    @classmethod
    def is_item(cls, channel: int, n_items: int) -> bool:
        """Return True if the channel index corresponds to a known item type."""
        return cls.ITEM_ID_OFFSET <= channel < cls.ITEM_ID_OFFSET + n_items

    @classmethod
    def all_channels(cls, n_items: int) -> list[int]:
        """Return all channel indices in order for a given number of item types."""
        return list(range(cls.ITEM_ID_OFFSET + n_items))

    @classmethod
    def name(cls, channel: int, n_items: int = 0) -> str:
        """Human-readable name for a channel index. Useful for logging and visualization."""
        names = {
            cls.EMPTY:         "empty",
            cls.WALL:          "wall",
            cls.ALLY_STORAGE:  "ally_storage",
            cls.ENEMY_STORAGE: "enemy_storage",
            cls.ALLY_UNIT:     "ally_unit",
            cls.ENEMY_UNIT:    "enemy_unit",
        }
        if channel in names:
            return names[channel]
        if cls.is_item(channel, n_items):
            return f"item_{cls.item_index(channel)}"
        return f"unknown_{channel}"
