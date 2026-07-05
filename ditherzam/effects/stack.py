from __future__ import annotations

import numpy as np

from .post import EFFECTS


class EffectStack:
    """Ordered, mutable list of (effect_name, params) applied left-to-right."""

    def __init__(self) -> None:
        self.items: list[tuple[str, dict]] = []

    def add(self, name: str, **params) -> None:
        if name not in EFFECTS:
            raise KeyError(name)
        self.items.append((name, params))

    def move(self, index: int, new_index: int) -> None:
        item = self.items.pop(index)
        self.items.insert(new_index, item)

    def remove(self, index: int) -> None:
        self.items.pop(index)

    def apply(self, rgb_u8: np.ndarray) -> np.ndarray:
        out = rgb_u8
        for name, params in self.items:
            out = EFFECTS[name](out, **params)
        return out
