from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from ..color.engine import ColorEngine
from ..color.palette import Palette
from ..effects.stack import EffectStack
from ..presets import preset_to_settings, settings_to_preset


def _freeze(value):
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"unsupported normalized preset type: {type(value).__name__}")


class Look:
    def __init__(self, name: str, preset: dict):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-blank string")
        if not isinstance(preset, dict):
            raise ValueError("preset must be a dict")

        owned = copy.deepcopy(preset)
        parsed = preset_to_settings(owned)
        self.name = name
        self._preset = owned
        self._settings = parsed.settings
        self._palette = parsed.palette
        self._effects = tuple(
            (effect_name, copy.deepcopy(params))
            for effect_name, params in parsed.effects
        )
        self._smart_mask = parsed.smart_mask
        self._source_dither = parsed.source_dither
        self._source_dither_brighten = parsed.source_dither_brighten

        normalized_stack = self.build_effect_stack()
        normalized = settings_to_preset(
            self._settings,
            self._palette,
            normalized_stack,
            self._color_mode,
            self._smart_mask,
            source_dither=self.source_dither,
            source_dither_brighten=self.source_dither_brighten,
        )
        self._signature = _freeze(normalized)

    @property
    def _color_mode(self) -> str:
        color = self._preset.get("color")
        if not isinstance(color, dict):
            return "off"
        return str(color.get("mode", "off"))

    @property
    def source_dither(self) -> int:
        return self._source_dither

    @property
    def source_dither_brighten(self) -> bool:
        return self._source_dither_brighten

    @property
    def preset(self) -> dict:
        return copy.deepcopy(self._preset)

    @property
    def settings(self):
        return replace(self._settings, params=copy.deepcopy(self._settings.params))

    @property
    def smart_mask(self):
        return self._smart_mask

    @property
    def signature(self) -> tuple:
        return self._signature

    def build_effect_stack(self) -> EffectStack:
        stack = EffectStack()
        for name, params in self._effects:
            stack.add(name, **copy.deepcopy(params))
        return stack

    def build_color_engine(self, source_rgb=None):
        if self._color_mode == "off" or self._palette is None:
            return None
        colors = np.array(self._palette.colors, dtype=np.float32, copy=True, order="C")
        palette = Palette(
            name=self._palette.name,
            colors=colors,
            category=self._palette.category,
        )
        kwargs = {}
        if self._color_mode == "source":
            kwargs["source_rgb"] = source_rgb
        return ColorEngine(
            palette,
            self._color_mode,
            depth=self._settings.depth,
            mapping=self._settings.color_mapping,
            source_dither=self.source_dither,
            source_dither_brighten=self.source_dither_brighten,
            **kwargs,
        )
