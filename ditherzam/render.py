from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .adjustments import (
    apply_contrast, apply_midtones, apply_highlights, apply_blur,
    apply_saturation, apply_invert,
)
from .dithering.pipeline import apply_dither
from .imaging import clamp_u8


@dataclass
class RenderSettings:
    contrast: float = 50
    midtones: float = 50
    highlights: float = 50
    blur: float = 50
    luminance_threshold: float = 50
    invert: bool = False
    saturation: float = 50
    style: str = "None"
    scale: int = 5
    preview_disabled: bool = False
    params: dict = field(default_factory=dict)


class RenderPipeline:
    """Compose adjustments -> dither -> color -> saturation -> effects -> invert."""

    def __init__(self, registry, color_engine=None, effect_stack=None) -> None:
        self.registry = registry
        self.color_engine = color_engine
        self.effect_stack = effect_stack

    def render(self, base_gray_f32, settings: RenderSettings,
               temporal_field=None) -> np.ndarray:
        g = np.asarray(base_gray_f32, dtype=np.float32)

        # 1-4: tonal adjustments (grayscale float32, 0..255)
        g = apply_contrast(g, settings.contrast)
        g = apply_midtones(g, settings.midtones)
        g = apply_highlights(g, settings.highlights)
        g = apply_blur(g, settings.blur)

        # 5: dither (downscale -> kernel -> upscale); temporal field forwarded
        d = apply_dither(
            g,
            style=settings.style,
            scale=settings.scale,
            luminance_threshold=settings.luminance_threshold,
            params=settings.params,
            registry=self.registry,
            preview_disabled=settings.preview_disabled,
            threshold_field=temporal_field,
        )

        # 6: color — palette map, or broadcast grayscale to RGB
        if self.color_engine is not None:
            rgb = self.color_engine.map(d).astype(np.float32)
        else:
            rgb = np.repeat(np.asarray(d, np.float32)[..., None], 3, axis=2)

        # 7: saturation (RGB float32) then clamp to uint8
        rgb = apply_saturation(rgb, settings.saturation)
        rgb_u8 = clamp_u8(rgb)

        # 8: effects stack (RGB uint8)
        if self.effect_stack is not None:
            rgb_u8 = self.effect_stack.apply(rgb_u8)

        # 9: invert LAST (on RGB)
        if settings.invert:
            rgb_u8 = clamp_u8(apply_invert(rgb_u8.astype(np.float32), True))

        return np.asarray(rgb_u8, np.uint8)
