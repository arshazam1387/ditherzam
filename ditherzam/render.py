from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from .adjustments import (
    apply_contrast, apply_midtones, apply_highlights, apply_blur,
    apply_saturation, apply_invert,
)
from .dithering.pipeline import apply_dither
from .imaging import clamp_u8


def _params_sig(params: dict):
    # repr() keeps this hashable/comparable regardless of value types (numbers,
    # tuples, lists) while staying stable for identical content.
    return tuple(sorted((k, repr(v)) for k, v in (params or {}).items()))


def _color_sig(engine):
    if engine is None:
        return (None,)
    return (engine.mode, engine.palette.name, engine.palette.colors.tobytes())


def _effect_sig(stack):
    if stack is None:
        return None
    return tuple((name, tuple(sorted((k, repr(v)) for k, v in params.items())))
                 for name, params in stack.items)


@dataclass
class RenderSettings:
    contrast: float = 50
    midtones: float = 50
    highlights: float = 50
    blur: float = 0  # blur's identity is 0 (value=50 == 25px Gaussian blur)
    luminance_threshold: float = 50
    invert: bool = False
    saturation: float = 50
    style: str = "None"
    scale: int = 5
    depth: int = 2
    color_mapping: str = "match"
    preview_disabled: bool = False
    params: dict = field(default_factory=dict)


class RenderPipeline:
    """Compose adjustments -> dither -> color -> saturation -> effects -> invert."""

    # FROZEN stage order (spec §8.1 + color/saturation/effects insert). The
    # render() body MUST call stages in exactly this sequence; test_render_order
    # spies on each stage and asserts the recorded call order equals this tuple.
    STAGE_ORDER: tuple[str, ...] = (
        "contrast", "midtones", "highlights", "blur", "dither",
        "color", "saturation", "effects", "invert",
    )

    def __init__(self, registry, color_engine=None, effect_stack=None) -> None:
        self.registry = registry
        self.color_engine = color_engine
        self.effect_stack = effect_stack
        self._cache: dict = {}
        self._cache_lock = threading.Lock()

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
            levels=settings.depth,
        )

        # 6: color — palette map, or broadcast grayscale to RGB. Snapshot the
        # engine once: the GUI thread can reassign self.color_engine (via
        # ImageEditor._sync_pipeline) while this runs on a render worker, and a
        # split read would dereference None mid-render.
        engine = self.color_engine
        if engine is not None:
            if getattr(engine, "mode", None) == "ramp":
                engine.depth = settings.depth
                engine.mapping = settings.color_mapping
            rgb = engine.map(d).astype(np.float32)
        else:
            rgb = np.repeat(np.asarray(d, np.float32)[..., None], 3, axis=2)

        # 7: saturation (RGB float32) then clamp to uint8
        rgb = apply_saturation(rgb, settings.saturation)
        rgb_u8 = clamp_u8(rgb)

        # 8: effects stack (RGB uint8) — snapshot once, same reassignment race.
        stack = self.effect_stack
        if stack is not None:
            rgb_u8 = stack.apply(rgb_u8)

        # 9: invert LAST (on RGB)
        if settings.invert:
            rgb_u8 = clamp_u8(apply_invert(rgb_u8.astype(np.float32), True))

        return np.asarray(rgb_u8, np.uint8)

    # ------------------------------------------------------------------ cache
    def render_cached(self, base_gray_f32, settings: RenderSettings,
                      temporal_field=None) -> np.ndarray:
        """Output-identical to ``render()`` but reuses intermediate arrays whose
        inputs are unchanged since the last call. Intended for interactive editing
        where one control moves at a time. Not on the frozen ``render()`` contract;
        ``test_render_cache`` proves byte-for-byte equality against ``render()``.

        Stages recompute from the first changed layer downward:
          L1 adjustments (contrast/midtones/highlights/blur)
          L2 dither      (+ style/scale/threshold/params/preview; temporal bypasses)
          L3 color       (+ color-engine signature)
          L4 saturation  (+ saturation value, includes clamp to uint8)
          L5 effects     (+ effect-stack signature)
          L6 invert
        """
        with self._cache_lock:
            c = self._cache
            g_in = np.asarray(base_gray_f32, dtype=np.float32)
            dirty = False

            # L1: tonal adjustments
            adj_sig = (settings.contrast, settings.midtones,
                       settings.highlights, settings.blur)
            if (c.get("_base") is not base_gray_f32 or c.get("adj_sig") != adj_sig
                    or "g" not in c):
                g = apply_contrast(g_in, settings.contrast)
                g = apply_midtones(g, settings.midtones)
                g = apply_highlights(g, settings.highlights)
                g = apply_blur(g, settings.blur)
                c["_base"] = base_gray_f32
                c["adj_sig"] = adj_sig
                c["g"] = g
                dirty = True
            g = c["g"]

            # L2: dither. A temporal field changes every frame, so it bypasses the
            # dither cache (and invalidates any stored non-temporal result).
            if temporal_field is not None:
                d = apply_dither(
                    g, style=settings.style, scale=settings.scale,
                    luminance_threshold=settings.luminance_threshold,
                    params=settings.params, registry=self.registry,
                    preview_disabled=settings.preview_disabled,
                    threshold_field=temporal_field,
                    levels=settings.depth)
                c.pop("dith_sig", None)
                c["d"] = d
                dirty = True
            else:
                dith_sig = (settings.style, settings.scale,
                            settings.luminance_threshold,
                            _params_sig(settings.params), settings.preview_disabled,
                            settings.depth)
                if dirty or c.get("dith_sig") != dith_sig or "d" not in c:
                    d = apply_dither(
                        g, style=settings.style, scale=settings.scale,
                        luminance_threshold=settings.luminance_threshold,
                        params=settings.params, registry=self.registry,
                        preview_disabled=settings.preview_disabled,
                        threshold_field=None,
                        levels=settings.depth)
                    c["dith_sig"] = dith_sig
                    c["d"] = d
                    dirty = True
            d = c["d"]

            # L3: color map (or grayscale->RGB broadcast). Snapshot the engine
            # once — a concurrent GUI-thread reassignment must not split reads.
            engine = self.color_engine
            col_sig = _color_sig(engine) + (settings.depth, settings.color_mapping)
            if dirty or c.get("col_sig") != col_sig or "colored" not in c:
                if engine is not None:
                    if getattr(engine, "mode", None) == "ramp":
                        engine.depth = settings.depth
                        engine.mapping = settings.color_mapping
                    colored = engine.map(d).astype(np.float32)
                else:
                    colored = np.repeat(np.asarray(d, np.float32)[..., None], 3, axis=2)
                c["col_sig"] = col_sig
                c["colored"] = colored
                dirty = True
            colored = c["colored"]

            # L4: saturation then clamp to uint8
            if dirty or c.get("sat_sig") != settings.saturation or "satout" not in c:
                satout = clamp_u8(apply_saturation(colored, settings.saturation))
                c["sat_sig"] = settings.saturation
                c["satout"] = satout
                dirty = True
            satout = c["satout"]

            # L5: effects stack — snapshot once (same reassignment race).
            stack = self.effect_stack
            fx_sig = _effect_sig(stack)
            if dirty or c.get("fx_sig") != fx_sig or "fx" not in c:
                fx = stack.apply(satout) if stack is not None else satout
                c["fx_sig"] = fx_sig
                c["fx"] = fx
                dirty = True
            fx = c["fx"]

            # L6: invert LAST (cheap; recomputed each call)
            if settings.invert:
                return clamp_u8(apply_invert(np.asarray(fx, np.float32), True))
            return np.asarray(fx, np.uint8)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()
