from __future__ import annotations

import inspect
import hashlib
import threading
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .adjustments import (
    apply_contrast, apply_midtones, apply_highlights, apply_blur,
    apply_saturation, apply_invert,
)
from .dithering.pipeline import apply_dither
from .imaging import clamp_u8
from .render_cache import DEFAULT_CACHE_BUDGET_BYTES, RenderCache


class RenderCancelled(Exception):
    """Raised at a stage boundary when ``is_cancelled`` reports obsolete work."""


def _check_cancelled(is_cancelled) -> None:
    # Boundary-only: never interrupts a running stage mid-computation, so a
    # stage's private buffers are always left in a consistent state.
    if is_cancelled is not None and is_cancelled():
        raise RenderCancelled


@lru_cache(maxsize=None)
def _accepts_out(fn) -> bool:
    # True iff ``fn`` can receive ``out=`` -- either an explicit ``out``
    # parameter (real adjustment funcs) or a ``**kwargs`` catch-all (e.g. a
    # MagicMock, whose signature is ``(*args, **kwargs)``). Cached per function
    # object (monkeypatched doubles are distinct objects, so each is inspected
    # once). Dispatch stays OUT of the live execution path: no try/except around
    # the stage body, so a real error propagates and the stage runs exactly once.
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "out" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _tonal_stage(fn, img, value, buf):
    """Call a tonal-adjustment stage through the shared L1 buffer (task 3.2:
    contrast/midtones/highlights must share ONE private buffer) when ``fn``
    accepts ``out=``. Production stages always do, so they take the fused
    ``out=buf`` branch every call; 2-arg test doubles take the allocating
    branch. The stage runs exactly once -- dispatch is by signature, not by
    catching a TypeError around live execution."""
    if _accepts_out(fn):
        return fn(img, value, out=buf)
    return fn(img, value)


def _params_sig(params: dict):
    # repr() keeps this hashable/comparable regardless of value types (numbers,
    # tuples, lists) while staying stable for identical content.
    return tuple(sorted((k, repr(v)) for k, v in (params or {}).items()))


def _color_sig(engine):
    if engine is None:
        return (None,)
    context = getattr(engine, "context", None)
    if context is not None:
        source = getattr(engine, "source_rgb", None)
        source_sig = ((source.shape, source.dtype.str,
                       hashlib.sha256(np.ascontiguousarray(source).tobytes()).hexdigest(),
                       getattr(engine, "source_dither", None),
                       getattr(engine, "source_dither_brighten", None))
                      if source is not None else None)
        return context.key, source_sig
    colors = np.asarray(engine.palette.colors)
    return (engine.mode, colors.shape, colors.dtype.str,
            colors.tobytes(order="C"), getattr(engine, "depth", None),
            getattr(engine, "mapping", None), getattr(engine, "phase", None))


def _engine_for_settings(engine, settings):
    if engine is not None and getattr(engine, "mode", None) == "ramp":
        return engine.with_settings(
            depth=settings.depth, mapping=settings.color_mapping
        )
    return engine


def _effect_sig(stack):
    if stack is None:
        return None
    return tuple((name, tuple(sorted((k, repr(v)) for k, v in params.items())))
                 for name, params in stack.items)


def render_settings_signature(settings) -> tuple:
    """Stable value signature for every creative render setting."""
    return tuple(
        (name, _params_sig(value) if name == "params" else value)
        for name, value in vars(settings).items()
    )


def render_context_signature(color_engine=None, effect_stack=None) -> tuple:
    """Stable value/content signature matching staged-cache context keys."""
    return (_color_sig(color_engine), _effect_sig(effect_stack))


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

    def __init__(self, registry, color_engine=None, effect_stack=None, *,
                 cache_budget_bytes=DEFAULT_CACHE_BUDGET_BYTES) -> None:
        self.registry = registry
        self.color_engine = color_engine
        self.effect_stack = effect_stack
        self._cache = RenderCache(cache_budget_bytes)
        self._cache_lock = threading.Lock()

    @property
    def cache_metrics(self):
        """Read-only snapshot of staged-cache memory and eviction metrics."""
        return self._cache.metrics

    def snapshot_context(self, color_engine=None, effect_stack=None) -> "RenderPipeline":
        """Return an immutable-context facade over this pipeline's cache owner.

        Request workers need fixed engine/effect references, while all facades
        must retain exactly one bounded staged cache and its synchronization.
        """
        snapshot = RenderPipeline(
            self.registry, color_engine, effect_stack, cache_budget_bytes=0)
        snapshot._cache = self._cache
        snapshot._cache_lock = self._cache_lock
        return snapshot

    def render(self, base_gray_f32, settings: RenderSettings,
               temporal_field=None, is_cancelled=None) -> np.ndarray:
        g = np.asarray(base_gray_f32, dtype=np.float32)

        # 1-4: tonal adjustments (grayscale float32, 0..255). Contrast/midtones/
        # highlights share ONE private buffer (proven byte-identical to three
        # separate allocations) instead of each allocating;
        # a true single-pass fusion changes pixel values, so three passes remain.
        buf = np.empty_like(g)
        g = _tonal_stage(apply_contrast, g, settings.contrast, buf)
        _check_cancelled(is_cancelled)
        g = _tonal_stage(apply_midtones, g, settings.midtones, buf)
        _check_cancelled(is_cancelled)
        g = _tonal_stage(apply_highlights, g, settings.highlights, buf)
        _check_cancelled(is_cancelled)
        g = apply_blur(g, settings.blur)
        _check_cancelled(is_cancelled)

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
        _check_cancelled(is_cancelled)

        # 6: color — palette map, or broadcast grayscale to RGB. Snapshot the
        # engine once: the GUI thread can reassign self.color_engine (via
        # ImageEditor._sync_pipeline) while this runs on a render worker, and a
        # split read would dereference None mid-render.
        engine = self.color_engine
        if engine is not None:
            engine = _engine_for_settings(engine, settings)
            rgb = engine.map(d)
        else:
            rgb = np.asarray(d, np.float32)
        _check_cancelled(is_cancelled)

        # 7: saturation and clamp fused into the final RGB uint8 allocation.
        rgb_u8 = apply_saturation(rgb, settings.saturation, output_u8=True)
        _check_cancelled(is_cancelled)

        # 8: effects stack (RGB uint8) — snapshot once, same reassignment race.
        stack = self.effect_stack
        if stack is not None:
            rgb_u8 = stack.apply(rgb_u8)
        _check_cancelled(is_cancelled)

        # 9: invert LAST (on RGB). Task 3.4: route through ONE call-private
        # float32 scratch buffer (never cached, never returned as itself --
        # the returned array is always the fresh clamp_u8 uint8 result)
        # instead of four full-image allocations. Proven byte-identical:
        # tests/test_render_scratch_reuse.py.
        if settings.invert:
            buf = np.empty_like(rgb_u8, dtype=np.float32)
            rgb_u8 = clamp_u8(apply_invert(rgb_u8, True, out=buf), inplace=True)

        return np.asarray(rgb_u8, np.uint8)

    # ------------------------------------------------------------------ cache
    def render_cached(self, base_gray_f32, settings: RenderSettings,
                      temporal_field=None, is_cancelled=None, cache_key=None) -> np.ndarray:
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
            entry_key = id(base_gray_f32) if cache_key is None else cache_key
            cached = self._cache.get(entry_key)
            # Never mutate a retained group: admission/eviction is atomic and a
            # failed or oversized render cannot publish a partial chain.
            c = dict(cached) if cached is not None else {}
            g_in = np.asarray(base_gray_f32, dtype=np.float32)
            dirty = False

            # L1: tonal adjustments
            adj_sig = (settings.contrast, settings.midtones,
                       settings.highlights, settings.blur)
            same_base = (c.get("_base") is base_gray_f32 if cache_key is None
                         else c.get("_base_key") == cache_key)
            if (not same_base or c.get("adj_sig") != adj_sig
                    or "g" not in c):
                # Fresh, call-private buffer -- never the shared/module-global
                # kind, so a later render's in-place work can't corrupt this
                # cached array once it's stored below.
                buf = np.empty_like(g_in)
                g = _tonal_stage(apply_contrast, g_in, settings.contrast, buf)
                g = _tonal_stage(apply_midtones, g, settings.midtones, buf)
                g = _tonal_stage(apply_highlights, g, settings.highlights, buf)
                g = apply_blur(g, settings.blur)
                c["_base"] = base_gray_f32
                c["_base_key"] = cache_key
                c["adj_sig"] = adj_sig
                c["g"] = g
                dirty = True
            g = c["g"]
            _check_cancelled(is_cancelled)

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
            _check_cancelled(is_cancelled)

            # L3: color map (or grayscale->RGB broadcast). Snapshot the engine
            # once — a concurrent GUI-thread reassignment must not split reads.
            engine = self.color_engine
            engine = _engine_for_settings(engine, settings)
            col_sig = _color_sig(engine)
            if dirty or c.get("col_sig") != col_sig or "colored" not in c:
                if engine is not None:
                    colored = engine.map(d)
                else:
                    colored = np.asarray(d, np.float32)
                c["col_sig"] = col_sig
                c["colored"] = colored
                dirty = True
            colored = c["colored"]
            _check_cancelled(is_cancelled)

            # L4: saturation then clamp to uint8
            if dirty or c.get("sat_sig") != settings.saturation or "satout" not in c:
                satout = apply_saturation(colored, settings.saturation, output_u8=True)
                c["sat_sig"] = settings.saturation
                c["satout"] = satout
                dirty = True
            satout = c["satout"]
            _check_cancelled(is_cancelled)

            # L5: effects stack — snapshot once (same reassignment race).
            stack = self.effect_stack
            fx_sig = _effect_sig(stack)
            if dirty or c.get("fx_sig") != fx_sig or "fx" not in c:
                fx = stack.apply(satout) if stack is not None else satout
                c["fx_sig"] = fx_sig
                c["fx"] = fx
                dirty = True
            fx = c["fx"]
            _check_cancelled(is_cancelled)

            # L6: invert LAST (recomputed each call; never cached). Same
            # call-private scratch route as render()'s L9 (task 3.4).
            if settings.invert:
                buf = np.empty_like(fx, dtype=np.float32)
                result = clamp_u8(apply_invert(fx, True, out=buf), inplace=True)
            else:
                result = np.asarray(fx, np.uint8)
            self._cache.put(entry_key, c)
            return result

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def configure_cache_budget(self, budget_bytes: int) -> None:
        """Replace the editor-owned render cache with an empty bounded cache."""
        replacement = RenderCache(budget_bytes)
        with self._cache_lock:
            self._cache = replacement
