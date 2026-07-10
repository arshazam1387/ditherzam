"""Shared helpers for the ditherzam benchmark harness.

Qt-free. Builds deterministic synthetic inputs and times render-core work with
JIT ENABLED (the default). Do NOT set NUMBA_DISABLE_JIT here — JIT-off numbers
are meaningless for perf.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

from ditherzam.dithering import registry as REGISTRY
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.color.engine import ColorEngine
from ditherzam.color.palette import builtin_palettes
from ditherzam.effects.stack import EffectStack

SIZES = {
    "480p": (480, 854),
    "720p": (720, 1280),
    "512": (512, 512),
    "1080p": (1080, 1920),   # (H, W)
    "4K": (2160, 3840),
    "5K": (2880, 5120),
}


def make_gray(h: int, w: int, seed: int = 7) -> np.ndarray:
    """Deterministic gradient + structured noise grayscale, float32 0..255."""
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    base = 0.5 * yy + 0.5 * xx
    noise = rng.normal(0.0, 18.0, size=(h, w)).astype(np.float32)
    return np.clip(base + noise, 0, 255).astype(np.float32)


def gameboy_engine() -> ColorEngine:
    pals = builtin_palettes()
    for key in ("gameboy", "Game Boy", "GameBoy", "Gameboy"):
        if key in pals:
            return ColorEngine(pals[key], "nearest")
    first = next(iter(pals.values()))
    return ColorEngine(first, "nearest")


def cga_engine() -> ColorEngine:
    pals = builtin_palettes()
    return ColorEngine(pals["cga"], "nearest")


def heavy_effects() -> EffectStack:
    s = EffectStack()
    s.add("Chromatic Aberration", shift=2)
    s.add("Epsilon Glow", radius=4.0, strength=0.5)
    return s


def timeit(fn, n: int = 5, warmup: int = 1) -> float:
    """Median wall-clock ms over n runs after `warmup` untimed runs."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(samples))


@contextmanager
def stage_timers():
    """Monkeypatch render.py's stage callables to accumulate per-stage ms.

    Yields a dict {stage: ms_total}. Wraps the names as imported into
    ditherzam.render, plus ColorEngine.map / EffectStack.apply.
    """
    import ditherzam.render as R

    totals: dict[str, float] = {}
    originals: dict[str, object] = {}

    def wrap(mod, attr, label):
        orig = getattr(mod, attr)
        originals[(id(mod), attr)] = (mod, attr, orig)

        def timed(*a, **k):
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                totals[label] = totals.get(label, 0.0) + (time.perf_counter() - t0) * 1000.0
        setattr(mod, attr, timed)

    wrap(R, "apply_contrast", "contrast")
    wrap(R, "apply_midtones", "midtones")
    wrap(R, "apply_highlights", "highlights")
    wrap(R, "apply_blur", "blur")
    wrap(R, "apply_dither", "dither")
    wrap(R, "apply_saturation", "saturation")
    wrap(R, "clamp_u8", "clamp")
    wrap(R, "apply_invert", "invert")
    wrap(ColorEngine, "map", "color")
    wrap(EffectStack, "apply", "effects")
    try:
        yield totals
    finally:
        for mod, attr, orig in originals.values():
            setattr(mod, attr, orig)
