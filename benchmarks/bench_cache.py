"""Per-control cached-render latency: how long a single slider tick takes when
only that one control changed, vs a full uncached render. Qt-free, JIT enabled.

    .venv/Scripts/python.exe -m benchmarks.bench_cache
"""
from __future__ import annotations

import time

import numpy as np

from .common import (
    REGISTRY, RenderPipeline, RenderSettings, SIZES,
    make_gray, gameboy_engine, heavy_effects,
)


def _median_ms(fn, n=5):
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(xs))


def main() -> None:
    h, w = SIZES["1080p"]
    base = make_gray(h, w)
    pipe = RenderPipeline(REGISTRY, gameboy_engine(), heavy_effects())

    s0 = RenderSettings(style="Floyd-Steinberg", scale=5, saturation=50)
    # warm JIT + prime the cache
    pipe.render(base, s0)
    pipe.render_cached(base, s0)

    full = _median_ms(lambda: pipe.render(base, s0))

    # Each scenario: re-prime the cache with s0 (untimed) before every timed
    # render of s, so we always measure "one control changed since last render".
    def cached_after(change: dict) -> float:
        s = RenderSettings(**{**s0.__dict__, **change})
        xs = []
        for _ in range(5):
            pipe.render_cached(base, s0)
            t0 = time.perf_counter()
            pipe.render_cached(base, s)
            xs.append((time.perf_counter() - t0) * 1000.0)
        return float(np.median(xs))

    scenarios = {
        "saturation": {"saturation": 80},
        "effects (via stack swap)": {},   # handled below
        "luminance_threshold": {"luminance_threshold": 70},
        "contrast (top of pipeline)": {"contrast": 70},
        "invert": {"invert": True},
    }

    print("== @1080p heavy path: single-control cached tick vs full render ==")
    print(f"  {'full uncached render':>32}: {full:8.1f} ms")
    for label, change in scenarios.items():
        if label.startswith("effects"):
            continue
        ms = cached_after(change)
        print(f"  {label:>32}: {ms:8.1f} ms")

    # effects-only change: baseline stack (untimed) then a mutated stack (timed).
    base_stack = heavy_effects()
    alt_stack = heavy_effects()
    alt_stack.items[0] = ("Chromatic Aberration", {"shift": 5})
    xs = []
    for _ in range(5):
        pipe.effect_stack = base_stack
        pipe.render_cached(base, s0)
        pipe.effect_stack = alt_stack
        t0 = time.perf_counter()
        pipe.render_cached(base, s0)
        xs.append((time.perf_counter() - t0) * 1000.0)
    print(f"  {'effects param change':>32}: {float(np.median(xs)):8.1f} ms")


if __name__ == "__main__":
    main()
