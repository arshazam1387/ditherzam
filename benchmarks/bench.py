"""ditherzam render-core benchmark. Run with JIT ENABLED:

    .venv/Scripts/python.exe -m benchmarks.bench

Reports cold vs warm render times across sizes/styles and a per-stage breakdown
for the heavy path. Qt-free.
"""
from __future__ import annotations

import time

from .common import (
    REGISTRY, RenderPipeline, RenderSettings, SIZES,
    make_gray, gameboy_engine, heavy_effects, timeit, stage_timers,
)

STYLES = ["Floyd-Steinberg", "Atkinson", "Bayer-Matrix 4x4"]


def _pipeline(color=None, effects=None):
    return RenderPipeline(REGISTRY, color, effects)


def bench_cold_import() -> None:
    print("\n== cold import (already imported in this process) ==")
    print("  (measure separately: `python -X importtime -c \"import ditherzam.render\"`)")


def bench_warm_vs_cold() -> None:
    print("\n== warm vs cold render (ms) ==")
    print(f"{'size':>7} {'style':>20} {'cold':>8} {'warm':>8}")
    for size_label, (h, w) in SIZES.items():
        base = make_gray(h, w)
        for style in STYLES:
            settings = RenderSettings(style=style, scale=5)
            pipe = _pipeline()
            # cold: first call pays JIT (cache=True persists across processes,
            # so this is only a true "cold" number on a cache-cleared machine)
            t0 = time.perf_counter()
            pipe.render(base, settings)
            cold = (time.perf_counter() - t0) * 1000.0
            warm = timeit(lambda: pipe.render(base, settings), n=5)
            print(f"{size_label:>7} {style:>20} {cold:>8.1f} {warm:>8.1f}")


def bench_heavy_path() -> None:
    print("\n== heavy path: FS + Game Boy palette + Chromatic Aberration + Epsilon Glow ==")
    for size_label in ("512", "1080p"):
        h, w = SIZES[size_label]
        base = make_gray(h, w)
        settings = RenderSettings(style="Floyd-Steinberg", scale=5)
        pipe = _pipeline(gameboy_engine(), heavy_effects())
        warm = timeit(lambda: pipe.render(base, settings), n=5)
        print(f"  {size_label}: {warm:.1f} ms (warm)")


def bench_stages() -> None:
    print("\n== per-stage breakdown @1080p (heavy path, warm) ==")
    h, w = SIZES["1080p"]
    base = make_gray(h, w)
    settings = RenderSettings(style="Floyd-Steinberg", scale=5)
    pipe = _pipeline(gameboy_engine(), heavy_effects())
    pipe.render(base, settings)  # warm JIT
    with stage_timers() as totals:
        runs = 5
        for _ in range(runs):
            pipe.render(base, settings)
    for stage, ms in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {stage:>12}: {ms / runs:8.2f} ms")


def main() -> None:
    bench_cold_import()
    bench_warm_vs_cold()
    bench_heavy_path()
    bench_stages()


if __name__ == "__main__":
    main()
