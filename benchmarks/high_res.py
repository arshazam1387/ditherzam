"""Reproducible high-resolution benchmark with JIT enabled.

The safe default keeps Python RGB diffusion at 480p. ``--tier full`` adds 5K;
large diffusion requires the explicit ``--large-diffused`` flag.
"""
from __future__ import annotations

import argparse
import gc
import time
import tracemalloc
from dataclasses import replace

import numpy as np

from ditherzam.color.engine import ColorEngine
from ditherzam.color.palette import Palette
from ditherzam.ui.preview import render_preview

from .common import REGISTRY, RenderPipeline, RenderSettings, SIZES, make_gray

try:
    import psutil
except ImportError:  # optional working-set measurement
    psutil = None


def palette(k: int) -> Palette:
    i = np.arange(k, dtype=np.uint32)
    colors = np.column_stack(
        ((i * 67) % 256, (i * 151 + 31) % 256, (i * 211 + 97) % 256)
    ).astype(np.float32)
    return Palette(f"bench-{k}", colors)


def measure(fn, repeats: int):
    gc.collect()
    process = psutil.Process() if psutil else None
    rss0 = process.memory_info().rss if process else None
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    first = (time.perf_counter() - t0) * 1000.0
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss1 = process.memory_info().rss if process else None
    assert result is not None
    rss_delta = None if rss0 is None else (rss1 - rss0) / 1048576
    return first, float(np.median(samples)), peak / 1048576, rss_delta


def row(case: str, size: str, values) -> None:
    first, warm, peak, rss = values
    rss_text = "n/a" if rss is None else f"{rss:.1f}"
    print(f"{case:<25} {size:>6} {first:>10.1f} {warm:>10.1f} "
          f"{peak:>10.1f} {rss_text:>10}")


def header(title: str) -> None:
    print(f"\n== {title} ==")
    print(f"{'case':<25} {'size':>6} {'first ms':>10} {'warm ms':>10} "
          f"{'py peak MB':>10} {'rss dMB':>10}")


def exact(args) -> None:
    sizes = {
        "quick": ("1080p",),
        "default": ("1080p", "4K"),
        "full": ("1080p", "4K", "5K"),
    }[args.tier]
    palette_sizes = (4, 16, 64) if args.tier == "full" else (4, 16)
    header("exact render: first vs warm")
    for size in sizes:
        h, w = SIZES[size]
        base = make_gray(h, w)
        for mode in ("off", "ramp", "nearest", "ordered", "diffused"):
            run_size, run_base = size, base
            if mode == "diffused" and not args.large_diffused:
                if size != sizes[0]:
                    continue
                run_size = "480p"
                hh, ww = SIZES[run_size]
                run_base = make_gray(hh, ww)
            ks = palette_sizes if mode in ("nearest", "ordered") else (4,)
            for k in ks:
                pipe = RenderPipeline(REGISTRY, ColorEngine(palette(k), mode), None)
                settings = RenderSettings(
                    style="Bayer-Matrix 4x4", scale=5, depth=4)
                row(f"{mode}/k{k}", run_size,
                    measure(lambda: pipe.render(run_base, settings), args.repeats))


def previews(args) -> None:
    sizes = ("1080p",) if args.tier == "quick" else ("1080p", "4K")
    if args.tier == "full":
        sizes += ("5K",)
    header("capped preview (result currently source-sized)")
    for size in sizes:
        h, w = SIZES[size]
        base = make_gray(h, w)
        for cap in args.caps:
            pipe = RenderPipeline(
                REGISTRY, ColorEngine(palette(16), "nearest"), None)
            settings = RenderSettings(
                style="Bayer-Matrix 4x4", scale=5, depth=4)
            row(f"preview/cap{cap}", size,
                measure(lambda: render_preview(
                    pipe, base, settings, cap), args.repeats))


def cached(args) -> None:
    size = "1080p" if args.tier == "quick" else "4K"
    h, w = SIZES[size]
    base = make_gray(h, w)
    pipe = RenderPipeline(REGISTRY, ColorEngine(palette(16), "nearest"), None)
    settings = RenderSettings(style="Bayer-Matrix 4x4", scale=5, depth=4)
    cases = {
        "unchanged": settings,
        "dither": replace(settings, luminance_threshold=63),
        "palette/mode": settings,
        "saturation": replace(settings, saturation=72),
        "invert": replace(settings, invert=True),
    }
    header(f"cached mutations @{size} (re-primed per sample)")
    for label, changed in cases.items():
        def tick():
            pipe.color_engine = ColorEngine(palette(16), "nearest")
            pipe.render_cached(base, settings)
            if label == "palette/mode":
                pipe.color_engine = ColorEngine(palette(16), "ordered")
            return pipe.render_cached(base, changed)
        row(label, size, measure(tick, args.repeats))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier", choices=("quick", "default", "full"), default="default")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--caps", type=int, nargs="+", default=(480, 720, 1080, 1440, 2160))
    parser.add_argument(
        "--large-diffused", action="store_true",
        help="DANGER: benchmark large Python RGB diffusion")
    parser.add_argument(
        "--section", choices=("all", "exact", "preview", "cache"),
        default="all")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    print(f"tier={args.tier}; repeats={args.repeats}; "
          f"large_diffused={args.large_diffused}")
    print("memory: py peak=tracemalloc; rss dMB=working-set delta "
          "(psutil optional)")
    if args.section in ("all", "exact"):
        exact(args)
    if args.section in ("all", "preview"):
        previews(args)
    if args.section in ("all", "cache"):
        cached(args)


if __name__ == "__main__":
    main()
