"""Profile tonal-adjustment fusion candidates with JIT enabled.

Run from the repository root::

    .venv/Scripts/python.exe -m benchmarks.adjustment_fusion

The benchmark deliberately keeps candidate implementations local.  It produces
evidence for Task 3.2 without changing the frozen production adjustment API.
"""
from __future__ import annotations

import hashlib
import json
import time
import tracemalloc

import numpy as np
from numba import njit, prange

from benchmarks.common import SIZES, make_gray, timeit
from ditherzam.adjustments import apply_contrast, apply_highlights, apply_midtones


PARAMETERS = ((70.0, 30.0, 80.0), (13.0, 91.0, 47.0))


def production_chain(image: np.ndarray, contrast: float, midtones: float,
                     highlights: float) -> np.ndarray:
    out = apply_contrast(image, contrast)
    out = apply_midtones(out, midtones)
    return apply_highlights(out, highlights)


def inplace_three_pass(image: np.ndarray, contrast: float, midtones: float,
                       highlights: float) -> np.ndarray:
    """Allocation-reduced candidate which retains every float32 stage boundary."""
    out = np.array(image, dtype=np.float32, copy=True)
    np.multiply(out, contrast / 50.0, out=out)
    gamma = max(1.0 + (midtones - 50.0) / 200.0, 0.1)
    np.divide(out, 255.0, out=out)
    np.power(out, 1.0 / gamma, out=out)
    np.multiply(out, 255.0, out=out)
    np.multiply(out, 1.0 + (highlights - 50.0) / 100.0, out=out)
    return out


@njit(cache=True, parallel=True)
def one_pass_numba(image: np.ndarray, contrast: float, midtones: float,
                   highlights: float) -> np.ndarray:
    """True fused candidate; exactness is measured, never assumed."""
    h, w = image.shape
    out = np.empty_like(image)
    contrast_factor = contrast / 50.0
    gamma = max(1.0 + (midtones - 50.0) / 200.0, 0.1)
    exponent = 1.0 / gamma
    highlight_factor = 1.0 + (highlights - 50.0) / 100.0
    for y in prange(h):
        for x in range(w):
            value = np.float32(image[y, x] * contrast_factor)
            value = np.float32(255.0 * (value / 255.0) ** exponent)
            out[y, x] = np.float32(value * highlight_factor)
    return out


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def peak_tracemalloc(fn) -> int:
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
        return peak
    finally:
        tracemalloc.stop()


def timed_ms(fn, *, size: str) -> float:
    # 4K is expensive enough that three samples are sufficient; five reduces
    # noise at 1080p.  Compilation is always excluded by timeit's warmup.
    return timeit(fn, n=3 if size == "4K" else 5, warmup=1)


def main() -> None:
    results: dict[str, object] = {
        "numpy": np.__version__,
        "jit_disabled": bool(__import__("numba").config.DISABLE_JIT),
        "sizes": {},
    }
    if results["jit_disabled"]:
        raise SystemExit("Refusing to profile with NUMBA_DISABLE_JIT enabled")

    # Compile before recording any timings or allocations.
    one_pass_numba(np.zeros((2, 2), np.float32), *PARAMETERS[0])

    for size in ("1080p", "4K"):
        h, w = SIZES[size]
        image = make_gray(h, w)
        contrast, midtones, highlights = PARAMETERS[0]
        funcs = {
            "contrast": lambda: apply_contrast(image, contrast),
            "midtones": lambda: apply_midtones(image, midtones),
            "highlights": lambda: apply_highlights(image, highlights),
            "production_together": lambda: production_chain(
                image, contrast, midtones, highlights),
            "inplace_three_pass": lambda: inplace_three_pass(
                image, contrast, midtones, highlights),
            "one_pass_numba": lambda: one_pass_numba(
                image, contrast, midtones, highlights),
        }
        expected = funcs["production_together"]()
        size_results: dict[str, object] = {}
        for name, fn in funcs.items():
            output = fn()
            size_results[name] = {
                "median_ms": round(timed_ms(fn, size=size), 3),
                "tracemalloc_peak_mib": round(peak_tracemalloc(fn) / 2**20, 3),
                "sha256": digest(output),
                "exact": bool(np.array_equal(output, expected))
                if name.endswith("pass") or name == "one_pass_numba" else None,
                "different_values": int(np.count_nonzero(output != expected))
                if name.endswith("pass") or name == "one_pass_numba" else None,
            }
        results["sizes"][size] = size_results

    # Exactness sweep includes non-finite-producing negative upstream values,
    # zeros, boundaries, and deterministic random float32 pixels.
    rng = np.random.default_rng(20260709)
    exactness_input = np.concatenate((
        np.array([-255.0, -0.0, 0.0, np.nextafter(np.float32(0), np.float32(1)),
                  1.0, 127.5, 254.99998, 255.0, 511.0], dtype=np.float32),
        rng.uniform(0, 255, 100_000).astype(np.float32),
    )).reshape(1, -1)
    sweep = {}
    for params in PARAMETERS:
        expected = production_chain(exactness_input, *params)
        sweep[str(params)] = {}
        for name, fn in (("inplace_three_pass", inplace_three_pass),
                         ("one_pass_numba", one_pass_numba)):
            actual = fn(exactness_input, *params)
            finite = np.isfinite(expected) & np.isfinite(actual)
            sweep[str(params)][name] = {
                "exact_equal_nan": bool(np.array_equal(actual, expected, equal_nan=True)),
                "different_values": int(np.count_nonzero(
                    (actual != expected) & ~(np.isnan(actual) & np.isnan(expected)))),
                "max_abs_finite_difference": float(
                    np.max(np.abs(actual[finite] - expected[finite]), initial=0.0)),
                "sha256": digest(actual),
                "production_sha256": digest(expected),
            }
    results["exactness_sweep"] = sweep
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
