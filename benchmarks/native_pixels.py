"""Shared exactness and timing helpers for native pixel workstreams."""
from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections.abc import Callable, Iterable

import numpy as np


BOUNDARY_U8 = np.array([0, 1, 127, 128, 254, 255], dtype=np.uint8)
DEFAULT_SEED = 0xD17E


def deterministic_u8(shape: tuple[int, ...], *, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Return owned, C-contiguous repeatable input containing boundary values."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=shape, dtype=np.uint8)
    flat = array.reshape(-1)
    flat[: min(flat.size, BOUNDARY_U8.size)] = BOUNDARY_U8[: flat.size]
    return np.ascontiguousarray(array)


def output_digest(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    metadata = f"{array.dtype.str}|{array.shape}".encode("ascii")
    return hashlib.sha256(metadata + array.tobytes()).hexdigest()


def compare_exact_outputs(
    reference_call: Callable[..., np.ndarray],
    native_call: Callable[..., np.ndarray],
    *args: object,
    **kwargs: object,
) -> dict[str, object]:
    """Run independent seams and enforce the complete native output contract."""
    reference = reference_call(*args, **kwargs)
    native = native_call(*args, **kwargs)
    for label, value in (("reference", reference), ("native", native)):
        if not isinstance(value, np.ndarray):
            raise AssertionError(f"{label} output is not a NumPy array")
        if not value.flags.owndata:
            raise AssertionError(f"{label} output does not own its buffer")
        if not value.flags.c_contiguous:
            raise AssertionError(f"{label} output is not C-contiguous")
    if native.dtype != reference.dtype:
        raise AssertionError(f"dtype differs: {native.dtype} != {reference.dtype}")
    if native.shape != reference.shape:
        raise AssertionError(f"shape differs: {native.shape} != {reference.shape}")
    if not np.array_equal(native, reference):
        raise AssertionError("native bytes differ from the reference")
    reference_digest = output_digest(reference)
    native_digest = output_digest(native)
    if native_digest != reference_digest:
        raise AssertionError("native digest differs from the reference")
    return {
        "dtype": native.dtype.str,
        "shape": native.shape,
        "digest": native_digest,
    }


def measure(
    function: Callable[[], object],
    *,
    warmup: int = 1,
    samples: int = 7,
) -> dict[str, object]:
    """Warm a callable, then report median/p95 milliseconds and output digest."""
    if warmup < 0 or samples < 1:
        raise ValueError("warmup must be >= 0 and samples must be >= 1")
    for _ in range(warmup):
        function()
    elapsed: list[float] = []
    result: object = None
    for _ in range(samples):
        started = time.perf_counter()
        result = function()
        elapsed.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(elapsed)
    p95_index = min(len(ordered) - 1, max(0, int(np.ceil(0.95 * len(ordered))) - 1))
    return {
        "median_ms": statistics.median(elapsed),
        "p95_ms": ordered[p95_index],
        "samples": samples,
        "digest": output_digest(result),
    }


def speedup(reference_ms: float, native_ms: float) -> float:
    if reference_ms < 0 or native_ms <= 0:
        raise ValueError("timings must be positive")
    return reference_ms / native_ms


def shapes(include_full_frame: bool = False) -> Iterable[tuple[int, int, int]]:
    """Return small test shapes, plus full-frame benchmark descriptors on request.

    Unit tests intentionally use only the base shapes. Consumers opt into 1080p
    and 4K descriptors when running benchmarks, avoiding large test allocations.
    """
    base = ((1, 1, 4), (3, 5, 4), (7, 2, 4))
    if include_full_frame:
        return (*base, (1080, 1920, 4), (2160, 3840, 4))
    return base


def benchmark_compositor(*, warmup: int = 1, samples: int = 7) -> list[dict[str, object]]:
    """Benchmark exact reference/native normal and overlay full-frame blends."""
    from ditherzam.layers.blend import _blend_layer_native, _blend_layer_reference

    rows: list[dict[str, object]] = []
    for height, width, channels in ((1080, 1920, 4), (2160, 3840, 4)):
        backdrop = deterministic_u8(
            (height, width, channels), seed=DEFAULT_SEED + height
        )
        source = deterministic_u8(
            (height, width, channels), seed=DEFAULT_SEED + width
        )
        for mode in ("normal", "overlay"):
            reference = measure(
                lambda: _blend_layer_reference(backdrop, source, mode, 73),
                warmup=warmup,
                samples=samples,
            )
            native = measure(
                lambda: _blend_layer_native(backdrop, source, mode, 73),
                warmup=warmup,
                samples=samples,
            )
            if reference["digest"] != native["digest"]:
                raise AssertionError(
                    f"{width}x{height} {mode}: native digest differs from reference"
                )
            rows.append(
                {
                    "operation": "blend_layer",
                    "resolution": f"{width}x{height}",
                    "mode": mode,
                    "opacity": 73,
                    "samples": samples,
                    "reference_median_ms": reference["median_ms"],
                    "reference_p95_ms": reference["p95_ms"],
                    "native_median_ms": native["median_ms"],
                    "native_p95_ms": native["p95_ms"],
                    "speedup": speedup(
                        float(reference["median_ms"]), float(native["median_ms"])
                    ),
                    "digest": native["digest"],
                }
            )
    return rows


def benchmark_selection(*, samples: int = 7) -> list[dict[str, object]]:
    """Benchmark exact selection mapping at 1080p and 4K after warmup."""
    from ditherzam.layers.selection import (
        TemporarySelection,
        _selection_to_source_native,
        _selection_to_source_reference,
    )

    cases = (
        ("1080p", (1080, 1920)),
        ("4k", (2160, 3840)),
    )
    transforms = (
        ("identity", dict(layer_x=0.0, layer_y=0.0, scale_x=1.0, scale_y=1.0)),
        (
            "transformed",
            dict(layer_x=-13.25, layer_y=7.75, scale_x=0.83, scale_y=1.17),
        ),
    )
    results: list[dict[str, object]] = []
    for resolution, shape in cases:
        selection = TemporarySelection(deterministic_u8(shape, seed=sum(shape)))
        for transform_name, transform in transforms:
            comparison = compare_exact_outputs(
                _selection_to_source_reference,
                _selection_to_source_native,
                selection,
                shape,
                **transform,
            )
            reference = measure(
                lambda: _selection_to_source_reference(
                    selection, shape, **transform
                ),
                warmup=1,
                samples=samples,
            )
            native = measure(
                lambda: _selection_to_source_native(selection, shape, **transform),
                warmup=1,
                samples=samples,
            )
            results.append(
                {
                    "operation": "selection_to_source",
                    "resolution": resolution,
                    "transform": transform_name,
                    "shape": shape,
                    "samples": samples,
                    "reference_median_ms": reference["median_ms"],
                    "reference_p95_ms": reference["p95_ms"],
                    "native_median_ms": native["median_ms"],
                    "native_p95_ms": native["p95_ms"],
                    "speedup": speedup(
                        float(reference["median_ms"]), float(native["median_ms"])
                    ),
                    "digest": comparison["digest"],
                }
            )
    return results


def benchmark_transitions(
    *, warmup: int = 1, samples: int = 7
) -> list[dict[str, object]]:
    """Benchmark exact scalar blends and soft spatial wipes at 1080p and 4K."""
    from ditherzam.composition.transitions import (
        _blend_straight_rgba_native,
        _blend_straight_rgba_reference,
    )

    results: list[dict[str, object]] = []
    for resolution, (height, width) in (
        ("1080p", (1080, 1920)),
        ("4k", (2160, 3840)),
    ):
        a = deterministic_u8((height, width, 4), seed=DEFAULT_SEED + height)
        b = deterministic_u8((height, width, 4), seed=DEFAULT_SEED + width)
        comparison = compare_exact_outputs(
            _blend_straight_rgba_reference,
            _blend_straight_rgba_native,
            a,
            b,
            0.37,
        )
        reference = measure(
            lambda: _blend_straight_rgba_reference(a, b, 0.37),
            warmup=warmup,
            samples=samples,
        )
        native = measure(
            lambda: _blend_straight_rgba_native(a, b, 0.37),
            warmup=warmup,
            samples=samples,
        )
        results.append(
            {
                "operation": "transition_scalar",
                "resolution": resolution,
                "samples": samples,
                "reference_median_ms": reference["median_ms"],
                "reference_p95_ms": reference["p95_ms"],
                "native_median_ms": native["median_ms"],
                "native_p95_ms": native["p95_ms"],
                "speedup": speedup(
                    float(reference["median_ms"]), float(native["median_ms"])
                ),
                "digest": comparison["digest"],
            }
        )

        base_gray = deterministic_u8((height, width), seed=height + width)

        def make_weight(mode: str) -> np.ndarray:
            if mode == "linear":
                coord = np.broadcast_to(
                    np.arange(width, dtype=np.float64) / max(width - 1, 1),
                    (height, width),
                )
            elif mode == "radial":
                yy, xx = np.indices((height, width), dtype=np.float64)
                center_y = (height - 1) / 2.0
                center_x = (width - 1) / 2.0
                coord = np.hypot(yy - center_y, xx - center_x)
                coord /= np.hypot(center_y, center_x)
            else:
                coord = base_gray.astype(np.float64) / 255.0
            return np.ascontiguousarray(
                np.clip((0.57 - coord) / 0.2 + 0.5, 0.0, 1.0)
            )

        for mode in ("linear", "radial", "luma"):
            weight = make_weight(mode)
            comparison = compare_exact_outputs(
                _blend_straight_rgba_reference,
                _blend_straight_rgba_native,
                a,
                b,
                weight,
            )
            reference = measure(
                lambda mode=mode: _blend_straight_rgba_reference(
                    a, b, make_weight(mode)
                ),
                warmup=warmup,
                samples=samples,
            )
            native = measure(
                lambda mode=mode: _blend_straight_rgba_native(
                    a, b, make_weight(mode)
                ),
                warmup=warmup,
                samples=samples,
            )
            results.append(
                {
                    "operation": "spatial_wipe_end_to_end",
                    "resolution": resolution,
                    "mode": mode,
                    "samples": samples,
                    "reference_median_ms": reference["median_ms"],
                    "reference_p95_ms": reference["p95_ms"],
                    "native_median_ms": native["median_ms"],
                    "native_p95_ms": native["p95_ms"],
                    "speedup": speedup(
                        float(reference["median_ms"]), float(native["median_ms"])
                    ),
                    "digest": comparison["digest"],
                }
            )
    return results


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    # Direct-script execution otherwise resolves whichever editable checkout is
    # installed in the interpreter instead of the repository being benchmarked.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument(
        "--selection", action="store_true", help="run selection mapping benchmarks"
    )
    parser.add_argument(
        "--transitions", action="store_true", help="run transition benchmarks"
    )
    arguments = parser.parse_args()
    if arguments.selection:
        benchmark = benchmark_selection(samples=arguments.samples)
    elif arguments.transitions:
        benchmark = benchmark_transitions(
            warmup=arguments.warmup,
            samples=arguments.samples,
        )
    else:
        benchmark = benchmark_compositor(
            warmup=arguments.warmup,
            samples=arguments.samples,
        )
    print(
        json.dumps(benchmark, indent=2)
    )
