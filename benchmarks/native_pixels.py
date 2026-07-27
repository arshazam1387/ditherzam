"""Shared exactness and timing helpers for native pixel workstreams."""
from __future__ import annotations

import hashlib
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
