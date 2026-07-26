"""Informational MT-12 capped-preview and exact-confirm benchmark."""
from __future__ import annotations

import statistics
import time
import tracemalloc

import numpy as np

from ditherzam.layers import (
    MaskRefinementKind,
    SmartRefinementSpec,
    derive_refined_smart_mask,
    derive_refined_smart_preview,
)
from ditherzam.masking.contracts import (
    InferenceIdentity,
    ModelIdentity,
    ProbabilityMap,
    source_identity,
)
from ditherzam.masking.settings import MaskTarget


def _case(height: int, width: int) -> tuple[np.ndarray, ProbabilityMap]:
    rgba = np.zeros((height, width, 4), np.uint8)
    rgba[..., 3] = 255
    yy, xx = np.ogrid[:height, :width]
    values = ((xx + yy) / float(height + width - 2)).astype(np.float32)
    identity = InferenceIdentity(
        source_identity(rgba), ModelIdentity("bench", "1", "0" * 64),
        "v1", "subject")
    return rgba, ProbabilityMap(identity, values)


def _measure(callable_, samples: int = 31) -> tuple[float, int]:
    callable_()
    timings = []
    peak = 0
    for _ in range(samples):
        tracemalloc.start()
        started = time.perf_counter()
        callable_()
        timings.append((time.perf_counter() - started) * 1000.0)
        _, observed = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak = max(peak, observed)
    return statistics.quantiles(
        timings, n=20, method="inclusive")[18], peak


def main() -> None:
    spec = SmartRefinementSpec(
        MaskTarget.SUBJECT, 50, feather_radius=8)
    for label, shape in (("1080p", (1080, 1920)), ("4K", (2160, 3840))):
        rgba, probability = _case(*shape)
        preview = _measure(lambda: derive_refined_smart_preview(
            probability, spec, rgba=rgba))
        exact = _measure(lambda: derive_refined_smart_mask(
            probability, spec, rgba=rgba))
        retained = rgba.nbytes + probability.values.nbytes
        print(
            f"{label} (31 samples, inclusive p95): "
            f"preview={preview[0]:.2f} ms traced-working={preview[1]:,}; "
            f"exact={exact[0]:.2f} ms traced-working={exact[1]:,}; "
            f"retained-inputs={retained:,}; "
            f"accounted-exact={retained + exact[1]:,}")


if __name__ == "__main__":
    main()
