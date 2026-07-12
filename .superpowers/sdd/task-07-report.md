# SM-07 Report — Canonical decoded source RGBA retention

## Outcome

- `_DecodeWorker` decodes once to canonical straight `uint8` RGBA, derives the
  existing RGB and PIL RGB-to-L render inputs from its RGB channels, and emits all
  three arrays.
- `ImageEditor` retains `_base_rgba` as owned, immutable, C-contiguous storage.
- Programmatic RGB and grayscale loads synthesize opaque RGBA while preserving
  the prior `_base_rgb` semantics and exact `_base_gray` render input.
- Shape and cross-channel validation completes before source fields change, so a
  rejected replacement leaves the prior source intact.
- Canonical inputs are validated before coercion: gray is non-empty finite
  `float32` in `[0, 255]`; RGB/RGBA are exact non-empty `uint8` ndarrays.
- Already-owned, read-only C-contiguous decoded RGBA is adopted without another
  copy; mutable, borrowed, or strided programmatic RGBA is defensively copied.
- No premultiplication or speculative source abstraction was introduced.

## Verification

```text
QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1
.venv/Scripts/python.exe -m pytest -q tests/test_source_rgba.py tests/test_preview_lifecycle.py --basetemp=.pytest-tmp-sm07-fix2
31 passed in 108.08s
```

The focused SM-07 suite has zero failures. The repository-wide known-red baseline
remains 71 unrelated kernel golden failures and was not changed by this task.
