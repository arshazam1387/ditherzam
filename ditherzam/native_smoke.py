"""Small deterministic smoke for development and frozen standard builds."""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _digest(array) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def run() -> int:
    from ditherzam._native import _brush, _composite, _selection, native_available
    from ditherzam.layers.mask_brush import (
        BrushMode, BrushSettings, _stamp_mask_brush_native,
    )

    if not native_available():
        raise RuntimeError("native extensions are unavailable")
    rgba_a = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
    rgba_b = np.flip(rgba_a, axis=1).copy()
    selection = np.arange(16, dtype=np.uint8).reshape(4, 4)
    mapped = _selection.selection_to_source_u8(selection, 4, 4, 0., 0., 1., 1.)
    composed = _composite.blend_layer_u8(rgba_a, rgba_b, 3, 73)
    transitioned = _composite.blend_transition_scalar_u8(rgba_a, rgba_b, 0.37)
    mask = np.zeros((16, 16), dtype=np.uint8)
    brush_result = _stamp_mask_brush_native(
        mask, 8.0, 8.0, BrushSettings(8.0, 50, 100, BrushMode.REVEAL)
    )
    print(json.dumps({
        "native_available": True,
        "threads": _composite.get_thread_budget(),
        "compositor": _digest(composed),
        "transition": _digest(transitioned),
        "selection": _digest(mapped),
        "brush": _digest(mask),
        "brush_result": None if brush_result is None else [
            brush_result.x0, brush_result.y0, brush_result.x1, brush_result.y1
        ],
    }, sort_keys=True))
    return 0
