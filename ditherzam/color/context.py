from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from .palette import Palette
from .ramp import build_ramp


COLOR_CONTEXT_VERSION = 1
_LUM_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def color_context_key(palette: Palette, mode: str, depth: int,
                      mapping: str, phase: float) -> tuple:
    """Return a deterministic key for every input to derived color state."""
    colors = np.asarray(palette.colors)
    return (
        COLOR_CONTEXT_VERSION,
        colors.shape,
        colors.dtype.str,
        colors.tobytes(order="C"),
        str(mode),
        int(depth),
        str(mapping),
        float(phase),
    )


def _readonly(array: np.ndarray, *, dtype=None) -> np.ndarray:
    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ColorContext:
    key: tuple
    palette_colors: np.ndarray
    luminance_order: np.ndarray
    ramp: np.ndarray | None


class ColorContextCache:
    """Thread-safe cache of immutable palette-derived color data."""

    def __init__(self) -> None:
        self._contexts: dict[tuple, ColorContext] = {}
        self._lock = threading.Lock()

    def get(self, palette: Palette, mode: str, depth: int,
            mapping: str, phase: float) -> ColorContext:
        key = color_context_key(palette, mode, depth, mapping, phase)
        with self._lock:
            context = self._contexts.get(key)
            if context is None:
                colors = _readonly(palette.colors, dtype=np.float32)
                luminance = colors @ _LUM_W
                order = _readonly(np.argsort(luminance, kind="stable"), dtype=np.int64)
                ramp = None
                if mode == "ramp":
                    ramp = _readonly(
                        build_ramp(
                            Palette("context", colors), depth, mapping, phase
                        ),
                        dtype=np.float32,
                    )
                context = ColorContext(key, colors, order, ramp)
                self._contexts[key] = context
            return context


DEFAULT_COLOR_CONTEXT_CACHE = ColorContextCache()
