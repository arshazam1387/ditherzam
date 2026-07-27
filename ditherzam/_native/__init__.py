"""Private native-extension availability and fallback convention.

Public Python wrappers should validate inputs first, then call their native
module only when ``native_available()`` is true. Their Python reference path
must remain independently callable for differential tests.
"""
from __future__ import annotations

import importlib
import os

import numpy as np


_DISABLE_ENV = "DITHERZAM_DISABLE_NATIVE"

try:
    if os.environ.get(_DISABLE_ENV) == "1":
        raise ImportError("native extensions disabled by environment")
    _smoke = importlib.import_module(f"{__name__}._smoke")
except ImportError:
    _smoke = None


def native_available() -> bool:
    """Return whether the smoke extension loaded in this interpreter."""
    return _smoke is not None


def smoke_add_reference(left: int, right: int) -> int:
    """Pure-Python reference used to prove the fallback convention."""
    return int(left) + int(right)


def smoke_add(left: int, right: int) -> int:
    """Exercise native dispatch, with an always-available Python fallback."""
    if _smoke is None:
        return smoke_add_reference(left, right)
    return _smoke.smoke_add(left, right)


def smoke_copy_u8_reference(source: np.ndarray) -> np.ndarray:
    """Pure-Python reference seam returning an owned C-contiguous uint8 array."""
    array = np.asarray(source)
    if array.dtype != np.uint8 or array.ndim != 3 or not array.flags.c_contiguous:
        raise ValueError("source must be a C-contiguous 3D uint8 array")
    return array.copy(order="C")


def smoke_copy_u8_native(source: np.ndarray) -> np.ndarray:
    """Native-only seam; never falls back, so differential tests stay honest."""
    if _smoke is None:
        raise RuntimeError("native smoke extension is unavailable")
    return _smoke.smoke_copy_u8(source)


def smoke_copy_u8(source: np.ndarray) -> np.ndarray:
    """Dispatch to native when available and otherwise retain exact fallback."""
    if _smoke is None:
        return smoke_copy_u8_reference(source)
    return smoke_copy_u8_native(source)


__all__ = [
    "native_available",
    "smoke_add",
    "smoke_add_reference",
    "smoke_copy_u8",
    "smoke_copy_u8_native",
    "smoke_copy_u8_reference",
]
