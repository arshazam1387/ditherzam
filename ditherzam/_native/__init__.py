"""Private native-extension availability and fallback convention.

Public Python wrappers should validate inputs first, then call their native
module only when ``native_available()`` is true. Their Python reference path
must remain independently callable for differential tests.
"""
from __future__ import annotations

import os


_DISABLE_ENV = "DITHERZAM_DISABLE_NATIVE"

try:
    if os.environ.get(_DISABLE_ENV) == "1":
        raise ImportError("native extensions disabled by environment")
    from . import _smoke
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


__all__ = ["native_available", "smoke_add", "smoke_add_reference"]
