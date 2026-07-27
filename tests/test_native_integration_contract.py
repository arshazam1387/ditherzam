"""Cross-module native availability, fallback, and thread-policy contracts."""
import importlib

import pytest

from ditherzam import threading_policy
from ditherzam._native import native_available


NATIVE_MODULES = ("_smoke", "_composite", "_selection", "_brush")


@pytest.mark.parametrize("name", NATIVE_MODULES)
def test_enabled_native_module_imports(name):
    if not native_available():
        pytest.skip("native extensions are unavailable")
    assert importlib.import_module(f"ditherzam._native.{name}") is not None


def test_native_compositor_thread_budget_round_trip():
    if not native_available():
        pytest.skip("native extensions are unavailable")
    previous = threading_policy.get_native_threads()
    try:
        for threads in (1, 2, 4, 8):
            assert threading_policy.set_native_threads(threads) == threads
            assert threading_policy.get_native_threads() == threads
    finally:
        threading_policy.set_native_threads(previous)


def test_selection_and_brush_remain_without_thread_setters():
    for name in ("_selection", "_brush"):
        module = importlib.import_module(f"ditherzam._native.{name}")
        assert not hasattr(module, "set_thread_budget")
