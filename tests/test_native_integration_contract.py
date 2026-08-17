"""Cross-module native availability, fallback, and thread-policy contracts."""
import importlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
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


def test_setter_overlap_cannot_change_pixels_or_restore_stale_budget():
    module = importlib.import_module("ditherzam._native._composite")
    rng = np.random.default_rng(20260727)
    first = rng.integers(0, 256, (512, 512, 4), dtype=np.uint8)
    second = rng.integers(0, 256, (512, 512, 4), dtype=np.uint8)
    module.set_thread_budget(2)
    expected = module.blend_layer_u8(first, second, 3, 73)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(module.blend_layer_u8, first, second, 3, 73)
        for budget in (1, 8, 4, 2):
            module.set_thread_budget(budget)
        actual = future.result()
    assert np.array_equal(actual, expected)
    assert module.get_thread_budget() == 2


def test_no_temporary_native_budget_context_is_exposed():
    assert not hasattr(threading_policy, "native_threads")
