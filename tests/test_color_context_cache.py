"""Contract tests for immutable, content-keyed color-derived state.

The public API proposed by Task 2.7 keeps ``ColorEngine`` source-compatible while
letting engines share immutable palette-derived data through ``ColorContextCache``.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from ditherzam.color.context import ColorContextCache, color_context_key
from ditherzam.color.engine import ColorEngine
from ditherzam.color.palette import Palette


def _palette(name="demo", colors=None):
    if colors is None:
        colors = [[0, 0, 0], [80, 130, 190], [255, 255, 255]]
    return Palette.from_list(name, colors)


def _engine(cache, palette=None, **changes):
    params = dict(mode="ramp", depth=3, mapping="match", phase=0.0)
    params.update(changes)
    return ColorEngine(palette or _palette(), context_cache=cache, **params)


def test_content_key_ignores_palette_name():
    a = _palette("first")
    b = _palette("renamed")

    assert color_context_key(a, "ramp", 3, "match", 0.0) == color_context_key(
        b, "ramp", 3, "match", 0.0
    )


@pytest.mark.parametrize(
    "edited",
    [
        [[0, 0, 0], [81, 130, 190], [255, 255, 255]],
        [[255, 255, 255], [80, 130, 190], [0, 0, 0]],
    ],
    ids=["color-edit", "order-edit"],
)
def test_same_name_palette_content_or_order_edit_misses(edited):
    original = _palette("same")
    changed = _palette("same", edited)

    assert color_context_key(original, "ramp", 3, "match", 0.0) != (
        color_context_key(changed, "ramp", 3, "match", 0.0)
    )


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "nearest"},
        {"depth": 4},
        {"mapping": "reverse"},
        {"phase": 0.25},
    ],
    ids=["mode", "depth", "mapping", "phase"],
)
def test_algorithm_inputs_invalidate_context(change):
    cache = ColorContextCache()
    base = _engine(cache)
    changed = base.with_settings(**change)

    assert changed.context is not base.context


def test_identical_content_reuses_context_across_engines_and_renames():
    cache = ColorContextCache()
    a = _engine(cache, _palette("one"))
    b = _engine(cache, _palette("two"))

    assert a is not b
    assert a.context is b.context


def test_derived_context_arrays_are_read_only():
    context = _engine(ColorContextCache()).context

    arrays = [context.palette_colors, context.luminance_order]
    if context.ramp is not None:
        arrays.append(context.ramp)

    assert arrays
    for array in arrays:
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array.flat[0] = 123


def test_with_settings_returns_new_engine_without_mutating_shared_engine():
    cache = ColorContextCache()
    base = _engine(cache)
    changed = base.with_settings(depth=5, mapping="reverse", phase=0.5)

    assert (base.depth, base.mapping, base.phase) == (3, "match", 0.0)
    assert (changed.depth, changed.mapping, changed.phase) == (5, "reverse", 0.5)
    assert changed is not base


def test_concurrent_derived_engines_do_not_race_on_shared_mutable_fields():
    cache = ColorContextCache()
    base = _engine(cache)
    low = base.with_settings(depth=2, mapping="match", phase=0.0)
    high = base.with_settings(depth=6, mapping="reverse", phase=0.25)
    image = np.linspace(0, 255, 1024, dtype=np.float32).reshape(32, 32)
    expected_low = low.map(image)
    expected_high = high.map(image)

    jobs = [low, high] * 12
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda engine: engine.map(image), jobs))

    for engine, output in zip(jobs, outputs):
        expected = expected_low if engine is low else expected_high
        np.testing.assert_array_equal(output, expected)
    assert (base.depth, base.mapping, base.phase) == (3, "match", 0.0)
