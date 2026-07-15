"""Task 3.2: tonal stages (contrast/midtones/highlights) must share ONE private
float32 buffer per render() / render_cached() call instead of each allocating.
These tests fail against the pre-fusion render.py (no ``out=`` routing) and
pass once render.py routes the three calls through a single buffer."""
import numpy as np
from unittest import mock

import ditherzam.render as R
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings


def _capture(calls, name):
    def _rec(*args, **kwargs):
        calls.append((name, kwargs.get("out")))
        return mock.DEFAULT
    return _rec


def test_render_reuses_one_buffer_across_tonal_stages():
    calls: list[tuple[str, object]] = []
    p = RenderPipeline(registry)
    with mock.patch.object(R, "apply_contrast", side_effect=_capture(calls, "contrast"), wraps=R.apply_contrast), \
         mock.patch.object(R, "apply_midtones", side_effect=_capture(calls, "midtones"), wraps=R.apply_midtones), \
         mock.patch.object(R, "apply_highlights", side_effect=_capture(calls, "highlights"), wraps=R.apply_highlights):
        p.render(np.full((8, 8), 100.0, np.float32), RenderSettings())

    assert [c[0] for c in calls] == ["contrast", "midtones", "highlights"]
    outs = [c[1] for c in calls]
    assert all(o is not None for o in outs), "tonal stages must be routed through out="
    assert outs[0] is outs[1] is outs[2], "all three stages must share ONE buffer object"


def test_render_cached_l1_reuses_one_buffer_across_tonal_stages():
    calls: list[tuple[str, object]] = []
    p = RenderPipeline(registry)
    base = np.full((8, 8), 100.0, np.float32)
    with mock.patch.object(R, "apply_contrast", side_effect=_capture(calls, "contrast"), wraps=R.apply_contrast), \
         mock.patch.object(R, "apply_midtones", side_effect=_capture(calls, "midtones"), wraps=R.apply_midtones), \
         mock.patch.object(R, "apply_highlights", side_effect=_capture(calls, "highlights"), wraps=R.apply_highlights):
        p.render_cached(base, RenderSettings())

    assert [c[0] for c in calls] == ["contrast", "midtones", "highlights"]
    outs = [c[1] for c in calls]
    assert all(o is not None for o in outs)
    assert outs[0] is outs[1] is outs[2]


def test_render_does_not_mutate_source_array():
    base = np.linspace(0, 255, 64, dtype=np.float32).reshape(8, 8)
    original = base.copy()
    p = RenderPipeline(registry)
    p.render(base, RenderSettings(contrast=70, midtones=30, highlights=80))
    np.testing.assert_array_equal(base, original)


def test_render_cached_does_not_mutate_source_array():
    base = np.linspace(0, 255, 64, dtype=np.float32).reshape(8, 8)
    original = base.copy()
    p = RenderPipeline(registry)
    p.render_cached(base, RenderSettings(contrast=70, midtones=30, highlights=80))
    np.testing.assert_array_equal(base, original)


def test_render_cached_l1_buffer_is_private_per_settings_change():
    # The buffer stored under cache key "g" must be this call's own private
    # array -- a later settings change must not corrupt an earlier snapshot.
    p = RenderPipeline(registry)
    rng = np.random.default_rng(1)
    base = rng.uniform(0, 255, (16, 16)).astype(np.float32)

    p.render_cached(base, RenderSettings(contrast=70, midtones=30, highlights=80, style="None"))
    g1 = p._cache.get(id(base))["g"]
    g1_snapshot = g1.copy()

    p.render_cached(base, RenderSettings(contrast=20, midtones=80, highlights=10, style="None"))
    g2 = p._cache.get(id(base))["g"]

    assert g2 is not g1
    np.testing.assert_array_equal(g1, g1_snapshot)
