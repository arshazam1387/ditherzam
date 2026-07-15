"""Warmup must run headlessly without error and leave renders working."""
import numpy as np

from ditherzam.warmup import (
    warmup_render, start_warmup_thread, DEFAULT_WARMUP_COLOR_MODES)
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
import ditherzam.render as _render_mod


def test_warmup_render_completes():
    warmup_render()  # should not raise


def test_warmup_render_tolerates_unknown_styles():
    warmup_render(styles=("No Such Style", "Floyd-Steinberg"))


def test_render_works_after_warmup():
    warmup_render()
    pipe = RenderPipeline(registry)
    out = pipe.render(np.full((16, 16), 100.0, np.float32),
                      RenderSettings(style="Floyd-Steinberg", scale=2))
    assert out.shape == (16, 16, 3) and out.dtype == np.uint8


def test_start_warmup_thread_joins():
    t = start_warmup_thread()
    t.join(timeout=30)
    assert not t.is_alive()


def test_default_warmup_color_modes_cover_selected_kernels():
    # The selected color paths whose Numba kernels are otherwise cold on first
    # real use (Task 4.5): nearest/ordered/ramp/diffused.
    assert set(DEFAULT_WARMUP_COLOR_MODES) == {
        "nearest", "ordered", "ramp", "diffused"}


def test_warmup_dispatches_each_selected_color_path(monkeypatch):
    calls = []
    orig = _render_mod.RenderPipeline.render

    def spy(self, gray, settings, *a, **k):
        calls.append((self.color_engine.mode, settings.saturation))
        return orig(self, gray, settings, *a, **k)

    monkeypatch.setattr(_render_mod.RenderPipeline, "render", spy)
    warmup_render(styles=())  # isolate the color-path warmup

    modes = {mode for mode, _ in calls}
    assert {"nearest", "ordered", "ramp", "diffused"} <= modes
    # a non-neutral saturation pass compiles apply_saturation's real path
    assert any(sat != 50 for _, sat in calls)


def test_warmup_thread_terminates_best_effort_on_kernel_error(monkeypatch):
    # Best-effort: a failure inside a warmed render must not escape the thread.
    def boom(self, *a, **k):
        raise RuntimeError("kernel exploded")

    monkeypatch.setattr(_render_mod.RenderPipeline, "render", boom)
    t = start_warmup_thread()
    t.join(timeout=30)
    assert not t.is_alive()  # swallowed, thread exits cleanly
