"""Regression: the ControlPanel's Color (palette/mode) and Effects selections must
actually reach the render pipeline. Previously the pipeline was built once with
color_engine=None/effect_stack=None and never refreshed, so those controls did
nothing (bug: 'doesn't apply any effects')."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from ditherzam.ui.main_window import ImageEditor
from ditherzam.color.palette import builtin_palettes


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _gradient(h=60, w=80):
    xx = np.linspace(0, 255, w, dtype=np.float32)[None, :].repeat(h, 0)
    return xx.copy()


def test_palette_and_mode_are_applied(app):
    win = ImageEditor()
    win.load_array(_gradient())
    # user picks a Game Boy palette in nearest mode + a dither style
    win.panel.set_style("Floyd-Steinberg")
    win.panel._on_palette_changed("gameboy")
    win.panel.state["color_mode"] = "nearest"
    win.render_now()  # exercises the viewport path (must not raise, must sync pipeline)

    out = win._rendered_rgb()
    colors = set(map(tuple, out.reshape(-1, 3).tolist()))
    palette = {tuple(c) for c in builtin_palettes()["gameboy"].colors.astype("uint8").tolist()}
    assert colors, "no pixels rendered"
    assert colors <= palette, f"non-palette colors leaked: {colors - palette}"


def test_default_state_does_not_blur_the_image(app):
    """With nothing touched (Style='None', all adjustments at their neutral), the
    preview must equal the source. Regression: Blur defaulted to 50, which is a
    25px Gaussian blur applied to every render (blur's neutral is 0, not 50)."""
    win = ImageEditor()
    # a sharp image: any blur destroys the hard black/white edges
    sharp = np.zeros((40, 60), dtype=np.float32)
    sharp[:, ::2] = 255.0  # 1px vertical stripes
    win.load_array(sharp)

    out = win._rendered_rgb()  # Style='None' default, nothing touched
    # None style + neutral adjustments => output is the source broadcast to RGB
    assert np.array_equal(out[..., 0], sharp.astype("uint8")), \
        "default render altered a sharp image (blur applied by default?)"


def test_export_pipeline_snapshots_and_isolates_from_later_edits(app):
    """A dedicated export pipeline must capture color+effects at launch and stay
    immutable when the user edits afterward (Task 4.2)."""
    win = ImageEditor()
    win.load_array(_gradient())
    win.panel.set_style("Floyd-Steinberg")
    win.panel._on_palette_changed("gameboy")
    win.panel.state["color_mode"] = "nearest"

    exp = win._export_pipeline()
    assert exp is not win.pipeline                      # dedicated context
    assert exp.color_engine is not None
    assert exp.color_engine.mode == "nearest"
    assert exp.effect_stack is None

    # user changes mode + adds an effect after the snapshot is taken
    win.panel.state["color_mode"] = "ordered"
    win.panel.state["effects"] = ["Chromatic Aberration"]
    win._sync_pipeline()                                 # mutates the live pipeline

    assert exp.color_engine.mode == "nearest"           # snapshot untouched
    assert exp.effect_stack is None
    exp2 = win._export_pipeline()                        # a fresh snapshot moves on
    assert exp2.color_engine.mode == "ordered"
    assert exp2.effect_stack is not None


def test_adding_effect_changes_output_without_crashing(app):
    win = ImageEditor()
    win.load_array(_gradient())
    win.panel.set_style("Floyd-Steinberg")

    before = win._rendered_rgb().copy()

    # user adds a post effect via the Effects panel
    win.panel.state["effects"] = ["Chromatic Aberration"]
    win.render_now()  # must not crash the render
    after = win._rendered_rgb()

    assert not np.array_equal(before, after), "adding an effect did not change the render"
