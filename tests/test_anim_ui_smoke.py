import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication
from ditherzam.animation.temporal import PATTERNS
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.animation.timeline import Timeline
from ditherzam.ui.timeline_panel import TimelinePanel, AnimationController

_app = QApplication.instance() or QApplication([])


def test_panel_lists_none_plus_nine_patterns():
    p = TimelinePanel(length=10)
    items = [p.pattern_combo.itemText(i) for i in range(p.pattern_combo.count())]
    assert items[0] == "none"
    assert tuple(items[1:]) == PATTERNS
    assert len(items) == 10


def test_panel_signals_exist():
    p = TimelinePanel()
    for sig in ("keyframe_requested", "frame_changed", "export_requested", "play_toggled"):
        assert hasattr(p, sig)


def test_length_updates_frame_slider_range():
    p = TimelinePanel(length=10)
    p.length_spin.setValue(20)
    assert p.frame_slider.maximum() == 19


def test_play_toggle_drives_timer():
    p = TimelinePanel(length=5)
    p.play_btn.setChecked(True)
    assert p._timer.isActive()
    p.play_btn.setChecked(False)
    assert not p._timer.isActive()


def test_controller_renders_frame_to_sink():
    p = TimelinePanel(length=5)
    p.pattern_combo.setCurrentText("static")
    p.amp_slider.setValue(60)
    pipeline = RenderPipeline(registry)
    tl = Timeline(length=5)
    base = np.full((16, 16), 128.0, np.float32)
    ctrl = AnimationController(
        p, pipeline, lambda: (base, RenderSettings(style="Bayer-Matrix 4x4", scale=1)),
        tl, seed=0)
    captured = {}
    ctrl.on_frame = lambda img: captured.setdefault("img", img)
    out = ctrl.render_frame(2)
    assert out.shape == (16, 16, 3) and out.dtype == np.uint8
    assert captured["img"].shape == (16, 16, 3)
