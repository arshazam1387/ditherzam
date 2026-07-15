"""Task 2.4: optional bucketed zoom refinement + Part A refit-tracking regression.

Covers:
- rerender_on_zoom Off schedules nothing.
- On + a zoom crossing a bucket schedules exactly one ZOOM request with the
  expected ``target_max_side``.
- A second zoom within the same bucket does not reschedule (once-per-bucket).
- Demand beyond the ceiling clamps to the ceiling (Auto -> 1440, Full -> source).
- Zooming back out to baseline resets ``_last_zoom_bucket`` so a later re-zoom
  schedules again.
- Part A: an ordinary SETTLE/ZOOM paint must not refit the viewport; only the
  first paint after ``load_array`` may.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from ditherzam.ui.render_request import RenderKind


class FakeSettings:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def value(self, key, defaultValue=None):
        return self.values.get(key, defaultValue)

    def setValue(self, key, value):
        self.values[key] = value


def _editor(resolution="Auto", rerender_on_zoom=False, longest=3840):
    from ditherzam.ui.main_window import ImageEditor
    from ditherzam.ui.preview_preferences import PreviewPreferences

    editor = ImageEditor(preference_store=FakeSettings())
    editor.preview_preferences = PreviewPreferences(resolution, rerender_on_zoom)
    # a wide source: longest side == `longest`, so caps are unambiguous
    editor.load_array(np.zeros((10, longest), dtype=np.float32))
    return editor


def _capture_launches(editor, monkeypatch):
    launched = []
    monkeypatch.setattr(editor, "_launch_worker", launched.append)
    return launched


# ---- Part B: bucketed zoom refinement --------------------------------------

def test_zoom_off_schedules_nothing(qapp_fixture, monkeypatch):
    editor = _editor(resolution="Auto", rerender_on_zoom=False)
    launched = _capture_launches(editor, monkeypatch)
    monkeypatch.setattr(editor, "_zoom_required_pixels", lambda: 2000)

    editor._on_zoom_debounced()

    assert launched == []


def test_zoom_in_crossing_bucket_schedules_one_zoom_request(qapp_fixture, monkeypatch):
    editor = _editor(resolution="Auto", rerender_on_zoom=True)  # ceiling 1440, baseline == _policy_cap()
    launched = _capture_launches(editor, monkeypatch)
    baseline = editor._policy_cap()
    monkeypatch.setattr(editor, "_zoom_required_pixels", lambda: baseline + 1)

    editor._on_zoom_debounced()

    assert len(launched) == 1
    req = launched[0]
    assert req.kind is RenderKind.ZOOM
    from ditherzam.ui.preview import zoom_preview_bucket
    expected = zoom_preview_bucket(baseline, baseline + 1, 1440, 3840)
    assert req.target_max_side == expected
    assert expected > baseline
    assert editor._last_zoom_bucket == expected


def test_second_zoom_within_same_bucket_does_not_reschedule(qapp_fixture, monkeypatch):
    editor = _editor(resolution="Auto", rerender_on_zoom=True)
    launched = _capture_launches(editor, monkeypatch)
    baseline = editor._policy_cap()
    monkeypatch.setattr(editor, "_zoom_required_pixels", lambda: baseline + 1)

    editor._on_zoom_debounced()
    assert len(launched) == 1

    # still zoomed within the same bucket -> no new request
    editor._on_zoom_debounced()
    assert len(launched) == 1


def test_zoom_demand_beyond_ceiling_clamps_to_auto_ceiling(qapp_fixture, monkeypatch):
    editor = _editor(resolution="Auto", rerender_on_zoom=True)
    launched = _capture_launches(editor, monkeypatch)
    monkeypatch.setattr(editor, "_zoom_required_pixels", lambda: 10_000)

    editor._on_zoom_debounced()

    assert len(launched) == 1
    assert launched[0].target_max_side == 1440
    assert editor._last_zoom_bucket == 1440


def test_zoom_demand_beyond_ceiling_clamps_to_full_source(qapp_fixture, monkeypatch):
    # In Full mode the settled baseline already IS the source-longest ceiling
    # (preview_cap("Full", ...) == source_longest), so no zoom demand -- however
    # large -- can push a bucket past it: nothing new is ever scheduled.
    editor = _editor(resolution="Full", rerender_on_zoom=True, longest=3840)
    launched = _capture_launches(editor, monkeypatch)
    baseline = editor._policy_cap()
    assert baseline == 3840
    monkeypatch.setattr(editor, "_zoom_required_pixels", lambda: 10_000)

    editor._on_zoom_debounced()

    assert launched == []
    assert editor._last_zoom_bucket is None


def test_zoom_back_out_resets_last_bucket_and_allows_rescheduling(qapp_fixture, monkeypatch):
    editor = _editor(resolution="Auto", rerender_on_zoom=True)
    launched = _capture_launches(editor, monkeypatch)
    baseline = editor._policy_cap()

    monkeypatch.setattr(editor, "_zoom_required_pixels", lambda: baseline + 1)
    editor._on_zoom_debounced()
    assert len(launched) == 1
    assert editor._last_zoom_bucket is not None
    editor._scheduler.on_finished()  # simulate the render completing (frees the scheduler)

    # zoom back out to baseline demand
    monkeypatch.setattr(editor, "_zoom_required_pixels", lambda: baseline)
    editor._on_zoom_debounced()
    assert editor._last_zoom_bucket is None
    assert len(launched) == 1  # no new request just for zooming back out

    # re-zoom past the bucket again -> schedules again
    monkeypatch.setattr(editor, "_zoom_required_pixels", lambda: baseline + 1)
    editor._on_zoom_debounced()
    assert len(launched) == 2


def test_zoom_handler_noop_without_image(qapp_fixture, monkeypatch):
    from ditherzam.ui.main_window import ImageEditor
    editor = ImageEditor(preference_store=FakeSettings())
    from ditherzam.ui.preview_preferences import PreviewPreferences
    editor.preview_preferences = PreviewPreferences("Auto", True)
    launched = _capture_launches(editor, monkeypatch)

    editor._on_zoom_debounced()

    assert launched == []


def test_schedule_render_resets_last_zoom_bucket(qapp_fixture):
    editor = _editor(resolution="Auto", rerender_on_zoom=True)
    editor._last_zoom_bucket = 1440
    editor.schedule_render()
    assert editor._last_zoom_bucket is None


def test_zoom_changed_signal_starts_debounce_timer(qapp_fixture):
    editor = _editor(resolution="Auto", rerender_on_zoom=True)
    editor._zoom_debounce.stop()
    editor.viewport.zoom_changed.emit(150)
    assert editor._zoom_debounce.isActive()


# ---- Part A: paint refit-tracking regression -------------------------------

def test_settle_paint_does_not_refit(qapp_fixture):
    from ditherzam.ui.render_request import RenderRequest

    editor = _editor(resolution="1080", rerender_on_zoom=False)
    editor.render_now()  # first paint fits
    editor.viewport.zoom_in()
    before = editor.viewport.transform()

    req = editor._build_request(RenderKind.SETTLE)
    from PySide6.QtGui import QImage
    qimg = QImage(4, 4, QImage.Format.Format_RGB32)
    qimg.fill(0)
    # forge a current request so is_current() is true
    stamped = editor._scheduler.request(req)
    assert stamped is not None
    editor._on_rendered(qimg, stamped)

    assert editor.viewport.transform() == before


def test_zoom_kind_paint_does_not_refit(qapp_fixture):
    editor = _editor(resolution="1080", rerender_on_zoom=False)
    editor.render_now()
    editor.viewport.zoom_in()
    before = editor.viewport.transform()

    req = editor._build_request(RenderKind.ZOOM, target_max_side=1440)
    from PySide6.QtGui import QImage
    qimg = QImage(4, 4, QImage.Format.Format_RGB32)
    qimg.fill(0)
    stamped = editor._scheduler.request(req)
    assert stamped is not None
    editor._on_rendered(qimg, stamped)

    assert editor.viewport.transform() == before


def test_load_array_then_render_now_fits(qapp_fixture):
    editor = _editor(resolution="1080", rerender_on_zoom=False)
    editor.render_now()
    editor.viewport.zoom_in()
    zoomed = editor.viewport.transform().m11()

    editor.load_array(np.zeros((10, 1000), dtype=np.float32))
    editor.render_now()

    assert editor.viewport.transform().m11() != pytest.approx(zoomed)
