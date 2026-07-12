"""Task 2.3: unified capped lifecycle (drag/settle/full) + async initial decode.

Covers: settled renders use the same preview-preference policy cap as drag
(scaled by ``_proxy_max_side``), the Full Quality Preview action forces one
uncapped exact render and makes the next settle tick exact too, any further
edit returns to the capped policy, and the initial image drop decodes off the
GUI thread instead of blocking on a synchronous ``render_now()``.
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


def _editor(resolution="1080", proxy_max_side=640):
    from ditherzam.ui.main_window import ImageEditor
    from ditherzam.ui.preview_preferences import PreviewPreferences

    editor = ImageEditor(preference_store=FakeSettings(), proxy_max_side=proxy_max_side)
    editor.preview_preferences = PreviewPreferences(
        resolution, editor.preview_preferences.rerender_on_zoom
    )
    # a wide source: longest side 4000, so caps are unambiguous
    editor.load_array(np.zeros((10, 4000), dtype=np.float32))
    return editor


# ---- capping architecture ---------------------------------------------------

def test_settle_build_uses_numeric_policy_cap(qapp_fixture):
    editor = _editor(resolution="1080")
    req = editor._build_request(RenderKind.SETTLE)
    assert req.target_max_side == 1080
    assert req.mode == "proxy"


def test_settle_build_uses_full_policy_cap(qapp_fixture):
    editor = _editor(resolution="Full")
    req = editor._build_request(RenderKind.SETTLE)
    assert req.target_max_side == 4000  # source longest side, uncapped


def test_full_kind_is_uncapped_and_exact_mode_regardless_of_preference(qapp_fixture):
    editor = _editor(resolution="1080")
    req = editor._build_request(RenderKind.FULL)
    assert req.target_max_side == 4000
    assert req.mode == "full"


def test_drag_target_is_min_of_proxy_and_policy_cap(qapp_fixture):
    # policy cap (1080) exceeds proxy_max_side (640) -> proxy wins
    tight_proxy = _editor(resolution="1080", proxy_max_side=640)
    assert tight_proxy._build_request(RenderKind.DRAG).target_max_side == 640

    # policy cap (480) is tighter than proxy_max_side (640) -> policy wins
    tight_policy = _editor(resolution="480", proxy_max_side=640)
    assert tight_policy._build_request(RenderKind.DRAG).target_max_side == 480


# ---- Full Quality Preview action -------------------------------------------

def test_full_preview_flag_makes_settle_tick_build_full_request(qapp_fixture, monkeypatch):
    editor = _editor(resolution="1080")
    editor._full_preview_requested = True
    launched = []
    monkeypatch.setattr(editor, "_launch_worker", launched.append)

    editor._do_full_render()  # the settle-timer tick

    assert len(launched) == 1
    assert launched[0].kind is RenderKind.FULL
    assert launched[0].target_max_side == 4000
    assert launched[0].mode == "full"


def test_full_quality_preview_action_sets_flag_and_schedules_full(qapp_fixture, monkeypatch):
    editor = _editor(resolution="1080")
    assert editor._full_preview_requested is False
    launched = []
    monkeypatch.setattr(editor, "_launch_worker", launched.append)

    editor._actions["full_quality_preview"].trigger()

    assert editor._full_preview_requested is True
    assert len(launched) == 1
    assert launched[0].kind is RenderKind.FULL


def test_full_quality_preview_action_is_in_view_menu(qapp_fixture):
    editor = _editor(resolution="1080")
    assert editor._actions["full_quality_preview"] in editor.view_menu.actions()


def test_schedule_render_resets_full_preview_flag(qapp_fixture):
    editor = _editor(resolution="1080")
    editor._full_preview_requested = True
    editor.schedule_render()
    assert editor._full_preview_requested is False


# ---- asynchronous initial decode -------------------------------------------

def test_image_decoded_slot_loads_array_and_schedules_without_render_now(qapp_fixture, monkeypatch):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor(preference_store=FakeSettings())
    calls = {"render_now": 0, "schedule_render": 0, "replace": []}
    monkeypatch.setattr(editor, "render_now",
                        lambda: calls.__setitem__("render_now", calls["render_now"] + 1))
    monkeypatch.setattr(editor, "schedule_render",
                        lambda: calls.__setitem__("schedule_render", calls["schedule_render"] + 1))
    monkeypatch.setattr(
        editor, "_replace_source_arrays",
        lambda g, r, a, *, adopt_decoded_rgba: calls["replace"].append(
            (g, r, a, adopt_decoded_rgba)))

    gray = np.zeros((8, 8), dtype=np.float32)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    editor._on_image_decoded(gray, rgb, rgba)  # the GUI-thread slot, invoked directly

    assert calls["render_now"] == 0
    assert calls["schedule_render"] == 1
    assert calls["replace"] == [(gray, rgb, rgba, True)]


def test_image_dropped_runs_decode_off_thread_pool_not_sync_render(qapp_fixture, monkeypatch, tmp_path):
    from ditherzam.ui.main_window import ImageEditor
    from PIL import Image

    img_path = tmp_path / "drop.png"
    Image.new("RGB", (4, 4), (10, 20, 30)).save(img_path)

    editor = ImageEditor(preference_store=FakeSettings())
    calls = {"render_now": 0, "schedule_render": 0}
    monkeypatch.setattr(editor, "render_now",
                        lambda: calls.__setitem__("render_now", calls["render_now"] + 1))
    monkeypatch.setattr(editor, "schedule_render",
                        lambda: calls.__setitem__("schedule_render", calls["schedule_render"] + 1))
    # Run the queued worker synchronously in-process -- deterministic, no real
    # thread/timing dependency -- instead of a real QThreadPool thread.
    monkeypatch.setattr(editor._pool, "start", lambda worker: worker.run())

    editor._on_image_dropped(str(img_path))

    assert calls["render_now"] == 0
    assert calls["schedule_render"] == 1
    assert editor._base_gray is not None
    assert editor._base_rgba is not None
