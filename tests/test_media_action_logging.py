from pathlib import Path

import numpy as np


def _preset():
    return {
        "adjustments": {"contrast": 50},
        "dither": {"style": "None", "scale": 1, "params": {}},
    }


def _source():
    gray = np.zeros((4, 4), np.float32)
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[..., 3] = 255
    return gray, rgba, None


def test_composition_logs_only_accepted_track_actions(qapp_fixture, monkeypatch):
    from ditherzam.dithering import registry
    from ditherzam.ui import composition_controller as module
    from ditherzam.ui.composition_panel import CompositionPanel

    events = []
    monkeypatch.setattr(
        module, "log_action",
        lambda action, **fields: events.append((action, fields)),
    )
    panel = CompositionPanel()
    controller = module.CompositionController(
        panel, registry, _preset, _source, lambda: 480, lambda _frame: None
    )

    controller.capture_current()
    controller.capture_current()
    controller.set_clip_duration(0, 12)
    controller.set_transition(1, "spatial-wipe", 4)
    controller.set_transition(0, "crossfade", 2)  # rejected
    controller.remove_look(1)

    assert [event[0] for event in events] == [
        "composition.look_captured",
        "composition.look_captured",
        "composition.clip_duration_changed",
        "composition.transition_changed",
        "composition.look_removed",
    ]
    assert events[2][1]["duration"] == 12
    assert events[3][1] == {"index": 1, "kind": "spatial-wipe", "duration": 4}


def test_composition_export_logs_terminal_result(qapp_fixture, monkeypatch, tmp_path):
    from ditherzam.dithering import registry
    from ditherzam.ui import composition_controller as module
    from ditherzam.ui.composition_panel import CompositionPanel

    events = []
    monkeypatch.setattr(
        module, "log_action",
        lambda action, **fields: events.append((action, fields)),
    )
    controller = module.CompositionController(
        CompositionPanel(), registry, _preset, _source, lambda: 480,
        lambda _frame: None,
    )

    assert controller.export_current(tmp_path / "missing.png") is None
    controller.capture_current()
    destination = controller.export_current(tmp_path / "frame.png")

    assert destination == Path(tmp_path / "frame.png")
    assert events[0] == (
        "composition.export_failed", {"reason": "no_composition"}
    )
    assert events[-1][0] == "composition.export_succeeded"
    assert events[-1][1]["path"] == destination


def test_timeline_logs_keyframe_and_playback_not_scrubbing(qapp_fixture, monkeypatch):
    from ditherzam.ui import timeline_panel as module

    events = []
    monkeypatch.setattr(
        module, "log_action",
        lambda action, **fields: events.append((action, fields)),
    )
    panel = module.TimelinePanel(length=5)
    panel.frame_slider.setValue(3)
    assert events == []

    panel.key_btn.click()
    panel.play_btn.setChecked(True)
    panel.play_btn.setChecked(False)

    assert [event[0] for event in events] == [
        "animation.keyframe_requested",
        "animation.playback_started",
        "animation.playback_stopped",
    ]
    assert all(event[1]["frame"] == 3 for event in events)


def test_animation_export_logs_success_failure_and_path(qapp_fixture, monkeypatch):
    from ditherzam.animation.timeline import Timeline
    from ditherzam.render import RenderSettings
    from ditherzam.ui import timeline_panel as module
    import ditherzam.animation as animation

    events = []
    monkeypatch.setattr(
        module, "log_action",
        lambda action, **fields: events.append((action, fields)),
    )
    panel = module.TimelinePanel(length=2)
    controller = module.AnimationController(
        panel, object(),
        lambda: (np.zeros((2, 2), np.float32), RenderSettings()),
        Timeline(length=2),
    )
    monkeypatch.setattr(animation, "export_animation", lambda *_a, **_k: "movie.mp4")

    assert controller.export("movie.mp4", fps=30) == "movie.mp4"
    assert [event[0] for event in events] == [
        "animation.export_started", "animation.export_succeeded"
    ]

    events.clear()
    controller.provide_base = lambda: None
    assert controller.export("missing.mp4") is None
    assert [event[0] for event in events] == [
        "animation.export_started", "animation.export_failed"
    ]
