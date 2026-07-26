import numpy as np


def _preset(contrast=50):
    return {
        "adjustments": {"contrast": contrast},
        "dither": {"style": "None", "scale": 1, "params": {}},
    }


def _source():
    gray = np.zeros((4, 4), np.float32)
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[..., 3] = 255
    return gray, rgba, None


def test_capture_rebuilds_contiguous_composition(qapp_fixture):
    from ditherzam.dithering import registry
    from ditherzam.ui.composition_controller import CompositionController
    from ditherzam.ui.composition_panel import CompositionPanel

    current = {"preset": _preset(20)}
    panel = CompositionPanel()
    controller = CompositionController(
        panel, registry, lambda: current["preset"], _source,
        lambda: 480, lambda _frame: None,
    )
    controller.capture_current()
    current["preset"] = _preset(80)
    controller.capture_current()
    comp = controller.composition
    assert [(c.look.name, c.start, c.end) for c in comp.clips] == [
        ("Look 1", 0, 24), ("Look 2", 24, 48),
    ]
    assert comp.transitions[1].kind == "crossfade"
    assert comp.transitions[1].duration == 6


def test_remove_and_duration_edit_keep_track_contiguous(qapp_fixture):
    from ditherzam.dithering import registry
    from ditherzam.ui.composition_controller import CompositionController
    from ditherzam.ui.composition_panel import CompositionPanel

    panel = CompositionPanel()
    controller = CompositionController(
        panel, registry, lambda: _preset(), _source,
        lambda: 480, lambda _frame: None,
    )
    controller.capture_current()
    controller.capture_current()
    controller.set_clip_duration(0, 10)
    assert [(c.start, c.end) for c in controller.composition.clips] == [(0, 10), (10, 34)]
    controller.remove_look(0)
    assert [(c.start, c.end) for c in controller.composition.clips] == [(0, 24)]


def test_preview_without_source_reports_inline_error(qapp_fixture):
    from ditherzam.dithering import registry
    from ditherzam.ui.composition_controller import CompositionController
    from ditherzam.ui.composition_panel import CompositionPanel

    panel = CompositionPanel()
    controller = CompositionController(
        panel, registry, lambda: _preset(), lambda: None,
        lambda: 480, lambda _frame: None,
    )
    controller.capture_current()
    controller.request_frame(0)
    assert "image" in panel.status_label.text().lower()
    assert panel.status_label.property("error") is True
