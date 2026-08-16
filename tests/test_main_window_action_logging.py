import numpy as np


def _editor(monkeypatch):
    import ditherzam.ui.main_window as module

    events = []
    monkeypatch.setattr(
        module,
        "log_action",
        lambda action, **fields: events.append((action, fields)),
    )
    window = module.ImageEditor()
    gray = np.zeros((4, 5), np.float32)
    rgba = np.zeros((4, 5, 4), np.uint8)
    rgba[..., 3] = 255
    window.load_array(gray, rgba[..., :3], rgba)
    events.clear()
    monkeypatch.setattr(window, "_launch_worker", lambda _request: None)
    monkeypatch.setattr(
        window.layers_controller, "request_preview", lambda *_args: None)
    return window, events


def test_slider_burst_logs_one_settled_control_action(qapp_fixture, monkeypatch):
    window, events = _editor(monkeypatch)

    window.panel.contrast_slider.setValue(60)
    window.panel.contrast_slider.setValue(70)
    window.panel.contrast_slider.setValue(80)
    assert events == []

    window._do_full_render()
    window._do_full_render()

    assert [action for action, _fields in events] == [
        "controls.settled_change"
    ]
    changes = events[0][1]["changes"]
    assert changes["contrast"] == {"from": 50, "to": 80}


def test_preview_policy_actions_are_semantic_not_render_noise(
    qapp_fixture, monkeypatch
):
    window, events = _editor(monkeypatch)

    window._set_preview_resolution("720p")
    window._set_rerender_on_zoom(True)

    assert events == [
        ("preview.resolution_changed", {"resolution": "720p"}),
        ("preview.zoom_rerender_changed", {"enabled": True}),
    ]


def test_decode_request_relies_on_core_path_redaction(qapp_fixture, monkeypatch):
    window, events = _editor(monkeypatch)
    captured = []
    monkeypatch.setattr(window._pool, "start", captured.append)

    window._start_layer_decode(
        r"C:\Users\arsha\private\subject.png",
        intent="place",
    )

    assert events == [
        (
            "image.decode_requested",
            {
                "path": r"C:\Users\arsha\private\subject.png",
                "intent": "place",
            },
        )
    ]
    assert len(captured) == 1
