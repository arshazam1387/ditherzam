import numpy as np


def _preset():
    return {
        "adjustments": {"contrast": 50},
        "dither": {"style": "None", "scale": 1, "params": {}},
    }


def _source():
    gray = np.zeros((3, 4), np.float32)
    rgba = np.zeros((3, 4, 4), np.uint8)
    rgba[..., 3] = 255
    return gray, rgba, None


def _controller(panel):
    from ditherzam.dithering import registry
    from ditherzam.ui.layers_controller import LayersController

    return LayersController(
        panel,
        registry,
        preset_provider=_preset,
        source_provider=_source,
        cap_provider=lambda: 480,
        frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None,
    )


def _capture(monkeypatch):
    import ditherzam.ui.layers_controller as module

    events = []
    monkeypatch.setattr(
        module,
        "log_action",
        lambda action, **fields: events.append((action, fields)),
    )
    return events


def test_layer_lifecycle_logs_semantic_completions_only(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel

    events = _capture(monkeypatch)
    controller = _controller(LayersPanel())
    controller.initialize_source_layer()
    events.clear()

    controller.new_blank_layer()
    blank_id = controller.document.layers[1].id
    controller.duplicate_layer(1)
    copy_id = controller.document.layers[2].id
    controller.move_layer(2, 0)
    controller.set_visibility(0, False)
    controller.set_visibility(0, False)  # no-op
    controller.set_opacity(0, 37)
    controller.set_blend_mode(0, "multiply")
    controller.delete_layer(0)

    actions = [action for action, _fields in events]
    assert actions.count("layer.blank_added") == 1
    assert actions.count("layer.duplicated") == 1
    assert actions.count("layer.reordered") == 1
    assert actions.count("layer.visibility_changed") == 1
    assert actions.count("layer.opacity_changed") == 1
    assert actions.count("layer.blend_changed") == 1
    assert actions.count("layer.deleted") == 1

    by_action = {action: fields for action, fields in events}
    assert by_action["layer.blank_added"]["layer_id"] == blank_id
    assert by_action["layer.duplicated"]["layer_id"] == copy_id
    assert by_action["layer.opacity_changed"]["opacity"] == 37
    assert all(
        "path" not in key and "pixels" not in key
        for _action, fields in events
        for key in fields
    )


def test_mask_history_and_confirmation_actions_are_metadata_only(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel

    events = _capture(monkeypatch)
    controller = _controller(LayersPanel())
    controller.initialize_source_layer()
    events.clear()

    assert controller.hide_all_raster_mask(replace_existing=False)
    assert controller.set_active_raster_mask_density(40)
    assert controller.invert_active_raster_mask()
    assert controller.reveal_all_raster_mask(replace_existing=False) is False
    proposal = controller.mask_replacement_proposal
    assert proposal is not None
    assert controller.confirm_mask_replacement(proposal.token)
    assert controller.undo()
    assert controller.redo()
    assert controller.delete_active_raster_mask()

    actions = [action for action, _fields in events]
    assert actions == [
        "mask.created",
        "mask.density_changed",
        "mask.inverted",
        "mask.replacement_confirmed",
        "history.undo",
        "history.redo",
        "mask.deleted",
    ]
    created = events[0][1]
    assert (created["width"], created["height"]) == (4, 3)
    assert all(
        not isinstance(value, np.ndarray)
        for _action, fields in events
        for value in fields.values()
    )
