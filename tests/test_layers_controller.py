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


def _controller(panel, **overrides):
    from ditherzam.dithering import registry
    from ditherzam.ui.layers_controller import LayersController

    values = dict(
        preset_provider=lambda: _preset(),
        source_provider=_source,
        cap_provider=lambda: 480,
        frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None,
    )
    values.update(overrides)
    return LayersController(panel, registry, **values)


def test_add_duplicate_move_and_delete_layers(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    controller = _controller(panel)
    controller.initialize_source_layer()
    controller.add_layer()
    assert [layer.name for layer in controller.stack.layers] == ["Layer 1", "Layer 2"]
    assert len({layer.id for layer in controller.stack.layers}) == 2

    controller.duplicate_layer(0)
    assert controller.stack.layers[1].name == "Layer 1 copy"
    assert controller.stack.layers[1].id != controller.stack.layers[0].id
    controller.move_layer(1, 2)
    assert controller.stack.layers[2].name == "Layer 1 copy"
    controller.delete_layer(2)
    assert [layer.name for layer in controller.stack.layers] == ["Layer 1", "Layer 2"]


def test_layer_edits_replace_immutable_stack(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    controller = _controller(panel)
    controller.initialize_source_layer()
    controller.add_layer()
    before = controller.stack
    controller.set_name(0, "Ink")
    controller.set_visibility(0, False)
    controller.set_opacity(0, 42)
    controller.set_blend_mode(0, "multiply")
    assert controller.stack is not before
    assert controller.stack.layers[0].name == "Ink"
    assert controller.stack.layers[0].visible is False
    assert controller.stack.layers[0].opacity == 42
    assert controller.stack.layers[0].blend_mode == "multiply"


def test_selected_layer_position_size_center_and_fit_are_independent(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    controller = _controller(panel)
    controller.initialize_source_layer()
    original = controller.document.layers[0]
    controller.add_layer()
    controller.set_transform(1, x=-3, y=5, width=8, height=6)
    changed = controller.document.layers[1]
    assert changed.transform.x == -3
    assert changed.transform.y == 5
    assert changed.transform.scale_x == 2.0
    assert changed.transform.scale_y == 1.5
    assert controller.document.layers[0] is original

    controller.center_layer(1)
    assert controller.document.layers[1].transform.x == -2
    assert controller.document.layers[1].transform.y == -1

    controller.fit_layer(1)
    fitted = controller.document.layers[1].transform
    assert fitted.scale_x == 1.0
    assert fitted.scale_y == 1.0
    assert fitted.x == 0
    assert fitted.y == 0


def test_active_layer_can_move_to_document_position(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    controller.initialize_source_layer()
    assert controller.active_geometry() == (0, 0, 4, 4)
    controller.move_active_layer_to(7, -2)
    assert controller.active_geometry() == (7, -2, 4, 4)


def test_transform_gesture_defers_panel_rebuild_and_thumbnail_invalidation(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel
    from ditherzam.ui.layers_controller import _thumbnail_key
    from PySide6.QtGui import QPixmap

    panel = LayersPanel()
    controller = _controller(panel)
    controller.initialize_source_layer()
    preview_requests = []
    monkeypatch.setattr(
        controller, "request_preview", lambda: preview_requests.append(
            controller.document
        )
    )
    original = controller.document.layers[0]
    original_key = _thumbnail_key(original)
    controller._thumbnail_cache[original_key] = QPixmap(1, 1)
    rebuilds = []
    monkeypatch.setattr(
        panel, "set_layers", lambda rows, selected: rebuilds.append(
            (tuple(rows), selected)
        )
    )

    assert controller.begin_transform_gesture() is True
    for step in range(120):  # two seconds of pointer updates at 60 Hz
        controller.move_active_layer_to(step, -step)

    assert controller.active_geometry() == (119, -119, 4, 4)
    assert rebuilds == []
    assert original_key in controller._thumbnail_cache
    assert controller._freeze_request()[3] == ()

    assert controller.end_transform_gesture() is True
    assert len(rebuilds) == 1
    assert len(preview_requests) == 1
    assert preview_requests[-1] is controller.document
    assert original_key not in controller._thumbnail_cache
    current_key = controller._freeze_request()[3]
    assert len(current_key) == 1
    assert current_key[0][0] == original.id


def test_transform_gesture_preserves_latest_exact_geometry(qapp_fixture, monkeypatch):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    controller.initialize_source_layer()
    requests = []
    monkeypatch.setattr(
        controller, "request_preview",
        lambda: requests.append(controller.document),
    )

    assert controller.begin_transform_gesture() is True
    controller.resize_active_layer_to(-5, 6, 11, 9)
    controller.resize_active_layer_to(7, -8, 13, 15)
    assert controller.end_transform_gesture() is True

    assert controller.active_geometry() == (7, -8, 13, 15)
    assert requests[-1] is controller.document
    assert len(requests) == 1


def test_apply_layer_passes_defensive_preset(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    applied = []
    panel = LayersPanel()
    controller = _controller(
        panel, apply_preset=applied.append)
    controller.initialize_source_layer()
    applied.clear()  # opening activates Layer 1; isolate the explicit activation
    controller.activate_layer(0)
    assert applied == [_preset()]


def test_active_layer_edit_replaces_only_its_look(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    current = {"value": _preset(contrast=10)}
    panel = LayersPanel()
    controller = _controller(
        panel,
        preset_provider=lambda: current["value"])
    controller.initialize_source_layer()
    current["value"] = _preset(contrast=90)
    controller.add_layer()
    untouched = controller.stack.layers[1]

    panel.layer_list.setCurrentRow(1)  # visual bottom row -> core Layer 1
    current["value"] = _preset(contrast=25)
    assert controller.update_active_from_editor() is True
    assert controller.stack.layers[0].look.preset == _preset(contrast=25)
    assert controller.stack.layers[1] is untouched
    assert controller.stack.layers[0].name == "Layer 1"
    assert controller.stack.layers[0].opacity == 100


def test_update_without_active_layer_does_not_capture_editor(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    controller = _controller(panel, source_provider=lambda: None)
    assert controller.has_active_layer is False
    assert controller.update_active_from_editor() is False


def test_delete_activates_the_new_selected_layer(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    applied = []
    current = {"value": _preset(contrast=10)}
    panel = LayersPanel()
    controller = _controller(
        panel,
        preset_provider=lambda: current["value"],
        apply_preset=applied.append)
    controller.initialize_source_layer()
    current["value"] = _preset(contrast=90)
    controller.add_layer()

    applied.clear()
    controller.delete_layer(1)
    assert applied == [_preset(contrast=10)]
    assert controller.active_index == 0


def test_new_source_resets_document_to_one_selected_base_layer(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    current = {"value": _preset(contrast=10)}
    panel = LayersPanel()
    controller = _controller(
        panel,
        preset_provider=lambda: current["value"])
    controller.initialize_source_layer()
    current["value"] = _preset(contrast=90)
    controller.add_layer()

    controller.initialize_source_layer()

    assert len(controller.stack.layers) == 1
    assert controller.stack.layers[0].name == "Layer 1"
    assert controller.stack.layers[0].look.preset == _preset(contrast=90)
    assert controller.active_index == 0
    assert controller._next_layer_number == 2


def test_last_source_layer_can_be_deleted_to_empty_document(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    controller = _controller(panel)
    controller.initialize_source_layer()

    controller.delete_layer(0)

    assert controller.stack.layers == ()
    assert controller.document is not None
    assert controller.active_index is None


def test_preview_without_source_reports_inline_error(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    controller = _controller(panel, source_provider=lambda: None)
    controller.add_layer()
    assert "image" in panel.status_label.text().lower()
    assert panel.status_label.property("error") is True


def test_shutdown_is_idempotent_and_blocks_mutation(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    controller = _controller(panel, source_provider=lambda: None)
    controller.shutdown()
    controller.shutdown()
    controller.add_layer()
    assert not controller.stack.layers


def test_mutation_guard_blocks_graph_changes_but_allows_geometry(
    qapp_fixture, monkeypatch, tmp_path
):
    from ditherzam.ui.layers_panel import LayersPanel
    import ditherzam.ui.layers_controller as module

    allowed = {"value": True}
    applied_sources = []
    applied_presets = []
    controller = _controller(
        LayersPanel(),
        mutation_guard=lambda: allowed["value"],
        apply_source=lambda *source: applied_sources.append(source),
        apply_preset=applied_presets.append,
    )
    controller.initialize_source_layer()
    applied_sources.clear()
    applied_presets.clear()
    original_document = controller.document
    original_geometry = controller.active_geometry()
    allowed["value"] = False
    render_calls = []
    monkeypatch.setattr(
        module,
        "render_layer_document",
        lambda *_args, **_kwargs: render_calls.append(True),
    )

    gray, rgba, _probability = _source()
    controller.open_document(gray, rgba)
    controller.place_source(gray, rgba)
    controller.new_blank_layer()
    controller.duplicate_layer(0)
    controller.delete_layer(0)
    controller.move_layer(0, 0)
    controller.set_visibility(0, False)
    controller.set_name(0, "Blocked")
    controller.set_blend_mode(0, "multiply")
    controller.set_opacity(0, 10)
    controller.activate_layer(0)
    assert controller.update_active_from_editor() is False
    assert controller.export_current(tmp_path / "blocked.png") is None

    assert controller.document is original_document
    assert controller.active_geometry() == original_geometry
    assert applied_sources == []
    assert applied_presets == []
    assert render_calls == []

    controller.set_transform(0, x=7, y=-3, width=6, height=5)
    assert controller.active_geometry() == (7, -3, 6, 5)


def test_stale_terminal_cannot_clear_active_worker_or_consume_pending(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    pending = (11, "document", 480, (), ())
    controller._busy = True
    controller._active_generation = 10
    controller._pending_request = pending
    starts = []
    monkeypatch.setattr(
        controller, "_start_request", lambda request: starts.append(request))

    controller._terminal(9)
    assert controller._busy is True
    assert controller._pending_request is pending
    assert starts == []

    controller._terminal(10)
    assert controller._busy is False
    assert controller._pending_request is None
    assert starts == [pending]

    controller._terminal(9)
    assert starts == [pending]


def test_same_generation_old_mask_publication_is_rejected(
    qapp_fixture, monkeypatch,
):
    from dataclasses import replace
    from ditherzam.layers import RasterLayerMask
    from ditherzam.ui.layers_controller import _publication_key
    from ditherzam.ui.layers_panel import LayersPanel

    frames = []
    controller = _controller(LayersPanel(), frame_sink=frames.append)
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    controller.initialize_source_layer()
    layer = controller.document.layers[0]
    mask_a = RasterLayerMask(np.zeros((4, 4), np.uint8))
    old_document = controller.document.replace(
        0, replace(layer, raster_mask=mask_a))
    generation = controller._generation
    old_keys = tuple(
        _publication_key(item, old_document, 480, generation)
        for item in old_document.layers)
    mask_b = mask_a.evolve(pixels=np.full((4, 4), 255, np.uint8))
    controller._document = old_document.replace(
        0, replace(old_document.layers[0], raster_mask=mask_b))
    controller._active_generation = generation
    controller._busy = True
    frame = np.zeros((4, 4, 4), np.uint8)

    controller._preview_finished((frame, (), None, old_keys), generation)

    assert frames == []
    assert controller._latest_proxy is None


def test_discrete_layer_and_mask_mutations_are_undoable(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    controller.initialize_source_layer()
    controller.set_name(0, "Ink")
    controller.reveal_all_raster_mask(replace_existing=False)
    assert controller.undo_label == "Edit Layer Mask"
    assert controller.undo() is True
    assert controller.document.layers[0].raster_mask is None
    assert controller.undo() is True
    assert controller.document.layers[0].name == "Layer 1"
    assert controller.redo() is True
    assert controller.document.layers[0].name == "Ink"


def test_selection_and_editor_look_capture_do_not_enter_history(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel

    current = {"value": _preset(contrast=10)}
    controller = _controller(
        LayersPanel(), preset_provider=lambda: current["value"])
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    controller.initialize_source_layer()
    controller.place_source(*_source())
    label = controller.undo_label
    controller.activate_layer(0)
    current["value"] = _preset(contrast=88)
    controller.update_active_from_editor()
    assert controller.undo_label == label


def test_transform_transaction_records_confirm_once_and_cancel_never(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    controller.initialize_source_layer()
    assert controller.begin_transform_transaction()
    controller.move_active_layer_to(4, 5)
    controller.move_active_layer_to(8, 9)
    assert controller.confirm_transform_transaction()
    assert controller.undo_label == "Transform Layer"
    assert controller.undo()
    assert controller.active_geometry() == (0, 0, 4, 4)

    assert controller.begin_transform_transaction()
    controller.move_active_layer_to(3, 2)
    assert controller.cancel_transform_transaction()
    assert controller.active_geometry() == (0, 0, 4, 4)
    assert controller.redo_label == "Transform Layer"


def test_direct_undo_redo_calls_are_blocked_during_transform_transaction(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    controller.initialize_source_layer()
    controller.set_name(0, "Ink")
    assert controller.begin_transform_transaction()
    current = controller.document
    assert controller.undo() is False
    assert controller.redo() is False
    assert controller.document is current
    controller.cancel_transform_transaction()


def test_history_move_invalidates_workers_thumbnails_and_proxy_not_look_cache(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel

    clears = []
    controller = _controller(
        LayersPanel(), proxy_clear=lambda: clears.append(True))
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    controller.initialize_source_layer()
    controller.set_name(0, "Ink")
    fake_worker = type("Worker", (), {"cancel": lambda self: setattr(
        self, "cancelled", True)})()
    controller._workers.add(fake_worker)
    controller._thumbnail_cache[("stale",)] = object()
    controller._latest_proxy = object()
    look_cache = controller._look_cache
    assert controller.undo()
    assert fake_worker.cancelled is True
    assert controller._thumbnail_cache == {}
    assert controller._latest_proxy is None
    assert controller._look_cache is look_cache
    assert clears


def test_successful_open_resets_history_but_failed_open_preserves_it(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    controller.initialize_source_layer()
    controller.set_name(0, "Ink")
    assert controller.can_undo
    controller.open_document("bad", "bad")
    assert controller.can_undo
    controller.open_document(*_source())
    assert not controller.can_undo
