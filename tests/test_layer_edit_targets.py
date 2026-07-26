import numpy as np


def _preset():
    return {
        "adjustments": {"contrast": 50},
        "dither": {"style": "None", "scale": 1, "params": {}},
    }


def _controller(panel):
    from ditherzam.dithering import registry
    from ditherzam.ui.layers_controller import LayersController

    gray = np.zeros((3, 4), np.float32)
    rgba = np.zeros((3, 4, 4), np.uint8)
    rgba[..., 3] = 255
    return LayersController(
        panel,
        registry,
        preset_provider=_preset,
        source_provider=lambda: (gray, rgba, None),
        cap_provider=lambda: 480,
        frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None,
    )


def test_dual_targets_are_stable_focusable_and_ui_only(qapp_fixture, monkeypatch):
    from PySide6.QtCore import Qt
    from ditherzam.ui.layers_controller import LayerEditTarget
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    controller = _controller(panel)
    controller.initialize_source_layer()
    monkeypatch.setattr(controller, "request_preview", lambda: (_ for _ in ()).throw(
        AssertionError("target switching must not render")))
    before = controller.document
    layer_id = before.layers[0].id
    layer_button, mask_button = panel._row_targets[layer_id]
    assert layer_button.focusPolicy() != Qt.FocusPolicy.NoFocus
    assert mask_button.focusPolicy() != Qt.FocusPolicy.NoFocus
    assert "pixels" in layer_button.accessibleName().lower()
    assert "raster mask" in mask_button.accessibleName().lower()

    mask_button.click()
    assert controller.edit_target == LayerEditTarget(layer_id, "mask")
    assert controller.document is before
    assert panel.mask_state_label.text() == "No raster mask"
    assert mask_button.isChecked() and not layer_button.isChecked()


def test_existing_mask_proposes_then_cancel_or_confirm_is_exactly_once(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    controller = _controller(panel)
    controller.initialize_source_layer()
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    panel.hide_mask_btn.click()
    assert np.all(controller.document.layers[0].raster_mask.pixels == 0)
    history_after_create = controller.undo_label

    panel.reveal_mask_btn.click()
    proposal = controller.mask_replacement_proposal
    assert proposal is not None
    assert controller.undo_label == history_after_create
    panel.cancel_replace_mask_btn.click()
    assert controller.mask_replacement_proposal is None
    assert np.all(controller.document.layers[0].raster_mask.pixels == 0)

    panel.reveal_mask_btn.click()
    token = controller.mask_replacement_proposal.token
    assert controller.confirm_mask_replacement(token)
    assert np.all(controller.document.layers[0].raster_mask.pixels == 255)
    assert not controller.confirm_mask_replacement(token)


def test_replacement_rejects_stale_mask_and_transform_forces_layer_target(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_controller import LayerEditTarget
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    controller = _controller(panel)
    controller.initialize_source_layer()
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    panel.hide_mask_btn.click()
    layer_id = controller.document.layers[0].id
    assert controller.set_edit_target(layer_id, "mask")
    panel.reveal_mask_btn.click()
    token = controller.mask_replacement_proposal.token
    assert controller.set_active_raster_mask_density(40)
    assert controller.mask_replacement_proposal is None
    assert not controller.confirm_mask_replacement(token)

    assert controller.set_edit_target(layer_id, "mask")
    assert controller.begin_transform_transaction()
    assert controller.edit_target == LayerEditTarget(layer_id, "layer")
    assert not controller.set_edit_target(layer_id, "mask")


def test_no_layer_and_disabled_mask_pixel_gate(qapp_fixture, monkeypatch):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    controller = _controller(panel)
    assert controller.edit_target is None
    assert not controller.can_edit_target_pixels
    assert not controller.set_edit_target("missing", "mask")

    controller.initialize_source_layer()
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    layer_id = controller.document.layers[0].id
    assert controller.set_edit_target(layer_id, "mask")
    assert controller.can_edit_target_pixels  # honest creation target
    assert controller.hide_all_raster_mask(replace_existing=False)
    assert controller.set_active_raster_mask_enabled(False)
    assert not controller.can_edit_target_pixels
    assert controller.edit_target.layer_id == layer_id
    assert controller.edit_target.kind == "mask"


def test_target_keyboard_and_focus_survive_row_rebuild(qapp_fixture, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    panel.show()
    panel.activateWindow()
    qapp_fixture.processEvents()
    controller = _controller(panel)
    controller.initialize_source_layer()
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    layer_id = controller.document.layers[0].id
    layer_button, mask_button = panel._row_targets[layer_id]

    mask_button.setFocus()
    qapp_fixture.processEvents()
    QTest.keyClick(mask_button, Qt.Key.Key_Return)
    assert controller.edit_target.kind == "mask"
    assert QApplication.focusWidget() is panel._row_targets[layer_id][1]

    controller.set_name(0, "Renamed")
    assert QApplication.focusWidget() is panel._row_targets[layer_id][1]
    QTest.keyClick(panel._row_targets[layer_id][1], Qt.Key.Key_Space)
    assert controller.edit_target.kind == "mask"
    assert panel.minimumWidth() == 280 and panel.maximumWidth() == 320


def test_proposal_cancels_on_selection_history_source_and_document_changes(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    controller = _controller(panel)
    controller.initialize_source_layer()
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    assert controller.hide_all_raster_mask(replace_existing=False)

    controller.reveal_all_raster_mask(replace_existing=True)
    assert controller.mask_replacement_proposal is not None
    controller.duplicate_layer(0)
    assert controller.mask_replacement_proposal is None

    controller.activate_layer(0)
    controller.reveal_all_raster_mask(replace_existing=True)
    assert controller.mask_replacement_proposal is not None
    assert controller.undo()
    assert controller.mask_replacement_proposal is None

    controller.reveal_all_raster_mask(replace_existing=True)
    controller.update_active_from_editor()
    assert controller.mask_replacement_proposal is None

    controller.reveal_all_raster_mask(replace_existing=True)
    controller.initialize_source_layer()
    assert controller.mask_replacement_proposal is None


def test_equal_pixels_with_nondefault_metadata_still_confirm(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    controller.initialize_source_layer()
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    controller.reveal_all_raster_mask(replace_existing=False)
    controller.set_active_raster_mask_enabled(False)
    controller.set_active_raster_mask_density(12)
    before = controller.document.layers[0].raster_mask

    assert controller.reveal_all_raster_mask(replace_existing=True) is False
    proposal = controller.mask_replacement_proposal
    assert proposal is not None
    assert controller.confirm_mask_replacement(proposal.token)
    after = controller.document.layers[0].raster_mask
    assert (after.enabled, after.density, after.revision) == (
        True, 100, before.revision + 1)


def test_pending_smart_source_refresh_preserves_raster_target(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_controller import LayerEditTarget
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    controller = _controller(panel)
    controller.initialize_source_layer()
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    layer_id = controller.document.layers[0].id
    controller.set_edit_target(layer_id, "mask")
    controller._source_provider = lambda: (
        controller.document.layers[0].source.gray,
        controller.document.layers[0].source.rgba,
        object(),  # pending/new Smart identity belongs to Look scope
    )
    assert controller.update_active_from_editor()
    assert controller.edit_target == LayerEditTarget(layer_id, "mask")
