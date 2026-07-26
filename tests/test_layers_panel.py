def test_layers_panel_has_real_layer_controls(qapp_fixture):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPixmap
    from ditherzam.ui.layers_panel import LayersPanel
    panel = LayersPanel()
    assert panel.empty_label.text() == (
        "No layers. Place an image or create a blank layer.")
    assert not panel.delete_btn.isEnabled()
    panel.set_layers([("a", "Layer 1", True, 100, "normal")], 0)
    assert panel.layer_list.count() == 1
    assert panel.name_edit.text() == "Layer 1"
    assert panel.opacity_label.text() == "100%"
    item = panel.layer_list.item(0)
    assert item.checkState() == Qt.CheckState.Checked
    preview = QPixmap(8, 8)
    preview.fill(QColor("red"))
    panel.set_layer_thumbnail("a", preview)
    assert not item.icon().isNull()
    assert panel.delete_btn.isEnabled()


def test_layer_row_selection_and_visibility_publish_core_index(qapp_fixture):
    from PySide6.QtCore import Qt
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    selected = []
    visible = []
    panel.selection_changed.connect(selected.append)
    panel.visibility_changed.connect(
        lambda index, state: visible.append((index, state)))
    panel.set_layers([
        ("bottom", "Layer 1", True, 100, "normal"),
        ("top", "Layer 2", False, 100, "normal"),
    ], None)
    assert selected == []
    panel.layer_list.setCurrentRow(0)
    assert selected == [1]
    panel.layer_list.item(0).setCheckState(Qt.CheckState.Checked)
    assert visible == [(1, True)]


def test_transform_controls_publish_selected_layer_geometry(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    changed = []
    centered = []
    fitted = []
    panel.transform_changed.connect(
        lambda index, x, y, width, height:
        changed.append((index, x, y, width, height)))
    panel.center_requested.connect(centered.append)
    panel.fit_requested.connect(fitted.append)
    panel.set_layers(
        [("a", "Layer 1", True, 100, "normal", 2, -4, 40, 20)],
        selected=0,
    )
    assert panel.x_spin.value() == 2
    assert panel.y_spin.value() == -4
    assert panel.width_spin.value() == 40
    assert panel.height_spin.value() == 20
    panel.x_spin.setValue(9)
    assert changed[-1] == (0, 9, -4, 40, 20)
    panel.center_btn.click()
    panel.fit_btn.click()
    assert centered == [0]
    assert fitted == [0]


def test_transform_mode_has_explicit_start_confirm_and_cancel(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    started = []
    confirmed = []
    cancelled = []
    panel.transform_mode_requested.connect(started.append)
    panel.transform_confirmed.connect(lambda: confirmed.append(True))
    panel.transform_cancelled.connect(lambda: cancelled.append(True))
    panel.set_layers(
        [("a", "Layer 1", True, 100, "normal", 2, 3, 40, 20)],
        selected=0,
    )
    assert panel.transform_btn.isEnabled()
    assert panel.confirm_transform_btn.isHidden()
    assert panel.cancel_transform_btn.isHidden()
    panel.transform_btn.click()
    assert started == [0]
    panel.set_transform_mode(True)
    assert panel.transform_btn.isHidden()
    assert not panel.confirm_transform_btn.isHidden()
    assert not panel.cancel_transform_btn.isHidden()
    assert panel.confirm_transform_btn.accessibleName() == (
        "Confirm layer position and size")
    panel.confirm_transform_btn.click()
    panel.cancel_transform_btn.click()
    assert confirmed == [True]
    assert cancelled == [True]


def test_transform_mode_disables_every_unsafe_dock_control(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    panel.set_document_available(True)
    panel.set_layers(
        [
            ("bottom", "Layer 1", True, 100, "normal", 0, 0, 40, 20),
            ("top", "Layer 2", True, 80, "multiply", 2, 3, 30, 10),
        ],
        selected=1,
    )
    panel.set_transform_mode(True)

    unsafe = (
        panel.new_blank_btn, panel.place_image_btn, panel.duplicate_btn,
        panel.delete_btn, panel.up_btn, panel.down_btn, panel.layer_list,
        panel.visibility_check, panel.name_edit, panel.blend_combo,
        panel.opacity_slider, panel.export_btn,
    )
    assert all(not control.isEnabled() for control in unsafe)
    allowed = (
        panel.x_spin, panel.y_spin, panel.width_spin, panel.height_spin,
        panel.lock_aspect_check, panel.center_btn, panel.fit_btn,
        panel.confirm_transform_btn, panel.cancel_transform_btn,
    )
    assert all(control.isEnabled() for control in allowed)
    assert panel.transform_guidance_label.text() == (
        "Confirm or cancel the active layer transform first.")


def test_layers_panel_preserves_canvas_with_compact_width_and_tab_order(qapp_fixture):
    from PySide6.QtCore import Qt
    from ditherzam.ui.layers_panel import LayersPanel

    def next_focusable(widget):
        candidate = widget.nextInFocusChain()
        while candidate.focusPolicy() is Qt.FocusPolicy.NoFocus:
            candidate = candidate.nextInFocusChain()
        return candidate

    panel = LayersPanel()
    assert panel.minimumWidth() == 280
    assert panel.maximumWidth() == 320
    assert panel.focusProxy() is panel.layer_list
    assert next_focusable(panel.new_blank_btn) is panel.place_image_btn
    assert next_focusable(panel.opacity_slider) is panel.create_mask_btn
    assert next_focusable(panel.mask_density_slider) is panel.inspection_combo
    assert next_focusable(panel.inspection_combo) is panel.x_spin
    assert next_focusable(panel.fit_btn) is panel.transform_btn


def test_mask_section_state_signals_accessibility_and_transform_guard(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    panel.set_layers(
        [("a", "Layer 1", True, 100, "normal", 0, 0, 4, 4)], 0)
    panel.set_mask_state(False)
    assert panel.mask_state_label.text() == "No raster mask"
    assert panel.reveal_mask_btn.isEnabled()
    assert not panel.mask_enabled_check.isEnabled()
    assert not panel.mask_density_slider.isEnabled()

    emitted = []
    panel.reveal_mask_requested.connect(
        lambda index, replace: emitted.append(("reveal", index, replace)))
    panel.hide_mask_requested.connect(
        lambda index, replace: emitted.append(("hide", index, replace)))
    panel.transparency_mask_requested.connect(
        lambda index, replace: emitted.append(("alpha", index, replace)))
    panel.mask_enabled_changed.connect(
        lambda index, enabled: emitted.append(("enabled", index, enabled)))
    panel.mask_density_changed.connect(
        lambda index, density: emitted.append(("density", index, density)))

    panel.reveal_mask_btn.click()
    panel.hide_mask_btn.click()
    panel.transparency_mask_btn.click()
    assert emitted[:3] == [
        ("reveal", 0, True), ("hide", 0, True), ("alpha", 0, True)]

    panel.set_mask_state(True, enabled=False, density=37)
    assert panel.mask_state_label.text() == "Raster mask disabled · 37%"
    assert panel.mask_enabled_check.isEnabled()
    assert panel.mask_density_slider.isEnabled()
    assert panel.mask_density_label.text() == "37%"
    panel.mask_enabled_check.click()
    panel.mask_density_slider.setValue(42)
    assert emitted[-2:] == [("enabled", 0, True), ("density", 0, 42)]

    assert panel.transparency_mask_btn.accessibleName() == (
        "Create raster mask from source transparency")
    assert panel.mask_enabled_check.accessibleName() == (
        "Enable selected layer raster mask")
    assert panel.mask_density_slider.accessibleName() == (
        "Selected layer raster mask density")

    panel.set_transform_mode(True)
    assert all(not widget.isEnabled() for widget in (
        panel.reveal_mask_btn, panel.hide_mask_btn,
        panel.transparency_mask_btn, panel.mask_enabled_check,
        panel.mask_density_slider,
    ))
