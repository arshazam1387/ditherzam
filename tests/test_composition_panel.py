def test_panel_empty_and_populated_states(qapp_fixture):
    from ditherzam.ui.composition_panel import CompositionPanel

    panel = CompositionPanel()
    assert panel.empty_label.isVisibleTo(panel) or panel.empty_label.text()
    assert not panel.remove_btn.isEnabled()
    assert not panel.play_btn.isEnabled()
    assert not panel.export_btn.isEnabled()

    panel.set_clips([("Look 1", 0, 24), ("Look 2", 24, 48)], 1)
    assert panel.look_list.count() == 2
    assert panel.selected_index() == 1
    assert panel.remove_btn.isEnabled()
    assert panel.transition_combo.isEnabled()


def test_panel_first_clip_disables_incoming_transition(qapp_fixture):
    from ditherzam.ui.composition_panel import CompositionPanel

    panel = CompositionPanel()
    panel.set_clips([("Look 1", 0, 24), ("Look 2", 24, 48)], 0)
    assert not panel.transition_combo.isEnabled()
    assert not panel.transition_duration_spin.isEnabled()
    panel.look_list.setCurrentRow(1)
    assert panel.transition_combo.isEnabled()
    assert panel.transition_duration_spin.isEnabled()


def test_panel_frame_range_and_status_contract(qapp_fixture):
    from ditherzam.ui.composition_panel import CompositionPanel

    panel = CompositionPanel()
    panel.set_frame_range(48, 12)
    assert panel.frame_slider.maximum() == 47
    assert panel.frame_slider.value() == 12
    assert panel.frame_label.text() == "13 / 48"
    panel.set_status("Model unavailable", error=True)
    assert panel.status_label.text() == "Model unavailable"
    assert panel.status_label.property("error") is True
