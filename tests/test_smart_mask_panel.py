import pytest

pytest.importorskip("PySide6")

from ditherzam.masking.settings import MaskTarget, SmartMaskSettings
from ditherzam.ui.smart_mask_panel import MaskPanelStatus, SmartMaskPanel


def test_exact_defaults_labels_and_ranges(qapp_fixture):
    panel = SmartMaskPanel()
    assert panel.settings == SmartMaskSettings()
    assert panel.title() == "Smart Mask"
    assert panel.enabled_check.text() == "Enabled"
    assert [panel.target_combo.itemText(i) for i in range(3)] == [
        "Subject", "Background", "Whole Image"]
    assert panel.candidate_combo.currentText() == "Primary (1 of 1)"
    assert not panel.candidate_combo.isEnabled()
    assert (panel.sensitivity_slider.minimum(), panel.sensitivity_slider.maximum(),
            panel.sensitivity_slider.value()) == (0, 100, 50)
    assert (panel.feather_slider.minimum(), panel.feather_slider.value()) == (0, 8)
    assert (panel.expansion_slider.minimum(), panel.expansion_slider.maximum(),
            panel.expansion_slider.value()) == (-64, 64, 0)
    assert panel.feather_spin.suffix() == " px"
    assert panel.expansion_spin.suffix() == " px"
    assert panel.outside_combo.currentText() == "Original"
    assert not panel.overlay_check.isChecked()


def test_settings_are_immutable_values_and_enable_is_signal_only(qapp_fixture):
    panel = SmartMaskPanel()
    seen = []
    panel.settings_changed.connect(seen.append)
    panel.enabled_check.click()
    assert panel.settings.enabled
    assert seen == [panel.settings]
    assert isinstance(seen[0], SmartMaskSettings)
    assert panel.status is MaskPanelStatus.NEEDS_DETECTION


def test_whole_image_disables_boundary_and_restores_values(qapp_fixture):
    panel = SmartMaskPanel()
    panel.set_availability(source=True, model=True)
    panel.enabled_check.click()
    panel.sensitivity_slider.setValue(73)
    panel.feather_slider.setValue(19)
    panel.expansion_slider.setValue(-7)
    panel.overlay_check.setChecked(True)
    panel.target_combo.setCurrentIndex(panel.target_combo.findData(MaskTarget.WHOLE_IMAGE))
    for control in (panel.candidate_combo, panel.sensitivity_slider,
                    panel.feather_slider, panel.expansion_slider, panel.invert_check,
                    panel.outside_combo, panel.overlay_check, panel.redetect_button,
                    panel.cancel_button):
        assert not control.isEnabled()
    panel.target_combo.setCurrentIndex(panel.target_combo.findData(MaskTarget.BACKGROUND))
    assert panel.sensitivity_slider.isEnabled()
    assert panel.feather_slider.isEnabled()
    assert panel.expansion_slider.isEnabled()
    assert panel.outside_combo.isEnabled()
    assert panel.overlay_check.isEnabled()
    assert panel.redetect_button.isEnabled()
    assert (panel.sensitivity_slider.value(), panel.feather_slider.value(),
            panel.expansion_slider.value(), panel.overlay_check.isChecked()) == (73, 19, -7, True)


def test_lifecycle_enablement_and_all_status_text(qapp_fixture):
    panel = SmartMaskPanel()
    panel.enabled_check.click()
    assert not panel.redetect_button.isEnabled()
    panel.set_availability(source=True, model=False)
    assert not panel.redetect_button.isEnabled()
    panel.set_availability(source=True, model=True)
    assert panel.redetect_button.isEnabled()
    for status in MaskPanelStatus:
        panel.set_status(status, 62)
        assert panel.status_label.text() == status.value
        assert panel.cancel_button.isEnabled() is (status is MaskPanelStatus.DETECTING)
    panel.set_status(MaskPanelStatus.DETECTING, 62)
    assert panel.progress_label.text() == "Detecting 62%"
    assert not panel.redetect_button.isEnabled()


def test_intent_signals_and_overlay_are_separate(qapp_fixture):
    panel = SmartMaskPanel()
    panel.enabled_check.click()
    panel.set_availability(source=True, model=True)
    events = []
    settings = []
    panel.redetect_requested.connect(lambda: events.append("redetect"))
    panel.cancel_requested.connect(lambda: events.append("cancel"))
    panel.overlay_changed.connect(lambda value: events.append(("overlay", value)))
    panel.settings_changed.connect(settings.append)
    panel.redetect_button.click()
    panel.set_status(MaskPanelStatus.DETECTING)
    panel.cancel_button.click()
    prior = len(settings)
    panel.overlay_check.click()
    assert events == ["redetect", "cancel", ("overlay", True)]
    assert len(settings) == prior
    assert not hasattr(panel.settings, "overlay")


def test_source_reset_turns_off_overlay_and_resets_lifecycle(qapp_fixture):
    panel = SmartMaskPanel()
    panel.enabled_check.click()
    panel.overlay_check.setChecked(True)
    panel.set_status(MaskPanelStatus.READY)
    panel.reset_for_source()
    assert not panel.overlay_check.isChecked()
    assert panel.status is MaskPanelStatus.NEEDS_DETECTION


def test_disclosure_is_independent_and_preserves_state(qapp_fixture):
    panel = SmartMaskPanel()
    panel.show()
    assert panel.disclosure_button.accessibleName() == "Hide Smart Mask controls"
    settings_seen = []
    overlay_seen = []
    panel.settings_changed.connect(settings_seen.append)
    panel.overlay_changed.connect(overlay_seen.append)
    panel.enabled_check.click()
    panel.sensitivity_slider.setValue(71)
    before = panel.settings
    settings_seen.clear()
    panel.disclosure_button.click()
    assert not panel.controls_widget.isVisible()
    assert panel.disclosure_button.accessibleName() == "Show Smart Mask controls"
    assert panel.settings == before
    assert settings_seen == [] and overlay_seen == []
    panel.disclosure_button.click()
    assert panel.controls_widget.isVisibleTo(panel)
    assert panel.disclosure_button.accessibleName() == "Hide Smart Mask controls"
    assert panel.settings == before


def test_set_settings_reconciles_stale_lifecycle_without_signals(qapp_fixture):
    panel = SmartMaskPanel()
    settings_seen = []
    panel.settings_changed.connect(settings_seen.append)
    panel.set_status(MaskPanelStatus.DETECTING, 44)
    panel.set_settings(SmartMaskSettings(enabled=True))
    assert panel.status is MaskPanelStatus.NEEDS_DETECTION
    assert panel.progress_label.text() == ""
    panel.set_valid_mask_available(True)
    assert panel.status is MaskPanelStatus.READY
    panel.set_status(MaskPanelStatus.DETECTING, 12)
    panel.set_settings(SmartMaskSettings(enabled=False))
    assert panel.status is MaskPanelStatus.DISABLED
    panel.set_settings(SmartMaskSettings(enabled=True))
    assert panel.status is MaskPanelStatus.READY
    assert settings_seen == []


def test_direct_enable_toggle_preserves_current_valid_mask(qapp_fixture):
    panel = SmartMaskPanel()
    panel.enabled_check.click()
    assert panel.status is MaskPanelStatus.NEEDS_DETECTION
    panel.set_status(MaskPanelStatus.READY)
    panel.enabled_check.click()
    assert panel.status is MaskPanelStatus.DISABLED
    panel.enabled_check.click()
    assert panel.status is MaskPanelStatus.READY


def test_labels_accessibility_and_focus_policy(qapp_fixture):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel

    panel = SmartMaskPanel()
    buddies = {label.text(): label.buddy() for label in panel.findChildren(QLabel)
               if label.buddy() is not None}
    assert buddies["Target"] is panel.target_combo
    assert buddies["Detected subject"] is panel.candidate_combo
    assert buddies["Sensitivity"] is panel.sensitivity_slider
    assert buddies["Edge feather"] is panel.feather_slider
    assert buddies["Expand / contract"] is panel.expansion_slider
    assert buddies["Outside region"] is panel.outside_combo
    assert panel.target_combo.accessibleName() == "Target"
    assert panel.status_label.accessibleName() == "Smart Mask status"
    assert panel.candidate_combo.focusPolicy() is Qt.FocusPolicy.NoFocus
    for spin in (panel.sensitivity_spin, panel.feather_spin, panel.expansion_spin):
        assert spin.focusPolicy() is Qt.FocusPolicy.NoFocus
