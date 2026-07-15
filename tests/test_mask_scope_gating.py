import pytest

from ditherzam.masking.scope import mask_allows_media
from ditherzam.masking.settings import MaskTarget, SmartMaskSettings


@pytest.mark.parametrize("kind", ["svg", "batch", "video", "animation"])
def test_subject_and_background_block_deferred_media(kind):
    for target in (MaskTarget.SUBJECT, MaskTarget.BACKGROUND):
        assert not mask_allows_media(kind, SmartMaskSettings(enabled=True, target=target))


@pytest.mark.parametrize("kind", ["svg", "batch", "video", "animation"])
def test_disabled_and_whole_image_preserve_media_paths(kind):
    assert mask_allows_media(kind, SmartMaskSettings())
    assert mask_allows_media(
        kind, SmartMaskSettings(enabled=True, target=MaskTarget.WHOLE_IMAGE))


def test_still_raster_remains_supported_for_real_mask():
    settings = SmartMaskSettings(enabled=True)
    assert mask_allows_media("png", settings)
    assert mask_allows_media("jpeg", settings)


def test_video_guard_runs_before_dialog_pipeline_or_worker(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QWidget
    import ditherzam.ui.video_controller as module
    from ditherzam.ui.video_controller import VideoController

    QApplication.instance() or QApplication([])
    win = QWidget()
    win.viewport = object()
    calls = []
    monkeypatch.setattr(module.QMessageBox, "warning",
                        lambda *_args: calls.append("warning"))
    monkeypatch.setattr(module.QFileDialog, "getSaveFileName",
                        lambda *_args: (_ for _ in ()).throw(AssertionError("dialog opened")))
    ctrl = VideoController(
        win, object(), lambda: None, lambda: False,
        export_pipeline_provider=lambda: (_ for _ in ()).throw(
            AssertionError("pipeline created")),
        mask_settings_provider=lambda: SmartMaskSettings(enabled=True),
    )
    ctrl.temp_dir = tmp_path
    ctrl.export_video()
    assert calls == ["warning"]


def test_video_import_action_is_retained_and_tracks_mask_scope(qapp_fixture):
    from PySide6.QtWidgets import QWidget
    from ditherzam.ui.video_controller import VideoController

    win = QWidget()
    win.viewport = object()
    current = [SmartMaskSettings(enabled=True)]
    ctrl = VideoController(
        win, object(), lambda: None, lambda: False,
        mask_settings_provider=lambda: current[0],
    )
    ctrl.build_menu()
    ctrl.refresh_mask_scope()
    assert not ctrl._import_action.isEnabled()
    assert "video" in ctrl._import_action.toolTip().lower()
    assert not ctrl._export_action.isEnabled()

    current[0] = SmartMaskSettings()
    ctrl.refresh_mask_scope()
    assert ctrl._import_action.isEnabled()
    # Qt falls back to the action text when an explicit empty tooltip is set.
    assert ctrl._import_action.toolTip() == "Import Video"
    assert ctrl._import_action.statusTip() == ""
    assert not ctrl._export_action.isEnabled()  # no imported media yet
