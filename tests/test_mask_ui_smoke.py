from pathlib import Path

import numpy as np
import pytest

from ditherzam.masking.release_gate import ReleaseBundleError, verify_release_bundle
from ditherzam.masking.settings import SmartMaskSettings
from ditherzam.ui.main_window import ImageEditor
from ditherzam.ui.smart_mask_panel import MaskPanelStatus


def test_frozen_smoke_is_explicitly_pending_selected_asset():
    root = Path(__file__).resolve().parents[1]
    lock = root / "packaging" / "smart-mask-release.lock.example.json"
    with pytest.raises(ReleaseBundleError, match="pending approval"):
        verify_release_bundle(root, lock)


def test_build_entry_preflights_release_lock_before_builder(monkeypatch):
    from tools import build_smart_mask_release
    called = []
    monkeypatch.setattr(build_smart_mask_release, "verify_release_bundle",
                        lambda *a: (_ for _ in ()).throw(ReleaseBundleError("unapproved")))
    monkeypatch.setattr(build_smart_mask_release.subprocess, "call", lambda *a, **k: called.append(a))
    with pytest.raises(ReleaseBundleError, match="unapproved"):
        build_smart_mask_release.main()
    assert called == []


def test_missing_model_editor_starts_renders_unmasked_and_reports_unavailable(qapp_fixture, monkeypatch):
    import socket
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network attempted"))
    window = ImageEditor(mask_adapter=None, mask_model=None)
    rgba = np.zeros((3, 4, 4), np.uint8); rgba[..., :3] = 91; rgba[..., 3] = 255
    gray = np.full((3, 4), 91, np.float32)
    window.load_array(gray, rgba[..., :3], rgba)
    historical = np.frombuffer(window.render_now().bits(), dtype=np.uint8).copy()
    window.panel.smart_mask_panel.set_settings(SmartMaskSettings(enabled=True))
    window._on_mask_settings_changed(window.panel.smart_mask_panel.settings)
    window._request_mask_detection()
    assert window.panel.smart_mask_panel.status is MaskPanelStatus.MODEL_UNAVAILABLE
    assert window._current_mask_context() is None
    request = window._build_request(__import__("ditherzam.ui.render_request", fromlist=["RenderKind"]).RenderKind.FULL)
    assert request.mask_context is None
    after = np.frombuffer(window.render_now().bits(), dtype=np.uint8).copy()
    assert np.array_equal(after, historical)
