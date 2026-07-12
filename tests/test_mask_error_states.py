import json
import socket

import pytest

from ditherzam.masking.release_gate import ReleaseBundleError, verify_release_bundle


def test_pending_release_lock_fails_closed_without_assets(tmp_path):
    lock = tmp_path / "release.json"
    lock.write_text(json.dumps({"status": "pending-approved-asset"}), encoding="utf-8")
    with pytest.raises(ReleaseBundleError, match="pending approval"):
        verify_release_bundle(tmp_path, lock)


def test_release_verification_works_with_all_socket_creation_denied(tmp_path, monkeypatch):
    lock = tmp_path / "release.json"
    lock.write_text(json.dumps({"status": "pending-approved-asset"}), encoding="utf-8")
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network attempted"))
    with pytest.raises(ReleaseBundleError, match="pending approval"):
        verify_release_bundle(tmp_path, lock)


def test_approved_label_cannot_bypass_missing_notice_and_runtime_records(tmp_path):
    lock = tmp_path / "release.json"
    lock.write_text(json.dumps({"status": "approved", "onnxruntime": "1.22.1", "files": {}, "ort_dlls": {}}), encoding="utf-8")
    with pytest.raises(ReleaseBundleError, match="NOTICE"):
        verify_release_bundle(tmp_path, lock)


def test_release_records_cannot_escape_bundle(tmp_path):
    records = {name: {"path": "../escape", "sha256": "0" * 64}
               for name in ("model_manifest", "license", "notice", "provenance")}
    lock = tmp_path / "release.json"
    lock.write_text(json.dumps({"status": "approved", "onnxruntime": "1.22.1",
                                "files": records, "ort_dlls": {"runtime.dll": records["notice"]}}), encoding="utf-8")
    with pytest.raises(ReleaseBundleError, match="escapes"):
        verify_release_bundle(tmp_path, lock)
