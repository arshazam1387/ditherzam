import hashlib
import json
import socket
import io
from pathlib import Path

import pytest

from ditherzam.masking.release_gate import ReleaseBundleError, verify_release_bundle


def _put(root: Path, relative: str, payload: bytes, content_id=None):
    path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
    result = {"path": relative, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    if content_id is not None: result["content_id"] = content_id
    return result


def _valid_bundle(root: Path):
    model = b"synthetic-not-an-onnx-model"; model_hash = hashlib.sha256(model).hexdigest()
    _put(root, "models/u2netp.onnx", model)
    manifest = f'''model_id: u2netp
model_version: synthetic
upstream_repository: upstream
upstream_commit: ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29
upstream_source_url: local-fixture
upstream_source_sha256: {'a'*64}
license: test-only
attribution: test
conversion_revision: test
conversion_opset: 12
conversion_tool_versions: {{test: "1"}}
onnx_sha256: {model_hash}
onnx_byte_count: {len(model)}
input_tensor: {{name: input.1, shape: [1, 3, 320, 320], dtype: float32}}
output_tensor: {{name: "1959", shape: [1, 1, 320, 320], dtype: float32}}
preprocessing: test
output_semantics: test
algorithm_version: test
relative_path: models/u2netp.onnx
output_names: [a, b, c, d, e, f, g]
'''.encode()
    content_id = hashlib.sha256(manifest).hexdigest()
    files = {"model_manifest": _put(root, "models/manifest.yaml", manifest)}
    for role, name in (("license", "LICENSE.txt"), ("notice", "NOTICE.txt"), ("provenance", "provenance.json")):
        files[role] = _put(root, name, f"{role}:{content_id}".encode(), content_id)
    from PIL import Image
    fixture = io.BytesIO(); Image.new("RGBA", (3, 2), (40, 50, 60, 255)).save(fixture, "PNG")
    files["smoke_fixture"] = _put(root, "fixtures/smoke.png", fixture.getvalue(), content_id)
    dlls = {name: _put(root, f"onnxruntime/capi/{name}", f"fake-{name}".encode())
            for name in ("onnxruntime.dll", "onnxruntime_providers_shared.dll")}
    lock_data = {"schema": 1, "status": "approved", "content_id": content_id,
                 "onnxruntime": "1.22.1", "files": files, "ort_dlls": dlls}
    lock = root / "release.json"; lock.write_text(json.dumps(lock_data), encoding="utf-8")
    return lock, lock_data


def test_pending_release_lock_fails_closed_without_assets(tmp_path):
    lock = tmp_path / "release.json"; lock.write_text(json.dumps({"status": "pending-approved-asset"}), encoding="utf-8")
    with pytest.raises(ReleaseBundleError, match="pending approval"): verify_release_bundle(tmp_path, lock)


def test_release_verification_works_with_all_socket_creation_denied(tmp_path, monkeypatch):
    lock, _ = _valid_bundle(tmp_path)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network attempted"))
    assert verify_release_bundle(tmp_path, lock)["model"].name == "u2netp.onnx"


@pytest.mark.parametrize("component", ["model_manifest", "license", "notice", "provenance", "onnxruntime.dll", "onnxruntime_providers_shared.dll", "model"])
def test_every_release_component_corruption_fails_closed(tmp_path, component):
    lock, data = _valid_bundle(tmp_path)
    if component == "model": path = tmp_path / "models/u2netp.onnx"
    elif component in data["files"]: path = tmp_path / data["files"][component]["path"]
    else: path = tmp_path / data["ort_dlls"][component]["path"]
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ReleaseBundleError, match="missing or corrupt|model asset verification"):
        verify_release_bundle(tmp_path, lock)


def test_runtime_inventory_and_path_reuse_fail_closed(tmp_path):
    lock, data = _valid_bundle(tmp_path)
    data["ort_dlls"].pop("onnxruntime.dll"); lock.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReleaseBundleError, match="exact approved"): verify_release_bundle(tmp_path, lock)


def test_schema_and_duplicate_paths_fail_closed(tmp_path):
    lock, data = _valid_bundle(tmp_path)
    data["schema"] = 2; lock.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReleaseBundleError, match="schema"): verify_release_bundle(tmp_path, lock)
    _, data = _valid_bundle(tmp_path)
    data["files"]["notice"] = dict(data["files"]["license"])
    lock.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReleaseBundleError, match="reuse"): verify_release_bundle(tmp_path, lock)


def test_offline_smoke_executes_fake_local_adapter_and_emits_hashed_evidence(tmp_path):
    from types import SimpleNamespace
    from ditherzam.offline_smoke import run
    lock, _ = _valid_bundle(tmp_path)
    class Adapter:
        def infer(self, rgba):
            return SimpleNamespace(probability=SimpleNamespace(values=np.full(rgba.shape[:2], .75, np.float32)))
    import numpy as np
    output = io.StringIO()
    assert run(lock, adapter_factory=Adapter, output=output, bundle_root=tmp_path) == 0
    evidence = json.loads(output.getvalue())
    assert evidence["executed"] is True
    assert {v["format"] for v in evidence["outputs"].values()} == {"PNG", "JPEG"}


def test_post_build_inventory_accepts_only_exact_locked_dlls(tmp_path):
    from tools.verify_smart_mask_frozen_inventory import verify
    lock, data = _valid_bundle(tmp_path)
    dist = tmp_path / "dist"; capi = dist / "onnxruntime" / "capi"; capi.mkdir(parents=True)
    for name, record in data["ort_dlls"].items():
        (capi / name).write_bytes((tmp_path / record["path"]).read_bytes())
    verify(dist, lock)
    (capi / "unexpected.dll").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="inventory"): verify(dist, lock)


def test_pyinstaller_spec_uses_locked_one_folder_layout():
    spec = (Path(__file__).resolve().parents[1] / "packaging/ditherzam-smart-mask.spec").read_text(encoding="utf-8")
    assert 'contents_directory="."' in spec
    assert 'datas.append((str(LOCK), "packaging"))' in spec
    assert "collect_dynamic_libs" not in spec and "collect_data_files" not in spec
    assert '("onnxruntime.dll", "onnxruntime_providers_shared.dll")' in spec
