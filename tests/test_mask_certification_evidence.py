import hashlib
import json

import pytest

from benchmarks.smart_mask_certification import certify_case


def _evidence(case):
    evidence = {"executed": True, "case": case, "outputs": {"png": {
        "sha256": "a" * 64, "width": 3, "height": 2, "format": "PNG"}}}
    raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    evidence["evidence_sha256"] = hashlib.sha256(raw).hexdigest()
    return evidence


def test_concrete_evidence_is_required_and_bound_to_exact_case(tmp_path):
    bundle = {name: tmp_path / name for name in ("model", "model_manifest", "license", "notice",
        "provenance", "smoke_fixture", "ort:onnxruntime.dll", "ort:onnxruntime_providers_shared.dll")}
    case = {"source_kind": "rgb", "surface": "png-exact"}
    assert certify_case(bundle, _evidence(case), **case)["executed"] is True
    with pytest.raises(AssertionError, match="executed"): certify_case(bundle, {}, **case)
    with pytest.raises(AssertionError, match="identity"): certify_case(bundle, _evidence(case), source_kind="rgba")
