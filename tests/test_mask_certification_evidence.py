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


@pytest.mark.parametrize("tamper", ["uppercase", "zero-width", "wrong-format", "not-executed", "bad-signature"])
def test_placeholder_output_evidence_never_certifies(tmp_path, tamper):
    bundle = {name: tmp_path / name for name in ("model", "model_manifest", "license", "notice",
        "provenance", "smoke_fixture", "ort:onnxruntime.dll", "ort:onnxruntime_providers_shared.dll")}
    case = {"surface": "png-exact"}; evidence = _evidence(case)
    if tamper == "uppercase": evidence["outputs"]["png"]["sha256"] = "A" * 64
    elif tamper == "zero-width": evidence["outputs"]["png"]["width"] = 0
    elif tamper == "wrong-format": evidence["outputs"]["png"]["format"] = "JPEG"
    elif tamper == "not-executed": evidence["executed"] = False
    else: evidence["evidence_sha256"] = "0" * 64
    with pytest.raises(AssertionError): certify_case(bundle, evidence, **case)
