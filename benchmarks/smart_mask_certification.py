"""Fail-closed validation of evidence emitted by certification runners."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def certify_case(bundle: dict[str, Path], evidence: dict[str, object], **dimensions: object) -> dict[str, object]:
    required_bundle = {"model", "model_manifest", "license", "notice", "provenance",
                       "smoke_fixture", "ort:onnxruntime.dll", "ort:onnxruntime_providers_shared.dll"}
    if required_bundle - bundle.keys():
        raise AssertionError("verified bundle is incomplete")
    if not isinstance(evidence, dict) or evidence.get("executed") is not True:
        raise AssertionError("concrete executed certification evidence is required")
    if evidence.get("case") != dimensions:
        raise AssertionError("certification evidence case identity mismatch")
    outputs = evidence.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise AssertionError("output hash/dimension/format evidence is required")
    for output in outputs.values():
        if (not isinstance(output, dict) or set(output) < {"sha256", "width", "height", "format"}
                or len(str(output["sha256"])) != 64 or int(output["width"]) <= 0 or int(output["height"]) <= 0):
            raise AssertionError("invalid output evidence")
    canonical = json.dumps({k: v for k, v in evidence.items() if k != "evidence_sha256"}, sort_keys=True, separators=(",", ":")).encode()
    if evidence.get("evidence_sha256") != hashlib.sha256(canonical).hexdigest():
        raise AssertionError("evidence signature/hash mismatch")
    return evidence


def certify_resilience(bundle: dict[str, Path], scenario: str, evidence: dict[str, object]) -> dict[str, object]:
    required = {"terminal", "cache_bytes", "elapsed_ms", "heartbeat_ms", "cancel_ms"}
    if not required <= evidence.keys():
        raise AssertionError("resilience/cache/performance evidence is incomplete")
    return certify_case(bundle, evidence, scenario=scenario)
