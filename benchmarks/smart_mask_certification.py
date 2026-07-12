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
                or not _hash(output["sha256"]) or type(output["width"]) is not int
                or type(output["height"]) is not int or output["width"] <= 0 or output["height"] <= 0
                or output["format"] not in {"PNG", "JPEG"}):
            raise AssertionError("invalid output evidence")
    surface = dimensions.get("surface")
    if surface and (("png" in str(surface) and not any(o["format"] == "PNG" for o in outputs.values()))
                    or ("jpeg" in str(surface) and not any(o["format"] == "JPEG" for o in outputs.values()))):
        raise AssertionError("output format does not match case surface")
    if dimensions.get("target") == "disabled" or dimensions.get("scenario") == "disabled-byte-baseline-zero-work":
        if evidence.get("zero_mask_work") is not True or not _hash(evidence.get("baseline_hash")) or evidence.get("baseline_hash") != evidence.get("output_hash"):
            raise AssertionError("disabled evidence must prove baseline equality and zero mask work")
    elif "target" in dimensions:
        if not _hash(evidence.get("baseline_hash")) or not _hash(evidence.get("output_hash")) or evidence["baseline_hash"] == evidence["output_hash"]:
            raise AssertionError("masked evidence must prove an output relationship")
    canonical = json.dumps({k: v for k, v in evidence.items() if k != "evidence_sha256"}, sort_keys=True, separators=(",", ":")).encode()
    if evidence.get("evidence_sha256") != hashlib.sha256(canonical).hexdigest():
        raise AssertionError("evidence signature/hash mismatch")
    return evidence


def certify_resilience(bundle: dict[str, Path], scenario: str, evidence: dict[str, object]) -> dict[str, object]:
    required = {"terminal_count", "stale_publications", "cache_bytes", "growth_mb", "elapsed_ms", "heartbeat_ms", "cancel_ms"}
    if not required <= evidence.keys():
        raise AssertionError("resilience/cache/performance evidence is incomplete")
    numeric = {k: evidence[k] for k in required - {"terminal_count", "stale_publications"}}
    if evidence["terminal_count"] != 1 or evidence["stale_publications"] != 0 or any(type(v) not in (int, float) or v < 0 for v in numeric.values()):
        raise AssertionError("invalid resilience evidence")
    if evidence["cache_bytes"] > 192 * 1024 * 1024 or evidence["heartbeat_ms"] > 100 or evidence["cancel_ms"] > 100 or evidence["growth_mb"] > 64:
        raise AssertionError("resilience budget exceeded")
    if scenario == "performance-fields" and evidence["elapsed_ms"] > 800:
        raise AssertionError("performance budget exceeded")
    if scenario == "quality-fields" and (evidence.get("dice", -1) < .90 or evidence.get("iou", -1) < .82 or evidence.get("boundary_f", -1) < .80):
        raise AssertionError("quality budget exceeded")
    return certify_case(bundle, evidence, scenario=scenario)


def _hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
