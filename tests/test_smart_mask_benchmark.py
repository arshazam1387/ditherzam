import json

import pytest

import numpy as np

from benchmarks.smart_mask import (AcceptanceUnavailable, BakeoffFixture,
    REQUIRED_FIXTURE_CATEGORIES, build_report_skeleton, require_release_selection, run_bakeoff)
from tools.convert_u2net_onnx import parse_args, sha256_file


def test_report_records_every_unmeasured_acceptance_dimension():
    report = build_report_skeleton(None, None)
    row = report["candidates"][0]
    assert row["measured"] is False
    assert set(row["performance"]) == {
        "cold_ms", "warm_median_ms", "warm_p95_ms", "preprocess_ms",
        "postprocess_ms", "geometry_1080p_ms", "geometry_4k_ms",
        "composite_1080p_ms", "composite_4k_ms", "peak_rss_delta_mb",
        "retained_rss_delta_mb", "heartbeat_max_ms", "cancel_max_ms", "growth_50_mb",
    }
    assert row["asset"]["output_names"] is None


def test_release_selection_is_honestly_unavailable_without_assets():
    with pytest.raises(AcceptanceUnavailable, match="measured eligible"):
        require_release_selection(build_report_skeleton(None, None))


def _asset_record(tmp_path, prefix, model_id):
    import hashlib
    manifest = tmp_path / f"{prefix}.yaml"; manifest.write_bytes(f"manifest-{prefix}".encode())
    onnx = tmp_path / f"{prefix}.onnx"; onnx.write_bytes(f"onnx-{prefix}".encode())
    return {"model_id": model_id, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
            "onnx_byte_count": onnx.stat().st_size, "output_names": [f"out-{i}" for i in range(7)],
            "manifest_path": str(manifest), "onnx_path": str(onnx),
            "license_evidence": "approved evidence", "redistribution_approved": True}


def _fixtures(rgba, truth):
    return [BakeoffFixture(category, category, rgba, truth) for category in REQUIRED_FIXTURE_CATEGORIES]


def _reviews():
    return {model: {category: True for category in REQUIRED_FIXTURE_CATEGORIES}
            for model in ("u2net", "u2netp")}


def _resilience(*_):
    return {"heartbeat_max_ms": 20.0, "cancel_max_ms": 30.0}


def test_injected_runner_measures_quality_performance_and_applies_frozen_policy(tmp_path):
    rgba = np.zeros((4, 5, 4), np.uint8); rgba[..., 3] = 255
    truth = np.zeros((4, 5), np.float32); truth[1:3, 1:4] = 1
    ticks = iter(i * .001 for i in range(10000))
    report = run_bakeoff(_fixtures(rgba, truth), {"u2net": lambda _: truth, "u2netp": lambda _: truth},
        {"u2net": _asset_record(tmp_path, "full", "u2net"), "u2netp": _asset_record(tmp_path, "lite", "u2netp")}, repeats=50,
        manual_reviews=_reviews(), resilience_probe=_resilience,
        clock=lambda: next(ticks), operation_sizes=((4, 5), (6, 7)))
    assert all(row["measured"] for row in report["candidates"])
    assert report["selection"]["winner"] == "u2netp"
    assert require_release_selection(report) == "u2netp"


def test_release_gate_rejects_serialized_winner_or_quality_tampering(tmp_path):
    rgba = np.zeros((3, 3, 4), np.uint8); rgba[..., 3] = 255
    truth = np.ones((3, 3), np.float32)
    ticks = iter(i * .001 for i in range(10000))
    report = run_bakeoff(_fixtures(rgba, truth),
        {"u2net": lambda _: truth, "u2netp": lambda _: truth},
        {"u2net": _asset_record(tmp_path, "full", "u2net"), "u2netp": _asset_record(tmp_path, "lite", "u2netp")}, repeats=50,
        manual_reviews=_reviews(), resilience_probe=_resilience,
        clock=lambda: next(ticks), operation_sizes=((3, 3), (4, 4)))
    report["selection"]["winner"] = "u2net"
    with pytest.raises(AcceptanceUnavailable, match="frozen SM-02 policy"):
        require_release_selection(report)
    report["selection"]["winner"] = "u2netp"
    report["candidates"].append(dict(report["candidates"][0]))
    with pytest.raises(AcceptanceUnavailable, match="measured eligible"):
        require_release_selection(report)
    report["candidates"].pop()
    lite = next(row for row in report["candidates"] if row["model_id"] == "u2netp")
    original_hash = lite["asset"]["onnx_sha256"]
    lite["asset"]["onnx_sha256"] = original_hash.upper()
    with pytest.raises(AcceptanceUnavailable, match="provenance"):
        require_release_selection(report)
    lite["asset"]["onnx_sha256"] = original_hash
    lite["performance"]["geometry_1080p_ms"] = 51.0
    lite["performance"]["composite_1080p_ms"] = 1.0
    with pytest.raises(AcceptanceUnavailable):
        require_release_selection(report)
    lite["performance"]["geometry_1080p_ms"] = 1.0
    lite["performance"]["geometry_4k_ms"] = 151.0
    lite["performance"]["composite_4k_ms"] = 1.0
    with pytest.raises(AcceptanceUnavailable):
        require_release_selection(report)
    lite["performance"]["geometry_4k_ms"] = 1.0
    lite["aggregate_dice"] = 0.1
    with pytest.raises(AcceptanceUnavailable):
        require_release_selection(report)


def test_release_gate_rejects_pending_or_failed_real_probes(tmp_path):
    rgba = np.zeros((2, 2, 4), np.uint8); rgba[..., 3] = 255
    truth = np.ones((2, 2), np.float32)
    ticks = iter(i * .001 for i in range(10000))
    report = run_bakeoff(_fixtures(rgba, truth), {"u2net": lambda _: truth, "u2netp": lambda _: truth},
        {"u2net": _asset_record(tmp_path, "f", "u2net"), "u2netp": _asset_record(tmp_path, "l", "u2netp")}, repeats=50,
        manual_reviews=_reviews(), clock=lambda: next(ticks), operation_sizes=((2, 2), (3, 3)))
    with pytest.raises(AcceptanceUnavailable, match="performance report is incomplete"):
        require_release_selection(report)
    for row in report["candidates"]:
        row["performance"]["heartbeat_max_ms"] = 101.0
        row["performance"]["cancel_max_ms"] = 1.0
    with pytest.raises(AcceptanceUnavailable):
        require_release_selection(report)


def test_converter_cli_and_streaming_hash_are_asset_free(tmp_path):
    source = tmp_path / "local.pt"
    source.write_bytes(b"local-only")
    assert len(sha256_file(source)) == 64
    args = parse_args(["--input", str(source), "--output", str(tmp_path / "x.onnx"),
                       "--record", str(tmp_path / "record.json")])
    assert args.opset == 17


@pytest.mark.skipif(not __import__("os").environ.get("DITHERZAM_SMART_MASK_ASSETS"),
                    reason="licensed model and fixtures are not approved/staged")
def test_real_acceptance_requires_explicit_opt_in():
    # This remains pending until the user supplies approved assets. The runner
    # must replace this skeleton assertion as part of the gated real bakeoff.
    pytest.fail("real Smart Mask acceptance is unavailable pending licensed assets")
