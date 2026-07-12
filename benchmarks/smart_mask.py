"""Smart Mask quality/performance bakeoff harness -- ASSET-GATED SKELETON.

This is argument parsing and a JSON report schema only. It does not run any
inference, load any model, or read any fixture pixels -- there are no real
Smart Mask model weights or licensed fixtures staged in this repository yet.
SM-16 is the task that completes the real bakeoff runner: loading approved
fixtures/manifest, running each candidate model, timing cold/warm inference,
measuring quality against `ditherzam.masking.quality`, and populating this
report's candidate rows with real numbers before calling
`select_model_candidate`.

Running this module today produces a valid, schema-conformant report whose
candidates are all `measured: False` and whose `selection` is `None` --
there is nothing to bake off yet.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image

from ditherzam.masking.composite import composite_masked
from ditherzam.masking.geometry import derive_master_mask
from ditherzam.masking.model_assets import APPROVED_MODEL_IDS, ModelManifest, load_manifest
from ditherzam.masking.quality import (BOUNDARY_F_THRESHOLD, CATEGORY_DICE_FLOOR,
    DICE_AGGREGATE_THRESHOLD, IOU_AGGREGATE_THRESHOLD, COLD_LATENCY_MS, WARM_LATENCY_MEDIAN_MS,
    WARM_LATENCY_P95_MS, AggregateQuality, CandidateReport, ImageQualityResult,
    aggregate_quality, boundary_f_score, dice_score, iou_score, select_model_candidate)
from ditherzam.masking.settings import MaskTarget, OutsideMode

REPORT_SCHEMA_VERSION = 1
REQUIRED_FIXTURE_CATEGORIES = frozenset({"portrait", "product", "animal", "full_body",
    "multiple_people", "busy_background", "low_contrast", "transparent",
    "thin_structures", "no_subject"})
MAX_HEARTBEAT_MS = 100.0
MAX_CANCEL_MS = 100.0
MAX_RETAINED_MB = 192.0
MAX_GROWTH_50_MB = 192.0
MAX_GEOMETRY_COMPOSITE_1080_MS = 100.0
MAX_GEOMETRY_COMPOSITE_4K_MS = 300.0
MAX_GEOMETRY_1080_MS = 50.0
MAX_GEOMETRY_4K_MS = 150.0


def _operating_budgets_pass(perf: Mapping[str, object]) -> bool:
    try:
        values = [float(value) for value in perf.values()]
        return (len(values) == 14 and all(np.isfinite(v) and v >= 0 for v in values)
            and float(perf["cold_ms"]) <= COLD_LATENCY_MS
            and float(perf["warm_median_ms"]) <= WARM_LATENCY_MEDIAN_MS
            and float(perf["warm_p95_ms"]) <= WARM_LATENCY_P95_MS
            and float(perf["heartbeat_max_ms"]) <= MAX_HEARTBEAT_MS
            and float(perf["cancel_max_ms"]) <= MAX_CANCEL_MS
            and float(perf["retained_rss_delta_mb"]) <= MAX_RETAINED_MB
            and float(perf["growth_50_mb"]) <= MAX_GROWTH_50_MB
            and float(perf["geometry_1080p_ms"]) <= MAX_GEOMETRY_1080_MS
            and float(perf["geometry_4k_ms"]) <= MAX_GEOMETRY_4K_MS
            and float(perf["geometry_1080p_ms"]) + float(perf["composite_1080p_ms"]) <= MAX_GEOMETRY_COMPOSITE_1080_MS
            and float(perf["geometry_4k_ms"]) + float(perf["composite_4k_ms"]) <= MAX_GEOMETRY_COMPOSITE_4K_MS)
    except (KeyError, TypeError, ValueError):
        return False


def _memory_mb() -> tuple[float | None, float]:
    """Return current and OS lifetime-peak working set where available."""
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = (ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong)
        get_info.restype = ctypes.c_int
        if not get_info(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return counters.WorkingSetSize / 1048576.0, counters.PeakWorkingSetSize / 1048576.0
    import resource
    scale = 1048576.0 if sys.platform == "darwin" else 1024.0
    # ru_maxrss is a lifetime peak, not current RSS. Keep retained/growth
    # pending on POSIX rather than relabeling the peak as a current sample.
    return None, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


class AcceptanceUnavailable(RuntimeError):
    """The licensed inputs needed for a release decision are not available."""


@dataclass(frozen=True)
class BakeoffFixture:
    fixture_id: str
    category: str
    rgba: np.ndarray
    truth: np.ndarray


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fixture_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load and checksum an approved local fixture manifest; never fetch paths."""
    manifest_path = Path(path).resolve()
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    rows = data.get("fixtures") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise AcceptanceUnavailable("fixture manifest has no approved fixtures")
    required = {"fixture_id", "category", "image_relative_path", "source_sha256",
                "ground_truth_relative_path", "ground_truth_sha256", "license",
                "ground_truth_review_status"}
    for row in rows:
        if not isinstance(row, dict) or required - set(row):
            raise AcceptanceUnavailable("fixture provenance is incomplete")
        if row["ground_truth_review_status"] != "reviewed" or not str(row["license"]).strip():
            raise AcceptanceUnavailable("fixture license/review approval is incomplete")
        for path_key, hash_key in (("image_relative_path", "source_sha256"),
                                   ("ground_truth_relative_path", "ground_truth_sha256")):
            target = (manifest_path.parent / str(row[path_key])).resolve()
            if manifest_path.parent not in target.parents or not target.is_file() or _sha256(target) != row[hash_key]:
                raise AcceptanceUnavailable("fixture asset is absent, unsafe, or corrupt")
    if {str(row["category"]) for row in rows} != set(REQUIRED_FIXTURE_CATEGORIES):
        raise AcceptanceUnavailable("fixture manifest does not cover the required acceptance matrix")
    return rows


def execute_asset_bakeoff(fixtures_manifest: str | Path, candidate_manifests: Sequence[str | Path],
                          *, redistribution_approved: bool,
                          manual_approvals: Sequence[str] = (),
                          adapter_factory: Callable[[ModelManifest, Path], object] | None = None) -> dict[str, Any]:
    """Load only verified local assets and execute the real harness."""
    if not redistribution_approved:
        raise AcceptanceUnavailable("redistribution sign-off is required")
    fixture_path = Path(fixtures_manifest).resolve()
    fixture_rows = load_fixture_manifest(fixture_path)
    fixtures: list[BakeoffFixture] = []
    for row in fixture_rows:
        rgba = np.asarray(Image.open(fixture_path.parent / row["image_relative_path"]).convert("RGBA"), dtype=np.uint8)
        truth = np.asarray(Image.open(fixture_path.parent / row["ground_truth_relative_path"]).convert("L"), dtype=np.float32) / 255.0
        fixtures.append(BakeoffFixture(str(row["fixture_id"]), str(row["category"]), rgba, truth))
    manifests: dict[str, tuple[ModelManifest, Path]] = {}
    for value in candidate_manifests:
        path = Path(value).resolve(); manifest = load_manifest(path)
        if manifest.model_id in manifests:
            raise AcceptanceUnavailable("duplicate candidate manifest")
        manifests[manifest.model_id] = (manifest, path)
    if set(manifests) != set(APPROVED_MODEL_IDS):
        raise AcceptanceUnavailable("both candidate manifests are required")
    if adapter_factory is None:
        from ditherzam.masking.ort_adapter import OrtSegmentationAdapter
        adapter_factory = lambda manifest, root: OrtSegmentationAdapter(manifest, asset_root=root)
    candidates: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    records: dict[str, dict[str, Any]] = {}
    for model_id, (manifest, path) in manifests.items():
        adapter = adapter_factory(manifest, path.parent)
        candidates[model_id] = lambda rgba, adapter=adapter: adapter.infer(rgba).probability.values
        records[model_id] = {"manifest_sha256": _sha256(path), "onnx_sha256": manifest.onnx_sha256,
            "model_id": manifest.model_id,
            "onnx_byte_count": manifest.onnx_byte_count, "output_names": list(manifest.output_names or ()),
            "manifest_path": str(path), "onnx_path": str((path.parent / manifest.relative_path).resolve()),
            "license_evidence": f"{manifest.license}: {manifest.attribution}",
            "redistribution_approved": True}
    report = run_bakeoff(fixtures, candidates, records,
        manual_reviews={model_id: {category: model_id in manual_approvals
            for category in REQUIRED_FIXTURE_CATEGORIES} for model_id in APPROVED_MODEL_IDS})
    report["fixtures_manifest"] = str(fixture_path)
    report["fixtures_manifest_sha256"] = _sha256(fixture_path)
    return report


def run_bakeoff(fixtures: Sequence[BakeoffFixture], candidates: Mapping[str, Callable[[np.ndarray], np.ndarray]],
                asset_records: Mapping[str, dict[str, Any]], *, repeats: int = 50,
                manual_reviews: Mapping[str, Mapping[str, bool]] | None = None,
                clock: Callable[[], float] = time.perf_counter,
                resilience_probe: Callable[[str, Callable[[np.ndarray], np.ndarray], BakeoffFixture], Mapping[str, float]] | None = None,
                preprocess_probe: Callable[[np.ndarray], object] = lambda rgba: np.ascontiguousarray(rgba[..., :3], dtype=np.float32) / 255.0,
                postprocess_probe: Callable[[np.ndarray], object] = lambda probability: np.clip(probability, 0.0, 1.0),
                operation_sizes: Sequence[tuple[int, int]] = ((1080, 1920), (2160, 3840))) -> dict[str, Any]:
    """Execute the identical bakeoff with injected local candidate adapters."""
    if not fixtures or set(candidates) != set(APPROVED_MODEL_IDS) or repeats < 2:
        raise AcceptanceUnavailable("both candidates and approved fixtures are required")
    report = build_report_skeleton(None, None)
    rows = {row["model_id"]: row for row in report["candidates"]}
    policy_reports: dict[str, CandidateReport] = {}
    for model_id in sorted(APPROVED_MODEL_IDS):
        infer = candidates[model_id]
        metrics: list[ImageQualityResult] = []
        fixture_metrics: list[dict[str, Any]] = []
        samples: list[float] = []
        preprocess_samples: list[float] = []; postprocess_samples: list[float] = []
        before_current, before_peak = _memory_mb()
        for fixture_index, fixture in enumerate(fixtures):
            for repeat in range(repeats):
                started = clock()
                probability = np.asarray(infer(fixture.rgba), dtype=np.float32)
                elapsed = (clock() - started) * 1000.0
                if probability.shape != fixture.truth.shape or not np.isfinite(probability).all():
                    raise AcceptanceUnavailable("candidate returned an invalid probability map")
                samples.append(elapsed)
                if repeat == 0:
                    started = clock(); preprocess_probe(fixture.rgba); preprocess_samples.append((clock()-started)*1000)
                    started = clock(); postprocess_probe(probability); postprocess_samples.append((clock()-started)*1000)
                    measured = ImageQualityResult(fixture.category, dice_score(probability, fixture.truth),
                        iou_score(probability, fixture.truth), boundary_f_score(probability, fixture.truth))
                    metrics.append(measured)
                    fixture_metrics.append({"fixture_id": fixture.fixture_id, "category": fixture.category,
                        "dice": measured.dice, "iou": measured.iou, "boundary_f": measured.boundary_f})
        retained_rss, peak_rss = _memory_mb()
        aggregate = aggregate_quality(metrics)
        perf = rows[model_id]["performance"]
        perf.update(cold_ms=samples[0], warm_median_ms=statistics.median(samples[1:]),
                    warm_p95_ms=float(np.percentile(samples[1:], 95)),
                    preprocess_ms=statistics.median(preprocess_samples), postprocess_ms=statistics.median(postprocess_samples),
                    peak_rss_delta_mb=max(0.0, peak_rss-before_peak),
                    retained_rss_delta_mb=(max(0.0, retained_rss-before_current)
                        if retained_rss is not None and before_current is not None else None),
                    heartbeat_max_ms=None, cancel_max_ms=None,
                    growth_50_mb=(max(0.0, retained_rss-before_current)
                        if repeats >= 50 and retained_rss is not None and before_current is not None else None))
        if resilience_probe is not None:
            measured_resilience = dict(resilience_probe(model_id, infer, fixtures[0]))
            if set(measured_resilience) != {"heartbeat_max_ms", "cancel_max_ms"}:
                raise AcceptanceUnavailable("resilience probe returned an incomplete measurement")
            perf.update(measured_resilience)
        for index, shape in enumerate(operation_sizes):
            h, w = shape
            probability = np.zeros(shape, np.float32); probability.flags.writeable = False
            source = np.zeros((h, w, 4), np.uint8); source[..., 3] = 255
            rendered = np.zeros((h, w, 3), np.uint8)
            started = clock(); mask = derive_master_mask(probability, sensitivity=50,
                target=MaskTarget.SUBJECT); geometry_ms = (clock()-started)*1000
            started = clock(); composite_masked(rendered, source, mask, OutsideMode.WHITE)
            composite_ms = (clock()-started)*1000
            suffix = "1080p" if index == 0 else "4k"
            perf[f"geometry_{suffix}_ms"] = geometry_ms; perf[f"composite_{suffix}_ms"] = composite_ms
        rows[model_id].update(measured=True, aggregate_dice=aggregate.aggregate_dice,
            aggregate_iou=aggregate.aggregate_iou, aggregate_boundary_f=aggregate.aggregate_boundary_f,
            category_dice=dict(aggregate.category_dice), asset=dict(asset_records[model_id]),
            within_budgets=(aggregate.meets_thresholds and _operating_budgets_pass(perf)))
        rows[model_id]["fixture_metrics"] = fixture_metrics
        review = dict((manual_reviews or {}).get(model_id, {}))
        rows[model_id]["manual_review"] = review
        rows[model_id]["manually_approved"] = set(review) == set(REQUIRED_FIXTURE_CATEGORIES) and all(review.values())
        policy_reports[model_id] = CandidateReport(model_id, aggregate,
            bool(rows[model_id]["within_budgets"]), rows[model_id]["manually_approved"])
    selection = select_model_candidate(policy_reports["u2net"], policy_reports["u2netp"])
    report["selection"] = {"winner": selection.winner, "reason": selection.reason}
    report["status"] = "measured-pending-selection-validation"
    return report


def _candidate_slot(model_id: str) -> dict[str, Any]:
    """The fixed, per-model report row shape. SM-16 fills these fields in."""
    return {
        "model_id": model_id,
        "measured": False,
        "aggregate_dice": None,
        "aggregate_iou": None,
        "aggregate_boundary_f": None,
        "category_dice": {},
        "asset": {"manifest_sha256": None, "onnx_sha256": None,
                  "onnx_byte_count": None, "output_names": None,
                  "license_evidence": None, "redistribution_approved": False},
        "performance": {key: None for key in (
            "cold_ms", "warm_median_ms", "warm_p95_ms", "preprocess_ms",
            "postprocess_ms", "geometry_1080p_ms", "geometry_4k_ms",
            "composite_1080p_ms", "composite_4k_ms", "peak_rss_delta_mb",
            "retained_rss_delta_mb", "heartbeat_max_ms", "cancel_max_ms", "growth_50_mb")},
        "within_budgets": None,
        "manually_approved": False,
    }


def require_release_selection(report: dict[str, Any]) -> str:
    """Recompute the frozen policy; never trust a serialized winner or gate."""
    candidates = report.get("candidates")
    if (not isinstance(candidates, list) or len(candidates) != 2 or
            [row.get("model_id") for row in candidates].count("u2net") != 1 or
            [row.get("model_id") for row in candidates].count("u2netp") != 1):
        raise AcceptanceUnavailable("no measured eligible candidate report")
    reports: dict[str, CandidateReport] = {}
    for row in candidates:
        if not row.get("measured"):
            raise AcceptanceUnavailable("no measured eligible candidate report")
        asset = row.get("asset", {})
        names = asset.get("output_names")
        hashes_ok = all(isinstance(asset.get(key), str) and len(asset[key]) == 64 and
                        all(char in "0123456789abcdef" for char in asset[key])
                        for key in ("manifest_sha256", "onnx_sha256"))
        try:
            manifest_path, onnx_path = Path(asset["manifest_path"]), Path(asset["onnx_path"])
            local_ok = (manifest_path.is_file() and onnx_path.is_file() and
                _sha256(manifest_path) == asset["manifest_sha256"] and
                _sha256(onnx_path) == asset["onnx_sha256"] and
                isinstance(asset.get("onnx_byte_count"), int) and asset["onnx_byte_count"] > 0 and
                onnx_path.stat().st_size == asset["onnx_byte_count"])
        except (KeyError, OSError, TypeError):
            local_ok = False
        if (not hashes_ok or not asset.get("license_evidence") or
                asset.get("model_id") != row["model_id"] or not local_ok or
                not asset.get("redistribution_approved") or not isinstance(names, list)
                or len(names) != 7 or any(not isinstance(n, str) or not n.strip() for n in names)
                or len(set(names)) != 7):
            raise AcceptanceUnavailable("candidate provenance/output contract is incomplete")
        try:
            dice, iou, boundary = (float(row[key]) for key in
                ("aggregate_dice", "aggregate_iou", "aggregate_boundary_f"))
            categories = {str(k): float(v) for k, v in row["category_dice"].items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise AcceptanceUnavailable("candidate quality report is invalid") from exc
        if not categories or any(not np.isfinite(v) or v < 0 or v > 1 for v in (dice, iou, boundary, *categories.values())):
            raise AcceptanceUnavailable("candidate quality report is invalid")
        if set(categories) != set(REQUIRED_FIXTURE_CATEGORIES):
            raise AcceptanceUnavailable("candidate quality categories are incomplete")
        review = row.get("manual_review")
        if not isinstance(review, dict) or set(review) != set(REQUIRED_FIXTURE_CATEGORIES) or not all(value is True for value in review.values()):
            raise AcceptanceUnavailable("manual acceptance matrix is incomplete")
        aggregate = AggregateQuality(dice, iou, boundary, categories,
            dice >= DICE_AGGREGATE_THRESHOLD, iou >= IOU_AGGREGATE_THRESHOLD,
            boundary >= BOUNDARY_F_THRESHOLD, all(v >= CATEGORY_DICE_FLOOR for v in categories.values()))
        perf = row.get("performance", {})
        try:
            performance_values = [float(value) for value in perf.values()]
        except (TypeError, ValueError) as exc:
            raise AcceptanceUnavailable("candidate performance report is incomplete") from exc
        if len(performance_values) != 14 or any(not np.isfinite(value) or value < 0 for value in performance_values):
            raise AcceptanceUnavailable("candidate performance report is incomplete")
        measured_budget = _operating_budgets_pass(perf)
        reports[row["model_id"]] = CandidateReport(row["model_id"], aggregate,
            measured_budget, bool(row.get("manually_approved")))
    result = select_model_candidate(reports["u2net"], reports["u2netp"])
    serialized = report.get("selection")
    if result.winner is None or not isinstance(serialized, dict) or serialized.get("winner") != result.winner:
        raise AcceptanceUnavailable("serialized selection does not match frozen SM-02 policy")
    return result.winner


def build_report_skeleton(
    fixtures_manifest: str | None,
    asset_root: str | None,
) -> dict[str, Any]:
    """The fixed JSON report shape the real SM-16 runner will populate.

    No fixtures are read and no model is loaded here -- this only fixes the
    schema so SM-16 has a stable contract to fill in and the quality module's
    thresholds/policy can be exercised against it in tests before any
    candidate is ever measured.
    """
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "hardware": platform.platform(),
        "python_version": platform.python_version(),
        "status": "pending-licensed-assets-and-user-signoff",
        "boundary_tolerance_px": 2,
        "boundary_tolerance_status": "provisional-until-fixture-review",
        "fixtures_manifest": fixtures_manifest,
        "asset_root": asset_root,
        "candidates": [_candidate_slot(model_id) for model_id in sorted(APPROVED_MODEL_IDS)],
        # Populated by calling ditherzam.masking.quality.select_model_candidate
        # once every candidate row above has real measured data. None here
        # means "not yet measured", never a silent default winner.
        "selection": None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures-manifest",
        type=str,
        default=None,
        help="path to a fixture provenance manifest "
             "(see tests/fixtures/smart_mask/README.md); not yet read by this skeleton",
    )
    parser.add_argument(
        "--asset-root",
        type=str,
        default=None,
        help="staged Smart Mask asset root "
             "(see ditherzam.masking.model_assets.default_asset_root); "
             "not yet read by this skeleton",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="write the JSON report here instead of stdout",
    )
    parser.add_argument("--require-selection", action="store_true",
                        help="exit nonzero unless a measured approved winner exists")
    parser.add_argument("--candidate-manifest", action="append", default=[],
                        help="local finalized candidate manifest (specify once per candidate)")
    parser.add_argument("--approve-redistribution", action="store_true",
                        help="record explicit operator redistribution sign-off for this run")
    parser.add_argument("--manual-approved", action="append", default=[], choices=sorted(APPROVED_MODEL_IDS),
                        help="record completed manual edge review for a candidate")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.fixtures_manifest or args.candidate_manifest:
            if not args.fixtures_manifest:
                raise AcceptanceUnavailable("fixture manifest is required")
            report = execute_asset_bakeoff(args.fixtures_manifest, args.candidate_manifest,
                redistribution_approved=args.approve_redistribution,
                manual_approvals=args.manual_approved)
        else:
            report = build_report_skeleton(args.fixtures_manifest, args.asset_root)
    except (AcceptanceUnavailable, OSError, ValueError) as exc:
        print(f"Smart Mask acceptance unavailable: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.require_selection:
        try:
            require_release_selection(report)
        except AcceptanceUnavailable as exc:
            print(f"Smart Mask acceptance unavailable: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
