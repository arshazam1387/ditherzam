"""Smart Mask quality/performance bakeoff report -- SKELETON (SM-02).

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
import json
import platform
import sys
from pathlib import Path
from typing import Any

from ditherzam.masking.model_assets import APPROVED_MODEL_IDS

REPORT_SCHEMA_VERSION = 1


def _candidate_slot(model_id: str) -> dict[str, Any]:
    """The fixed, per-model report row shape. SM-16 fills these fields in."""
    return {
        "model_id": model_id,
        "measured": False,
        "aggregate_dice": None,
        "aggregate_iou": None,
        "aggregate_boundary_f": None,
        "category_dice": {},
        "warm_median_ms": None,
        "warm_p95_ms": None,
        "cold_ms": None,
        "peak_memory_mb": None,
        "retained_memory_mb": None,
        "within_budgets": None,
        "manually_approved": False,
    }


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report_skeleton(args.fixtures_manifest, args.asset_root)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
