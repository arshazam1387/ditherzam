"""Concrete, side-effect-free certification hooks shared by frozen E2E runners."""
from __future__ import annotations

from pathlib import Path


def certify_case(bundle: dict[str, Path], **dimensions: object) -> dict[str, object]:
    """Create an auditable case record after the bundle gate has succeeded.

    The Windows runner replaces measurements with observed values; collection
    never invents performance or quality results.
    """
    required = {"model", "model_manifest", "license", "notice", "provenance",
                "ort:onnxruntime.dll", "ort:onnxruntime_providers_shared.dll"}
    missing = required - bundle.keys()
    if missing:
        raise AssertionError(f"verified bundle lacks: {sorted(missing)}")
    return {"dimensions": dict(dimensions), "bundle_verified": True,
            "measurement_status": "pending-windows-run"}


def certify_resilience(bundle: dict[str, Path], scenario: str) -> dict[str, object]:
    return certify_case(bundle, scenario=scenario)
