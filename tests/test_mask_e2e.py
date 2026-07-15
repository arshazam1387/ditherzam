"""Real-model E2E matrix; collected now, enabled only after asset approval."""
from pathlib import Path
import hashlib
import json

import pytest

from ditherzam.masking.release_gate import ReleaseBundleError, verify_release_bundle
from benchmarks.smart_mask_certification import certify_case, certify_resilience

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "packaging" / "smart-mask-release.lock.json"


def _approved_bundle_or_skip():
    if not LOCK.is_file():
        pytest.skip("pending selected licensed Smart Mask asset")
    try:
        return verify_release_bundle(ROOT, LOCK)
    except ReleaseBundleError as exc:
        pytest.fail(f"configured release bundle must fail closed: {exc}")


def _evidence(case):
    key = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
    path = ROOT / "packaging" / "certification-evidence" / f"{key}.json"
    if not path.is_file():
        pytest.fail(f"approved bundle lacks concrete case evidence: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("source_kind", ["rgb", "rgba", "grayscale"])
@pytest.mark.parametrize("outside", ["original", "transparent", "black", "white"])
@pytest.mark.parametrize("target,invert", [("disabled", False), ("subject", False), ("subject", True), ("background", False), ("background", True), ("whole", False)])
@pytest.mark.parametrize("surface", ["capped", "full", "png-exact", "jpeg-white-flatten"])
def test_real_model_render_export_matrix_pending_asset(source_kind, outside, target, invert, surface):
    bundle = _approved_bundle_or_skip()
    case = dict(source_kind=source_kind, outside=outside, target=target, invert=invert, surface=surface)
    record = certify_case(bundle, _evidence(case), **case)
    assert record["executed"]


@pytest.mark.parametrize("scenario", [
    "no-subject", "oom", "cancel", "source-replacement", "stale-progress",
    "disabled-byte-baseline-zero-work", "source-colors", "colored-dither",
    "effects-invert", "overlay-excluded", "preview-full-export", "latest-wins",
    "terminal-recovery", "unsupported-media", "fifty-cycle-rss-cache-192mib",
    "performance-fields", "quality-fields", "heartbeat-cancel-fields", "missing", "corrupt",
    "offline-frozen-startup",
])
def test_real_model_resilience_matrix_pending_asset(scenario):
    bundle = _approved_bundle_or_skip()
    case = {"scenario": scenario}
    assert certify_resilience(bundle, scenario, _evidence(case))["executed"]
