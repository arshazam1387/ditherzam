"""Real-model E2E matrix; collected now, enabled only after asset approval."""
from pathlib import Path

import pytest

from ditherzam.masking.release_gate import ReleaseBundleError, verify_release_bundle

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "packaging" / "smart-mask-release.lock.json"


def _approved_bundle_or_skip():
    if not LOCK.is_file():
        pytest.skip("pending selected licensed Smart Mask asset")
    try:
        return verify_release_bundle(ROOT, LOCK)
    except ReleaseBundleError as exc:
        pytest.fail(f"configured release bundle must fail closed: {exc}")


@pytest.mark.parametrize("source_kind", ["rgb", "rgba", "grayscale"])
@pytest.mark.parametrize("outside", ["original", "transparent", "black", "white"])
@pytest.mark.parametrize("target,invert", [("subject", False), ("subject", True), ("background", False), ("background", True)])
def test_real_model_render_export_matrix_pending_asset(source_kind, outside, target, invert):
    bundle = _approved_bundle_or_skip()
    pytest.fail(f"approved asset E2E adapter not completed for {source_kind}/{outside}/{target}/{invert}: {bundle}")


@pytest.mark.parametrize("scenario", [
    "no-subject", "oom", "cancel", "source-replacement", "stale-progress",
    "source-colors", "colored-dither", "effects-invert", "preview-full-export",
    "unsupported-media", "fifty-cycle-rss", "offline-frozen-startup",
])
def test_real_model_resilience_matrix_pending_asset(scenario):
    bundle = _approved_bundle_or_skip()
    pytest.fail(f"approved asset E2E resilience adapter not completed for {scenario}: {bundle}")
