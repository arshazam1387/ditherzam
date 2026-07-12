from pathlib import Path

import pytest

from ditherzam.masking.release_gate import ReleaseBundleError, verify_release_bundle


def test_frozen_smoke_is_explicitly_pending_selected_asset():
    root = Path(__file__).resolve().parents[1]
    lock = root / "packaging" / "smart-mask-release.lock.example.json"
    with pytest.raises(ReleaseBundleError, match="pending approval"):
        verify_release_bundle(root, lock)
