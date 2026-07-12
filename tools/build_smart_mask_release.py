"""Local release entry point. It never retrieves assets."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ditherzam.masking.release_gate import verify_release_bundle


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    lock = root / "packaging" / "smart-mask-release.lock.json"
    verify_release_bundle(root, lock)
    if sys.version_info[:2] != (3, 12) or sys.platform != "win32":
        raise SystemExit("Smart Mask frozen release requires Windows Python 3.12")
    code = subprocess.call([sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm",
                            str(root / "packaging" / "ditherzam-smart-mask.spec")], cwd=root)
    if code == 0:
        from tools.verify_smart_mask_frozen_inventory import verify
        verify(root / "dist" / "ditherzam", lock)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
