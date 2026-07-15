"""Offline frozen-build smoke launcher used after an approved build exists."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ditherzam.masking.release_gate import verify_release_bundle


def main(executable: str) -> int:
    root = Path(__file__).resolve().parents[1]
    verify_release_bundle(root, root / "packaging" / "smart-mask-release.lock.json")
    exe = Path(executable).resolve()
    if not exe.is_file() or exe.suffix.lower() != ".exe":
        raise SystemExit("frozen Windows executable is missing")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", DITHERZAM_OFFLINE_SMOKE="1")
    return subprocess.run([str(exe), "--offline-smoke"], env=env, timeout=30,
                          check=False).returncode


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: smoke_smart_mask_release.py <ditherzam.exe>")
    raise SystemExit(main(sys.argv[1]))
