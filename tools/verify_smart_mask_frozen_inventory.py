"""Verify the frozen distribution contains exactly the locked ORT binaries."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def verify(dist: Path, lock: Path) -> None:
    data = json.loads(lock.read_text(encoding="utf-8"))
    expected = data["ort_dlls"]
    capi = dist / "onnxruntime" / "capi"
    actual = {p.name: p for p in capi.glob("*.dll")}
    if set(actual) != set(expected):
        raise RuntimeError("frozen ORT DLL inventory differs from release lock")
    for name, path in actual.items():
        record = expected[name]
        if path.stat().st_size != record["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError(f"frozen ORT DLL is corrupt: {name}")


if __name__ == "__main__":
    if len(sys.argv) != 3: raise SystemExit("usage: verify_smart_mask_frozen_inventory.py DIST LOCK")
    verify(Path(sys.argv[1]), Path(sys.argv[2]))
