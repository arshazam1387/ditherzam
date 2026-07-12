"""Fail-closed verification for a frozen Smart Mask release bundle.

This module does not build, retrieve, or select assets.  It validates the files
already placed in a frozen distribution by the release process.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from .model_assets import ModelAssetError, load_manifest, verify_model_asset


class ReleaseBundleError(RuntimeError):
    """The frozen distribution is incomplete or does not match its lock file."""


REQUIRED_RECORDS = frozenset({"model_manifest", "license", "notice", "provenance"})


def verify_release_bundle(bundle_root: str | Path, lock_path: str | Path) -> dict[str, Path]:
    """Verify every content-addressed release file before frozen-app smoke tests."""
    root = Path(bundle_root).resolve()
    lock = Path(lock_path)
    try:
        data = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReleaseBundleError(f"release lock is missing or invalid: {exc}") from exc
    if data.get("status") != "approved":
        raise ReleaseBundleError("Smart Mask release asset selection is pending approval")
    if data.get("onnxruntime") != "1.22.1":
        raise ReleaseBundleError("release lock must pin onnxruntime 1.22.1")
    records = data.get("files")
    if not isinstance(records, dict) or not REQUIRED_RECORDS <= records.keys():
        raise ReleaseBundleError("release lock lacks model/license/NOTICE/provenance records")
    dlls = data.get("ort_dlls")
    if not isinstance(dlls, dict) or not dlls:
        raise ReleaseBundleError("release lock lacks content-addressed ORT DLL records")

    verified = {name: _verify_record(root, name, record) for name, record in records.items()}
    verified.update({f"ort:{name}": _verify_record(root, name, record)
                     for name, record in dlls.items()})
    try:
        manifest = load_manifest(verified["model_manifest"])
        verified["model"] = verify_model_asset(root, manifest)
    except ModelAssetError as exc:
        raise ReleaseBundleError(f"model asset verification failed: {exc}") from exc
    return verified


def _verify_record(root: Path, name: str, record: object) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ReleaseBundleError(f"{name} record must contain only path and sha256")
    relative = record["path"]
    digest = record["sha256"]
    if not isinstance(relative, str) or not relative.strip():
        raise ReleaseBundleError(f"{name} path is invalid")
    if (PureWindowsPath(relative).is_absolute() or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative.replace("\\", "/")).parts):
        raise ReleaseBundleError(f"{name} path escapes the frozen bundle")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ReleaseBundleError(f"{name} SHA-256 is invalid")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ReleaseBundleError(f"{name} path escapes the frozen bundle") from None
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ReleaseBundleError(f"{name} is missing or corrupt")
    return path
