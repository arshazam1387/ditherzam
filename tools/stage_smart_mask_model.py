"""Developer-only Smart Mask model staging tool.

Fetches an approved upstream source checkpoint, verifies it against a pinned
SHA-256, and stages the raw bytes locally for the (separate, not-yet-built)
reproducible ONNX conversion step. This is the ONLY place in the repository
allowed to reach the network, and only when a developer runs it explicitly by
hand or a release pipeline invokes it out-of-band. Nothing in ``ditherzam/``
imports or calls this script; see ``tests/test_offline_security.py``.

Conversion to ONNX, opset pinning, and output-hash verification are the
licensed bakeoff/release task's responsibility, not this script's. This tool
only gets the exact upstream bytes onto disk and proves they match what was
promised before anyone touches them.

Usage (never run by the application):

    .venv/Scripts/python.exe tools/stage_smart_mask_model.py \\
        --source-url <approved-upstream-asset-location> \\
        --expected-sha256 <pinned-hex-digest> \\
        --out assets/models/smart_mask/_staging/u2netp-source.bin
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

_STREAM_CHUNK_BYTES = 1024 * 1024


def stage_source_asset(source_url: str, expected_sha256: str, out_path: Path) -> None:
    """Fetch ``source_url`` and refuse to keep it unless its SHA-256 matches.

    Never invoked by application code. Only reachable by running this script
    directly, and only for a hash the developer already pinned in a manifest.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")

    hasher = hashlib.sha256()
    with urllib.request.urlopen(source_url) as response, tmp_path.open("wb") as handle:
        while True:
            chunk = response.read(_STREAM_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
            handle.write(chunk)

    digest = hasher.hexdigest()
    expected = expected_sha256.strip().lower()
    if digest != expected:
        tmp_path.unlink(missing_ok=True)
        raise SystemExit(
            f"Staged asset SHA-256 {digest} does not match expected {expected}; "
            "refusing to stage an unverified file."
        )

    tmp_path.replace(out_path)
    print(f"Staged {out_path} ({out_path.stat().st_size} bytes, sha256={digest})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", required=True, help="Approved upstream asset location.")
    parser.add_argument("--expected-sha256", required=True, help="Pinned source SHA-256 hex digest.")
    parser.add_argument("--out", required=True, type=Path, help="Local staging output path.")
    args = parser.parse_args(argv)

    stage_source_asset(args.source_url, args.expected_sha256, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
