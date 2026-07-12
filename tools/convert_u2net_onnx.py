"""Developer-only, local TorchScript-to-ONNX converter for Smart Mask.

This tool never downloads code or weights.  The input must be a locally
approved TorchScript module.  It writes the ONNX graph plus a JSON conversion
record containing hashes, versions, and the graph-discovered output order.
The record is evidence for a later manifest; it does not approve an asset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="approved local TorchScript file")
    parser.add_argument("--output", required=True, type=Path, help="new .onnx output path")
    parser.add_argument("--record", required=True, type=Path, help="new JSON conversion record")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args(argv)


def convert(source: Path, output: Path, record: Path, *, opset: int) -> dict[str, object]:
    if not source.is_file() or source.suffix.lower() not in {".pt", ".pth", ".torchscript"}:
        raise ValueError("input must be an existing local TorchScript file")
    if output.suffix.lower() != ".onnx" or output.exists() or record.exists():
        raise ValueError("output/record must be new paths and output must end in .onnx")
    if opset < 12:
        raise ValueError("opset must be at least 12")
    try:
        import torch
        import onnx
    except ImportError as exc:
        raise RuntimeError("developer conversion requires locally installed torch and onnx") from exc

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = torch.jit.load(str(source), map_location="cpu").eval()
    sample = torch.zeros((1, 3, 320, 320), dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, sample, str(output), input_names=["input.1"],
                      opset_version=opset, do_constant_folding=True)
    graph = onnx.load(str(output), load_external_data=False)
    output_names = [item.name.strip() for item in graph.graph.output]
    if len(output_names) != 7 or any(not name for name in output_names) or len(set(output_names)) != 7:
        output.unlink(missing_ok=True)
        raise RuntimeError("converted graph must expose exactly seven unique non-blank outputs")
    payload: dict[str, object] = {
        "status": "conversion-evidence-only-not-approved",
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "onnx_path": str(output.resolve()),
        "onnx_sha256": sha256_file(output),
        "onnx_byte_count": output.stat().st_size,
        "opset": opset,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "output_names": output_names,
    }
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    convert(args.input, args.output, args.record, opset=args.opset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
