"""Strict, bounded PNG interchange for raster layer masks."""
from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_RASTER_MASK_PIXELS = 2**26
_SUPPORTED_MODES = frozenset({"L", "1", "RGB", "RGBA", "P"})


class RasterMaskIOError(ValueError):
    """A raster-mask PNG could not be safely imported or exported."""


def _shape(value: object) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
               for v in value)
    ):
        raise RasterMaskIOError(
            "expected dimensions must be a positive (height, width) tuple")
    return value


def import_raster_mask_png(
    path: str | os.PathLike[str],
    expected_shape: tuple[int, int],
) -> np.ndarray:
    expected_height, expected_width = _shape(expected_shape)
    try:
        with Path(path).open("rb") as stream:
            header = stream.read(24)
        if (
            len(header) != 24
            or header[:8] != b"\x89PNG\r\n\x1a\n"
            or header[12:16] != b"IHDR"
        ):
            raise RasterMaskIOError("raster mask import requires an actual PNG file")
        width, height = struct.unpack(">II", header[16:24])
        if width * height > MAX_RASTER_MASK_PIXELS:
            raise RasterMaskIOError(
                f"raster mask exceeds the {MAX_RASTER_MASK_PIXELS} pixel limit")
        with Image.open(Path(path)) as image:
            if image.format != "PNG":
                raise RasterMaskIOError("raster mask import requires an actual PNG file")
            if image.size != (width, height):
                raise RasterMaskIOError("raster mask PNG header is inconsistent")
            if (height, width) != (expected_height, expected_width):
                raise RasterMaskIOError(
                    "raster mask dimensions do not match the selected layer source")
            if image.mode not in _SUPPORTED_MODES:
                raise RasterMaskIOError(
                    f"unsupported raster mask PNG mode: {image.mode}")
            if image.mode == "L":
                converted = image
            elif image.mode == "1":
                converted = image.convert("L")
            else:
                converted = image.convert("RGB").convert("L")
            pixels = np.array(converted, dtype=np.uint8, order="C", copy=True)
    except RasterMaskIOError:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise RasterMaskIOError("could not decode raster mask PNG") from exc
    return pixels


def export_raster_mask_png(
    path: str | os.PathLike[str],
    pixels: np.ndarray,
) -> None:
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.uint8
        or pixels.ndim != 2
        or not pixels.size
    ):
        raise RasterMaskIOError(
            "raster mask export requires a non-empty 2-D uint8 array")
    target = Path(path)
    target.parent.mkdir(parents=False, exist_ok=True)
    temporary: Path | None = None
    try:
        handle = tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".tmp",
            dir=target.parent, delete=False)
        temporary = Path(handle.name)
        handle.close()
        Image.fromarray(pixels, mode="L").save(
            temporary, format="PNG", optimize=False)
        os.replace(temporary, target)
        temporary = None
    except (OSError, ValueError) as exc:
        raise RasterMaskIOError("could not export raster mask PNG") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
