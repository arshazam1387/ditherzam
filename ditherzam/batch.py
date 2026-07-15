from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .imaging import to_gray_f32
from .export.raster import save_raster

_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def batch_process(folder, out_folder, settings, pipeline, ref_size) -> tuple[int, int]:
    """Render every image in ``folder`` whose (w,h) equals ``ref_size``.

    Returns ``(processed, skipped)``. Outputs are written as PNG into
    ``out_folder`` using each source file's stem. Non-image files are ignored;
    images with a mismatched size are counted as skipped.
    """
    folder = Path(folder)
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    for src in sorted(folder.iterdir()):
        if not src.is_file() or src.suffix.lower() not in _EXTS:
            continue
        with Image.open(src) as im:
            if im.size != tuple(ref_size):
                skipped += 1
                continue
            gray = to_gray_f32(im)
        result = pipeline.render(gray, settings)
        save_raster(np.asarray(result, dtype=np.uint8), out_folder / f"{src.stem}.png")
        processed += 1
    return processed, skipped
