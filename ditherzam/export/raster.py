from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_raster(rgb_u8: np.ndarray, path) -> Path:
    """Save an HxWx3 (or HxW grayscale) uint8 array as PNG/JPG by file extension."""
    path = Path(path)
    arr = np.clip(np.asarray(rgb_u8), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        img.convert("RGB").save(path, "JPEG", quality=95)
    else:
        img.save(path)
    return path
