from __future__ import annotations
import numpy as np
from PIL import Image


def to_gray_f32(src) -> np.ndarray:
    if isinstance(src, Image.Image):
        arr = np.array(src.convert("L"), dtype=np.float32)
    else:
        arr = np.asarray(src, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=2)
    return arr.astype(np.float32)


def clamp_u8(arr) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def nearest_downscale(gray_f32: np.ndarray, factor: int) -> np.ndarray:
    factor = max(1, int(factor))
    h, w = gray_f32.shape[:2]
    pil = Image.fromarray(clamp_u8(gray_f32))
    small = pil.resize((max(1, w // factor), max(1, h // factor)), Image.NEAREST)
    return np.array(small, dtype=np.float32)


def nearest_upscale_to(small_f32: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(clamp_u8(small_f32))
    return np.array(pil.resize(size_wh, Image.NEAREST), dtype=np.float32)
