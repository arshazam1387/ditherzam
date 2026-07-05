from __future__ import annotations
import numpy as np
from PIL import Image, ImageFilter
from .imaging import clamp_u8


def apply_contrast(img: np.ndarray, value: float) -> np.ndarray:
    return (img * (value / 50.0)).astype(np.float32)


def apply_midtones(img: np.ndarray, value: float) -> np.ndarray:
    gamma = max(1.0 + (value - 50) / 200.0, 0.1)
    return (255.0 * (img / 255.0) ** (1.0 / gamma)).astype(np.float32)


def apply_highlights(img: np.ndarray, value: float) -> np.ndarray:
    return (img * (1.0 + (value - 50) / 100.0)).astype(np.float32)


def apply_blur(img: np.ndarray, value: float) -> np.ndarray:
    radius = (value / 10.0) ** 2
    if radius <= 0:
        return img
    pil = Image.fromarray(clamp_u8(img))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(pil, dtype=np.float32)


def apply_invert(img: np.ndarray, enabled: bool) -> np.ndarray:
    return (255.0 - img).astype(np.float32) if enabled else img


def apply_saturation(rgb: np.ndarray, value: float) -> np.ndarray:
    """Scale color saturation about per-pixel luminance.

    value in 0..100; 50 = identity, 0 = grayscale, 100 = 2x saturation.
    """
    factor = value / 50.0
    lum = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )[..., None]
    return (lum + (rgb - lum) * factor).astype(np.float32)
