from __future__ import annotations
import numpy as np
from numba import njit, prange
from PIL import Image, ImageFilter
from .imaging import clamp_u8


def apply_contrast(img: np.ndarray, value: float, out: np.ndarray | None = None) -> np.ndarray:
    if out is None:
        return (img * (value / 50.0)).astype(np.float32)
    # Same ufunc computation as the allocating path, just targeting a caller-
    # owned buffer (proven byte-identical: benchmarks/adjustment_fusion.py).
    np.multiply(img, value / 50.0, out=out)
    return out


def apply_midtones(img: np.ndarray, value: float, out: np.ndarray | None = None) -> np.ndarray:
    gamma = max(1.0 + (value - 50) / 200.0, 0.1)
    if out is None:
        return (255.0 * (img / 255.0) ** (1.0 / gamma)).astype(np.float32)
    np.divide(img, 255.0, out=out)
    np.power(out, 1.0 / gamma, out=out)
    np.multiply(out, 255.0, out=out)
    return out


def apply_highlights(img: np.ndarray, value: float, out: np.ndarray | None = None) -> np.ndarray:
    if out is None:
        return (img * (1.0 + (value - 50) / 100.0)).astype(np.float32)
    np.multiply(img, 1.0 + (value - 50) / 100.0, out=out)
    return out


def apply_blur(img: np.ndarray, value: float) -> np.ndarray:
    radius = (value / 10.0) ** 2
    if radius <= 0:
        return img
    pil = Image.fromarray(clamp_u8(img))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(pil, dtype=np.float32)


def apply_invert(img: np.ndarray, enabled: bool, out: np.ndarray | None = None) -> np.ndarray:
    if not enabled:
        return img
    if out is None:
        return (255.0 - img).astype(np.float32)
    # Same subtraction as the allocating path, targeting a caller-owned
    # buffer; skips the redundant uint8->float32 pre-cast the allocating
    # path needs (np.subtract casts internally). Proven byte-identical:
    # tests/test_render_scratch_reuse.py.
    np.subtract(255.0, img, out=out)
    return out


@njit(cache=True, parallel=True)
def _saturation_rgb_u8(rgb: np.ndarray, factor: np.float32) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = np.empty((h, w, 3), dtype=np.uint8)
    for y in prange(h):
        for x in range(w):
            r = np.float32(rgb[y, x, 0])
            g = np.float32(rgb[y, x, 1])
            b = np.float32(rgb[y, x, 2])
            lum = ((np.float32(0.299) * r + np.float32(0.587) * g)
                   + np.float32(0.114) * b)
            for channel, source in ((0, r), (1, g), (2, b)):
                adjusted = lum + (source - lum) * factor
                adjusted = min(np.float32(255.0), max(np.float32(0.0), adjusted))
                out[y, x, channel] = adjusted
    return out


@njit(cache=True, parallel=True)
def _saturation_gray_u8(gray: np.ndarray, factor: np.float32) -> np.ndarray:
    h, w = gray.shape
    out = np.empty((h, w, 3), dtype=np.uint8)
    for y in prange(h):
        for x in range(w):
            source = np.float32(gray[y, x])
            # Preserve the legacy RGB-repeat calculation, including its rare
            # neutral-50 one-byte truncation differences.
            lum = ((np.float32(0.299) * source + np.float32(0.587) * source)
                   + np.float32(0.114) * source)
            adjusted = lum + (source - lum) * factor
            adjusted = min(np.float32(255.0), max(np.float32(0.0), adjusted))
            byte = np.uint8(adjusted)
            out[y, x, 0] = byte
            out[y, x, 1] = byte
            out[y, x, 2] = byte
    return out


def apply_saturation_u8(gray_or_rgb: np.ndarray, value: float) -> np.ndarray:
    """Apply saturation and clamp directly into the final RGB uint8 frame."""
    image = np.asarray(gray_or_rgb)
    factor = np.float32(value / 50.0)
    if image.ndim == 2:
        return _saturation_gray_u8(image, factor)
    return _saturation_rgb_u8(image[..., :3], factor)


def apply_saturation(rgb: np.ndarray, value: float, *, output_u8: bool = False) -> np.ndarray:
    """Scale color saturation about per-pixel luminance.

    value in 0..100; 50 = identity, 0 = grayscale, 100 = 2x saturation.
    """
    if output_u8:
        return apply_saturation_u8(rgb, value)
    factor = value / 50.0
    lum = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )[..., None]
    return (lum + (rgb - lum) * factor).astype(np.float32)
