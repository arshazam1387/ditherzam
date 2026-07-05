from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter


def blur(rgb_u8: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian blur; radius <= 0 is the identity."""
    if radius <= 0:
        return rgb_u8
    pil = Image.fromarray(rgb_u8).filter(ImageFilter.GaussianBlur(float(radius)))
    return np.asarray(pil, np.uint8)


def sharpen(rgb_u8: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp mask: out = a + (a - blur(a)) * amount, clipped to 0..255."""
    pil = Image.fromarray(rgb_u8)
    blurred = pil.filter(ImageFilter.GaussianBlur(2))
    a = np.asarray(pil, np.float32)
    b = np.asarray(blurred, np.float32)
    return np.clip(a + (a - b) * amount, 0, 255).astype(np.uint8)


def chromatic_aberration(rgb_u8: np.ndarray, shift: int) -> np.ndarray:
    """Roll the red channel right by `shift` and the blue channel left; green stays."""
    out = rgb_u8.copy()
    s = int(shift)
    out[..., 0] = np.roll(rgb_u8[..., 0], s, axis=1)      # red -> right
    out[..., 2] = np.roll(rgb_u8[..., 2], -s, axis=1)     # blue -> left
    return out


def jpeg_glitch(rgb_u8: np.ndarray, quality: int) -> np.ndarray:
    """Lossy JPEG round-trip; low quality introduces block/DCT artifacts."""
    q = int(max(1, min(100, quality)))
    buf = io.BytesIO()
    Image.fromarray(rgb_u8).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), np.uint8)


def epsilon_glow(rgb_u8: np.ndarray, radius: float, strength: float) -> np.ndarray:
    """Additive bloom tuned for dithered art: base + blur(base) * strength."""
    pil = Image.fromarray(rgb_u8)
    glow = np.asarray(pil.filter(ImageFilter.GaussianBlur(float(radius))), np.float32)
    base = np.asarray(pil, np.float32)
    return np.clip(base + glow * strength, 0, 255).astype(np.uint8)


EFFECTS = {
    "Blur": blur,
    "Sharpen": sharpen,
    "Chromatic Aberration": chromatic_aberration,
    "JPEG Glitch": jpeg_glitch,
    "Epsilon Glow": epsilon_glow,
}
