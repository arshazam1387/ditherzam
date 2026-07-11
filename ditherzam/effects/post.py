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


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """0..1 soft ramp between edge0 and edge1; hard step when edge1 <= edge0."""
    if edge1 <= edge0:
        return (x >= edge0).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _aniso_source(src_f: np.ndarray, aspect: float) -> tuple[Image.Image, int]:
    """Convert and horizontally compress a source once for repeated blurs."""
    h, w = src_f.shape[:2]
    ax = max(float(aspect), 1e-3)
    new_w = max(1, int(round(w / ax)))
    img = Image.fromarray(np.clip(src_f, 0, 255).astype(np.uint8))
    if new_w != w:
        img = img.resize((new_w, h), Image.BILINEAR)
    return img, w


def _aniso_blur(img: Image.Image, sigma: float, output_w: int) -> np.ndarray:
    """Blur a prepared source, restoring its original width when needed."""
    h = img.height
    new_w = img.width
    img = img.filter(ImageFilter.GaussianBlur(float(max(sigma, 0.0))))
    if new_w != output_w:
        img = img.resize((output_w, h), Image.BILINEAR)
    return np.asarray(img, np.float32)


def epsilon_glow(rgb_u8: np.ndarray, threshold: float = 64.0, smoothing: float = 32.0,
                 radius: float = 8.0, intensity: float = 1.0, epsilon: float = 0.4,
                 falloff: float = 0.5, distance_scale: float = 1.0,
                 aspect: float = 1.0) -> np.ndarray:
    """Threshold-driven bloom tuned for dithered art.

    Weighted luminance selects bright pixels through a soft knee; the emissive
    source is spread by a multi-scale anisotropic blur; `epsilon` adds a tight,
    hot core. `intensity` is the master gate (0 => identity).
    """
    if intensity <= 0:
        return rgb_u8
    base = rgb_u8.astype(np.float32)
    lum = 0.299 * base[..., 0] + 0.587 * base[..., 1] + 0.114 * base[..., 2]
    mask = _smoothstep(float(threshold), float(threshold) + float(smoothing), lum)
    del lum
    src = base * mask[..., None]

    r = max(float(radius) * float(distance_scale), 0.0)
    scales = (0.5, 1.0, 2.0)
    base_w = np.array([1.0, 0.6, 0.35], np.float32)
    f = float(np.clip(falloff, 0.0, 1.0))
    bias = np.array([1.0 + f, 1.0, 1.0 - 0.5 * f], np.float32)
    weights = base_w * bias
    weights /= weights.sum()
    glow = np.zeros_like(base)
    src_img, output_w = _aniso_source(src, aspect)
    del src
    for s, wgt in zip(scales, weights):
        glow += _aniso_blur(src_img, r * s + 1e-3, output_w) * float(wgt)

    k = 1.0 + float(np.clip(epsilon, 0.0, 1.0)) * 8.0
    core = base * (mask[..., None] ** k)
    core_img, output_w = _aniso_source(core, aspect)
    del core, mask
    core_glow = _aniso_blur(core_img, max(r * 0.25, 1e-3), output_w) * float(np.clip(epsilon, 0.0, 1.0))

    out = base + (glow + core_glow) * float(intensity)
    return np.clip(out, 0, 255).astype(np.uint8)


EFFECTS = {
    "Blur": blur,
    "Sharpen": sharpen,
    "Chromatic Aberration": chromatic_aberration,
    "JPEG Glitch": jpeg_glitch,
    "Epsilon Glow": epsilon_glow,
}
