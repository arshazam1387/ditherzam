from __future__ import annotations

import numpy as np

PATTERNS: tuple[str, ...] = (
    "static",
    "scanline-drift",
    "interlace",
    "rolling-bar",
    "vhs-jitter",
    "blue-noise",
    "bayer-cycle",
    "plasma",
    "film-grain",
)

_TWO_PI = 2.0 * np.pi


def _rng(frame: int, seed: int) -> np.random.Generator:
    """Deterministic generator keyed on (frame, seed)."""
    return np.random.default_rng(int(seed) * 1_000_003 + int(frame))


def _bayer4() -> np.ndarray:
    b2 = np.array([[0, 2], [3, 1]], dtype=np.float32)
    return np.block([
        [4 * b2 + 0, 4 * b2 + 2],
        [4 * b2 + 3, 4 * b2 + 1],
    ]).astype(np.float32)  # values 0..15


def _static(frame: int, h: int, w: int, seed: int) -> np.ndarray:
    # TV "snow": full-field white noise, re-rolled every frame. -> [-1, 1)
    u = _rng(frame, seed).random((h, w), dtype=np.float32)
    return u * 2.0 - 1.0


def _scanline_drift(frame: int, h: int, w: int, seed: int) -> np.ndarray:
    # Horizontal scanlines whose phase drifts vertically over time. -> [-1, 1]
    y = np.arange(h, dtype=np.float32)[:, None]
    row = np.sin(_TWO_PI * (y / 8.0 + frame * 0.05))
    return np.broadcast_to(row, (h, w)).astype(np.float32)


def _interlace(frame: int, h: int, w: int, seed: int) -> np.ndarray:
    # Alternating even/odd lines that flip parity each frame. -> {-1, +1}
    y = np.arange(h, dtype=np.int64)[:, None]
    val = ((y + frame) % 2) * 2 - 1
    return np.broadcast_to(val.astype(np.float32), (h, w)).astype(np.float32)


def _rolling_bar(frame: int, h: int, w: int, seed: int) -> np.ndarray:
    # A bright horizontal hum-bar rolling vertically (rolling-shutter look). -> [0, 1]
    y = np.arange(h, dtype=np.float32)[:, None]
    center = (frame * 2.0) % max(h, 1)
    d = np.abs(y - center)
    d = np.minimum(d, h - d)                       # wrap vertically
    sigma = max(h / 8.0, 1.0)
    prof = np.exp(-(d * d) / (2.0 * sigma * sigma))
    return np.broadcast_to(prof, (h, w)).astype(np.float32)


def _vhs_jitter(frame: int, h: int, w: int, seed: int) -> np.ndarray:
    # Per-row horizontal jitter offset, re-rolled each frame. -> [-1, 1)
    row = _rng(frame, seed).random(h, dtype=np.float32) * 2.0 - 1.0
    return np.broadcast_to(row[:, None], (h, w)).astype(np.float32)


def _blue_noise(frame: int, h: int, w: int, seed: int) -> np.ndarray:
    # High-frequency noise: white noise minus its 4-neighbour blur, then
    # normalised to exactly [-1, 1]. -> [-1, 1]
    n = _rng(frame, seed).random((h, w), dtype=np.float32)
    blur = 0.25 * (np.roll(n, 1, 0) + np.roll(n, -1, 0)
                   + np.roll(n, 1, 1) + np.roll(n, -1, 1))
    hp = n - blur
    m = float(np.max(np.abs(hp)))
    if m <= 0.0:
        return np.zeros((h, w), dtype=np.float32)
    return (hp / m).astype(np.float32)


def _bayer_cycle(frame: int, h: int, w: int, seed: int) -> np.ndarray:
    # 4x4 Bayer threshold field, phase-rolled each frame, tiled to HxW. -> ~[-0.94, 0.94]
    b = _bayer4()
    shift = frame % 4
    b = np.roll(np.roll(b, shift, axis=0), shift, axis=1)
    norm = (b + 0.5) / 16.0 * 2.0 - 1.0
    tiled = np.tile(norm, (h // 4 + 1, w // 4 + 1))[:h, :w]
    return tiled.astype(np.float32)


def _plasma(frame: int, h: int, w: int, seed: int) -> np.ndarray:
    # Classic plasma: mean of four moving sines. -> [-1, 1]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = frame * 0.1
    p = (np.sin(xx / 6.0 + t)
         + np.sin(yy / 8.0 - t)
         + np.sin((xx + yy) / 10.0 + t)
         + np.sin(np.sqrt(xx * xx + yy * yy) / 7.0 + t))
    return (p / 4.0).astype(np.float32)


def _film_grain(frame: int, h: int, w: int, seed: int) -> np.ndarray:
    # Gaussian grain clipped to +/-3 sigma, scaled to [-1, 1]. -> [-1, 1]
    g = _rng(frame, seed).standard_normal((h, w)).astype(np.float32)
    return np.clip(g, -3.0, 3.0) / 3.0


_DISPATCH = {
    "static": _static,
    "scanline-drift": _scanline_drift,
    "interlace": _interlace,
    "rolling-bar": _rolling_bar,
    "vhs-jitter": _vhs_jitter,
    "blue-noise": _blue_noise,
    "bayer-cycle": _bayer_cycle,
    "plasma": _plasma,
    "film-grain": _film_grain,
}


def temporal_noise(frame: int, shape: tuple[int, int], pattern: str,
                   amplitude: float, seed: int = 0) -> np.ndarray:
    """Deterministic per-frame noise field, float32 HxW in ~[-amplitude, amplitude]."""
    if pattern not in _DISPATCH:
        raise KeyError(f"Unknown temporal pattern: {pattern!r}")
    h, w = int(shape[0]), int(shape[1])
    base = _DISPATCH[pattern](int(frame), h, w, int(seed))
    return (base * float(amplitude)).astype(np.float32)
