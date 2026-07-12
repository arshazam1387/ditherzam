from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage


def numpy_to_qimage(rgb_u8: np.ndarray) -> QImage:
    """Convert uint8 gray/RGB/RGBA into a standalone QImage.

    The returned image owns its pixels (``.copy()``) so it is safe after the
    numpy source is garbage-collected.
    """
    arr = np.asarray(rgb_u8)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("image must be HxW, HxWx3, or HxWx4")
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    if arr.shape[2] == 4:
        qimg = QImage(arr.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    else:
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return qimg.copy()


def qimage_to_numpy(qimg: QImage) -> np.ndarray:
    """Convert any QImage into a contiguous HxWx3 uint8 array (RGB order)."""
    img = qimg.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    bpl = img.bytesPerLine()
    buf = np.frombuffer(memoryview(img.constBits()), dtype=np.uint8, count=h * bpl)
    return buf.reshape(h, bpl)[:, : w * 3].reshape(h, w, 3).copy()
