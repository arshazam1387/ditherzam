from __future__ import annotations

import numpy as np

_WHITE = "#ffffff"
_BLACK = "#000000"


def raster_to_svg(gray_u8: np.ndarray, threshold: int, invert: bool = False) -> str:
    """Convert a 2-D grayscale array to an optimized SVG.

    Pixels strictly below ``threshold`` are "filled". Vertical runs of adjacent
    filled pixels in the same column are merged into a single <rect> to keep the
    element count low (spec §11.3). No external dependencies.
    """
    arr = np.asarray(gray_u8)
    if arr.ndim != 2:
        raise ValueError("raster_to_svg expects a 2-D grayscale array")
    h, w = arr.shape

    bg = _BLACK if invert else _WHITE
    fg = _WHITE if invert else _BLACK
    filled = arr < threshold

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" version="1.1">',
        f'<rect width="100%" height="100%" fill="{bg}"/>',
    ]
    for x in range(w):
        col = filled[:, x]
        y = 0
        while y < h:
            if col[y]:
                start = y
                while y < h and col[y]:
                    y += 1
                run = y - start
                parts.append(
                    f'<rect x="{x}" y="{start}" width="1" height="{run}" fill="{fg}"/>'
                )
            else:
                y += 1
    parts.append("</svg>")
    return "\n".join(parts)
