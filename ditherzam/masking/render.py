"""Thin outer-render integration for Smart Mask.

The disabled path deliberately calls the historical renderer directly.  Keep
all mask validation and geometry below that branch so disabled documents pay
no masking cost and retain their exact historical bytes.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image

from .composite import composite_masked
from .geometry import derive_master_mask, resize_mask_area


def _resize_source_rgba(source: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if source.shape[:2] == shape:
        return source
    height, width = shape
    # Match the preview proxy's nearest-neighbour source sampling.  The PIL
    # result owns its storage, avoiding a borrowed/zero-copy Qt lifetime.
    return np.asarray(
        Image.fromarray(source, mode="RGBA").resize(
            (width, height), resample=Image.Resampling.NEAREST),
        dtype=np.uint8,
    )


def render_with_mask(renderer: Callable[[], np.ndarray], mask_context=None) -> np.ndarray:
    """Render one complete branch and optionally outer-composite its mask.

    ``mask_context is None`` is the explicit historical bypass: no source hash,
    mask derivation, resizing, compositing, or mask allocation occurs.
    """
    if mask_context is None:
        return renderer()

    rendered = renderer()
    settings = mask_context.settings
    source = mask_context.source_rgba
    master = derive_master_mask(
        mask_context.probability,
        sensitivity=settings.sensitivity,
        target=settings.target,
        invert=settings.invert,
        expansion_px=settings.expansion_px,
        feather_px=settings.feather_px,
        source_shape=source.shape[:2],
    )
    target_shape = rendered.shape[:2]
    mask = master if master.shape == target_shape else resize_mask_area(master, target_shape)
    target_source = _resize_source_rgba(source, target_shape)
    return composite_masked(rendered, target_source, mask, settings.outside)
