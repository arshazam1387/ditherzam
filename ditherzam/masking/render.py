"""Thin outer-render integration for Smart Mask.

The disabled path deliberately calls the historical renderer directly.  Keep
all mask validation and geometry below that branch so disabled documents pay
no masking cost and retain their exact historical bytes.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image

from ditherzam.render import RenderCancelled
from .cache import CompositeIdentity, MaskCaches
from .composite import bake_outside_base, composite_masked
from .contracts import MaskIdentity
from .geometry import (
    FEATHER_ALGORITHM_VERSION, derive_master_mask, derive_preview_mask, resize_mask_area,
)
from .settings import OutsideMode

ALPHA_ALGORITHM_VERSION = "straight-u8-v1"


def _cancel(is_cancelled) -> None:
    if is_cancelled is not None and is_cancelled():
        raise RenderCancelled


def _mask_identity(context) -> MaskIdentity:
    settings = context.settings
    return MaskIdentity(
        context.probability.identity, settings.sensitivity, settings.target,
        settings.invert, settings.expansion_px, settings.feather_px,
        FEATHER_ALGORITHM_VERSION,
    )


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


def bake_fill_active(settings) -> bool:
    """True when the White/Black outside fill should be baked pre-dither."""
    return bool(settings.bake_fill) and settings.outside in (
        OutsideMode.WHITE, OutsideMode.BLACK)


def _derive_cached_master(mask_context, derive_shape, source_shape, *, caches, is_cancelled):
    """Return the master mask derived at ``derive_shape``, via the derived cache.

    ``derive_shape == source_shape`` is the exact full-resolution derivation used
    by exports and Full previews; any other shape is a capped preview derived
    directly at that resolution. Returns ``(master, mask_identity, derived_new)``
    so the caller owns cache publication (kept atomic with compositing).
    """
    settings = mask_context.settings
    mask_identity = _mask_identity(mask_context)
    master = caches.get_derived(mask_identity, derive_shape) if caches is not None else None
    derived_new = master is None
    if master is None:
        if derive_shape == source_shape:
            master = derive_master_mask(
                mask_context.probability,
                sensitivity=settings.sensitivity, target=settings.target,
                invert=settings.invert, expansion_px=settings.expansion_px,
                feather_px=settings.feather_px, source_shape=source_shape)
        else:
            master = derive_preview_mask(
                mask_context.probability,
                sensitivity=settings.sensitivity, target=settings.target,
                invert=settings.invert, expansion_px=settings.expansion_px,
                feather_px=settings.feather_px,
                source_shape=source_shape, target_shape=derive_shape)
        _cancel(is_cancelled)
    return master, mask_identity, derived_new


def derive_render_mask(mask_context, target_shape, *, caches=None, is_cancelled=None) -> np.ndarray:
    """Derive the mask a render/overlay needs at ``target_shape``, cached.

    Capped previews (``target_shape`` smaller than the source) derive at the
    preview resolution and share the derived cache with :func:`render_with_mask`,
    so a mask-overlay preview never re-derives the master at source resolution.
    """
    source_shape = mask_context.source_rgba.shape[:2]
    target_shape = tuple(target_shape)
    derive_shape = source_shape if target_shape == source_shape else target_shape
    master, mask_identity, derived_new = _derive_cached_master(
        mask_context, derive_shape, source_shape, caches=caches, is_cancelled=is_cancelled)
    if derived_new and caches is not None:
        caches.put_derived(mask_identity, master, derive_shape)
    return master if master.shape == target_shape else resize_mask_area(master, target_shape)


def _render_baked(renderer, settings, master, mask_identity, mask_context,
                  caches, rendered_identity, is_cancelled, target_shape,
                  derived_new, derive_shape) -> np.ndarray:
    """Bake the outside fill into the renderer's base and skip compositing.

    The renderer receives one ``bake(base) -> baked_base`` callable it must
    apply to its pipeline input BEFORE any proxy downscale, so the fill is
    dithered as part of the image instead of stamped over the result.
    """
    fill = 255.0 if settings.outside is OutsideMode.WHITE else 0.0
    composite_identity = None
    if caches is not None and rendered_identity is not None and target_shape is not None:
        composite_identity = CompositeIdentity(
            (rendered_identity, tuple(target_shape)), mask_identity, settings.outside,
            mask_context.source, ALPHA_ALGORITHM_VERSION, baked=True)
        cached = caches.get_composite(composite_identity)
        if cached is not None:
            _cancel(is_cancelled)
            if derived_new:
                caches.put_derived(mask_identity, master, derive_shape)
            return cached

    def bake(base: np.ndarray) -> np.ndarray:
        base = np.asarray(base, dtype=np.float32)
        mask = master if master.shape == base.shape else resize_mask_area(master, base.shape)
        return bake_outside_base(base, mask, fill)

    rendered = renderer(bake)
    _cancel(is_cancelled)
    if caches is not None:
        if derived_new:
            caches.put_derived(mask_identity, master, derive_shape)
        if composite_identity is not None:
            caches.put_composite(composite_identity, rendered)
    return rendered


def render_with_mask(renderer: Callable[..., np.ndarray], mask_context=None, *,
                     caches: MaskCaches | None = None, rendered_identity=None,
                     is_cancelled=None, target_shape=None) -> np.ndarray:
    """Render one complete branch and optionally outer-composite its mask.

    ``mask_context is None`` is the explicit historical bypass: no source hash,
    mask derivation, resizing, compositing, or mask allocation occurs.

    When the settings ask for a baked White/Black fill, ``renderer`` is called
    with one positional ``bake`` callable to apply to its base image and the
    outer composite is skipped; otherwise ``renderer`` is called with no
    arguments exactly as before.
    """
    if mask_context is None:
        return renderer()

    _cancel(is_cancelled)
    settings = mask_context.settings
    source = mask_context.source_rgba
    source_shape = source.shape[:2]
    resolved_target = tuple(target_shape) if target_shape is not None else None
    baked = bake_fill_active(settings)
    # A capped preview (target smaller than the source) derives the master at the
    # preview shape -- including the baked path, whose fill is now dithered into
    # the proxy-downscaled base at that same shape (preview.py bakes after the
    # downscale). An unknown target (None) or a source-equal target stays
    # full-res, matching exports and the historical path.
    preview = resolved_target is not None and resolved_target != source_shape
    derive_shape = resolved_target if preview else source_shape
    master, mask_identity, derived_new = _derive_cached_master(
        mask_context, derive_shape, source_shape, caches=caches, is_cancelled=is_cancelled)
    _cancel(is_cancelled)
    if baked:
        return _render_baked(renderer, settings, master, mask_identity, mask_context,
                             caches, rendered_identity, is_cancelled, resolved_target,
                             derived_new, derive_shape)
    rendered = None
    if resolved_target is None:
        rendered = renderer()
        _cancel(is_cancelled)
        resolved_target = rendered.shape[:2]
    resolved_target = tuple(resolved_target)
    mask = master if master.shape == resolved_target else resize_mask_area(master, resolved_target)
    _cancel(is_cancelled)
    target_source = _resize_source_rgba(source, resolved_target)
    _cancel(is_cancelled)
    composite_identity = None
    if caches is not None and rendered_identity is not None:
        composite_identity = CompositeIdentity(
            (rendered_identity, resolved_target), mask_identity, settings.outside,
            mask_context.source, ALPHA_ALGORITHM_VERSION)
        cached = caches.get_composite(composite_identity)
        if cached is not None:
            _cancel(is_cancelled)
            if derived_new:
                caches.put_derived(mask_identity, master, derive_shape)
            return cached
    if rendered is None:
        rendered = renderer()
        _cancel(is_cancelled)
    result = composite_masked(rendered, target_source, mask, settings.outside)
    _cancel(is_cancelled)
    if caches is not None:
        if derived_new:
            caches.put_derived(mask_identity, master, derive_shape)
        if composite_identity is not None:
            caches.put_composite(composite_identity, result)
    return result
