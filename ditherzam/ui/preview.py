"""Compatibility re-export for the Qt-free preview implementation."""

from ditherzam.preview import (
    PREVIEW_RESOLUTIONS,
    auto_preview_resolution,
    normalize_preview_resolution,
    preview_cap,
    preview_target_size,
    proxy_factor,
    proxy_scale,
    render_preview,
    resize_preview_bucket,
    zoom_preview_bucket,
)

__all__ = [
    "PREVIEW_RESOLUTIONS",
    "auto_preview_resolution",
    "normalize_preview_resolution",
    "preview_cap",
    "preview_target_size",
    "proxy_factor",
    "proxy_scale",
    "render_preview",
    "resize_preview_bucket",
    "zoom_preview_bucket",
]
