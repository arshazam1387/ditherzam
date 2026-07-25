from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from ..preview import preview_target_size
from ..render import RenderCancelled
from ..composition import LookRenderer
from .blend import blend_layer
from .model import LayerDocument, LayerStack


@dataclass(frozen=True)
class LayerRenderProxy:
    """Read-only pieces for moving one already-rendered layer interactively."""

    layer_id: str
    background_rgba: np.ndarray
    layer_rgba: np.ndarray
    x: int
    y: int
    blend_mode: str
    opacity: int
    visible: bool


@dataclass(frozen=True)
class LayerDocumentRender:
    composite_rgba: np.ndarray
    proxy: LayerRenderProxy


def _cancel(is_cancelled) -> None:
    if is_cancelled is not None and is_cancelled():
        raise RenderCancelled


def _transparent_canvas(renderer, target_max_side) -> np.ndarray:
    source = getattr(renderer, "source_rgba", None)
    if not isinstance(source, np.ndarray) or source.ndim != 3:
        raise ValueError("renderer must expose source_rgba for an empty layer stack")
    height, width = source.shape[:2]
    if target_max_side is not None:
        height, width = preview_target_size(height, width, target_max_side)
    return np.zeros((height, width, 4), dtype=np.uint8)


def _with_source_alpha(rendered, source_rgba) -> np.ndarray:
    rendered = np.asarray(rendered)
    if rendered.ndim != 3 or rendered.shape[2] not in (3, 4):
        raise ValueError("renderer output must be RGB or RGBA")
    if rendered.shape[2] == 4:
        return rendered
    alpha = source_rgba[..., 3]
    if alpha.shape != rendered.shape[:2]:
        height, width = rendered.shape[:2]
        alpha = np.asarray(
            Image.fromarray(alpha, mode="L").resize(
                (width, height), resample=Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    result = np.empty((*rendered.shape[:2], 4), dtype=np.uint8)
    result[..., :3] = rendered
    result[..., 3] = alpha
    return result


def _resize_rgba(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == shape:
        return image
    height, width = shape
    return np.asarray(
        Image.fromarray(image, mode="RGBA").resize(
            (width, height), resample=Image.Resampling.NEAREST),
        dtype=np.uint8,
    )


def _place_layer(
    canvas: np.ndarray, rendered: np.ndarray, x: int, y: int, mode: str, opacity: int
) -> np.ndarray:
    canvas_h, canvas_w = canvas.shape[:2]
    layer_h, layer_w = rendered.shape[:2]
    left = max(0, x)
    top = max(0, y)
    right = min(canvas_w, x + layer_w)
    bottom = min(canvas_h, y + layer_h)
    if left >= right or top >= bottom:
        return canvas
    source = rendered[top - y:bottom - y, left - x:right - x]
    result = canvas.copy()
    result[top:bottom, left:right] = blend_layer(
        canvas[top:bottom, left:right], source, mode, opacity)
    return result


def render_layer_stack(
    stack: LayerStack,
    renderer,
    *,
    target_max_side=None,
    is_cancelled=None,
) -> np.ndarray:
    if not isinstance(stack, LayerStack):
        raise ValueError("stack must be a LayerStack")
    _cancel(is_cancelled)
    canvas = None
    for layer in stack.layers:
        _cancel(is_cancelled)
        if not layer.visible or layer.opacity == 0:
            continue
        rendered = renderer.render(
            layer.look,
            target_max_side=target_max_side,
            is_cancelled=is_cancelled,
        )
        if canvas is None:
            rgba_shape = np.asarray(rendered).shape
            if len(rgba_shape) != 3 or rgba_shape[2] not in (3, 4):
                raise ValueError("renderer output must be RGB or RGBA")
            canvas = np.zeros((*rgba_shape[:2], 4), dtype=np.uint8)
        canvas = blend_layer(canvas, rendered, layer.blend_mode, layer.opacity)
        _cancel(is_cancelled)
    if canvas is None:
        canvas = _transparent_canvas(renderer, target_max_side)
    _cancel(is_cancelled)
    return canvas


def _render_layer_document(
    document: LayerDocument,
    registry,
    *,
    target_max_side=None,
    is_cancelled=None,
    allow_pending_masks: bool = False,
    proxy_layer_id: str | None = None,
) -> tuple[np.ndarray, LayerRenderProxy | None]:
    if not isinstance(document, LayerDocument):
        raise ValueError("document must be a LayerDocument")
    if target_max_side is not None and (
        isinstance(target_max_side, bool)
        or not isinstance(target_max_side, int)
        or target_max_side <= 0
    ):
        raise ValueError("target_max_side must be a positive non-bool integer")
    _cancel(is_cancelled)
    source_canvas_h = document.canvas.height
    source_canvas_w = document.canvas.width
    height, width = source_canvas_h, source_canvas_w
    if target_max_side is not None:
        height, width = preview_target_size(height, width, target_max_side)
    scale_y = height / float(source_canvas_h)
    scale_x = width / float(source_canvas_w)
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    proxy_canvas = (
        np.zeros_like(canvas) if proxy_layer_id is not None else None)
    proxy_rendered = None
    proxy_metadata = None
    for layer in document.layers:
        _cancel(is_cancelled)
        is_proxy_layer = layer.id == proxy_layer_id
        if (not layer.visible or layer.opacity == 0) and not is_proxy_layer:
            continue
        source = layer.source
        renderer = LookRenderer(
            registry,
            source.gray,
            source.rgba,
            probability=source.probability,
        )
        source_h, source_w = source.gray.shape
        target_shape = (
            max(1, int(round(
                source_h * layer.transform.scale_y * scale_y))),
            max(1, int(round(
                source_w * layer.transform.scale_x * scale_x))),
        )
        layer_cap = None if target_max_side is None else max(target_shape)
        rendered = renderer.render(
            layer.look,
            smart_mask=(
                replace(layer.look.smart_mask, enabled=False)
                if (
                    allow_pending_masks
                    and source.probability is None
                    and layer.look.smart_mask.enabled
                )
                else layer.look.smart_mask
            ),
            target_max_side=layer_cap,
            is_cancelled=is_cancelled,
        )
        rendered = _with_source_alpha(rendered, source.rgba)
        rendered = _resize_rgba(rendered, target_shape)
        _cancel(is_cancelled)
        x = int(round(layer.transform.x * scale_x))
        y = int(round(layer.transform.y * scale_y))
        if is_proxy_layer:
            proxy_rendered = rendered
            proxy_metadata = (
                x, y, layer.blend_mode, layer.opacity, layer.visible)
        if layer.visible and layer.opacity:
            canvas = _place_layer(
                canvas, rendered, x, y, layer.blend_mode, layer.opacity)
        if proxy_canvas is not None and not is_proxy_layer:
            proxy_canvas = _place_layer(
                proxy_canvas, rendered, x, y, layer.blend_mode, layer.opacity)
        _cancel(is_cancelled)
    _cancel(is_cancelled)
    proxy = None
    if proxy_layer_id is not None:
        if proxy_rendered is None or proxy_metadata is None:
            raise ValueError("layer_id must identify a live document layer")
        x, y, blend_mode, opacity, visible = proxy_metadata
        background = np.array(proxy_canvas, copy=True, order="C")
        foreground = np.array(proxy_rendered, copy=True, order="C")
        background.flags.writeable = False
        foreground.flags.writeable = False
        proxy = LayerRenderProxy(
            proxy_layer_id,
            background,
            foreground,
            x,
            y,
            blend_mode,
            opacity,
            visible,
        )
    return canvas, proxy


def render_layer_document(
    document: LayerDocument,
    registry,
    *,
    target_max_side=None,
    is_cancelled=None,
    allow_pending_masks: bool = False,
) -> np.ndarray:
    composite, _proxy = _render_layer_document(
        document,
        registry,
        target_max_side=target_max_side,
        is_cancelled=is_cancelled,
        allow_pending_masks=allow_pending_masks,
    )
    return composite


def render_layer_document_with_proxy(
    document: LayerDocument,
    registry,
    *,
    layer_id: str,
    target_max_side=None,
    is_cancelled=None,
    allow_pending_masks: bool = False,
) -> LayerDocumentRender:
    """Render once and capture reusable visual-proxy parts for ``layer_id``.

    Other layers are composited into a background that excludes the selected
    layer entirely. The selected layer RGBA is captured after its Look and
    document scaling, but before placement and blending.
    """
    if not isinstance(layer_id, str) or not layer_id:
        raise ValueError("layer_id must identify a live document layer")
    composite, proxy = _render_layer_document(
        document,
        registry,
        target_max_side=target_max_side,
        is_cancelled=is_cancelled,
        allow_pending_masks=allow_pending_masks,
        proxy_layer_id=layer_id,
    )
    assert proxy is not None
    return LayerDocumentRender(composite, proxy)
