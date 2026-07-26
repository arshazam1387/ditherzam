from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from ..preview import preview_target_size
from ..render import RenderCancelled
from ..composition import LookRenderer
from .blend import blend_layer
from .mask_contracts import LookInputsKey, RenderedLookKey
from .model import LayerDocument, LayerStack
from .render_cache import LayerLookRenderCache


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
    # MT-10 stack-order data. ``prefix_rgba`` is the accepted composite below
    # the selected layer; ``layer_before_raster_mask_rgba`` is normalized Look
    # output; ``above_layers`` are replayed in original order.
    prefix_rgba: np.ndarray | None = None
    layer_before_raster_mask_rgba: np.ndarray | None = None
    above_layers: tuple = ()
    mask_source_x: np.ndarray | None = None
    mask_source_y: np.ndarray | None = None

    @property
    def supports_mask_stroke(self) -> bool:
        return (
            self.prefix_rgba is not None
            and self.layer_before_raster_mask_rgba is not None)

    @property
    def mask_stroke_retained_bytes(self) -> int:
        arrays = [self.prefix_rgba, self.layer_before_raster_mask_rgba]
        arrays.extend(item[0] for item in self.above_layers)
        arrays.extend((self.mask_source_x, self.mask_source_y))
        return sum(
            int(array.nbytes) for array in arrays
            if isinstance(array, np.ndarray))


@dataclass(frozen=True)
class LayerDocumentRender:
    composite_rgba: np.ndarray
    proxy: LayerRenderProxy


def composite_mask_stroke_roi(
    proxy: LayerRenderProxy,
    mask_pixels: np.ndarray,
    density: int,
    dirty_rect,
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """Replay one changed mask ROI in exact original stack/blend order."""
    if not isinstance(proxy, LayerRenderProxy) or not proxy.supports_mask_stroke:
        raise ValueError("a mask-stroke-capable proxy is required")
    if (
        not isinstance(mask_pixels, np.ndarray)
        or mask_pixels.dtype != np.uint8 or mask_pixels.ndim != 2
    ):
        raise ValueError("mask_pixels must be a 2D uint8 array")
    if isinstance(density, bool) or not isinstance(density, int) or not 0 <= density <= 100:
        raise ValueError("density must be within 0..100")
    selected = proxy.layer_before_raster_mask_rgba
    prefix = proxy.prefix_rgba
    assert selected is not None and prefix is not None
    sh, sw = mask_pixels.shape
    th, tw = selected.shape[:2]
    map_x = proxy.mask_source_x
    map_y = proxy.mask_source_y
    if (
        map_x is None or map_y is None
        or map_x.shape != (tw,) or map_y.shape != (th,)
        or int(map_x.min(initial=0)) < 0 or int(map_x.max(initial=0)) >= sw
        or int(map_y.min(initial=0)) < 0 or int(map_y.max(initial=0)) >= sh
    ):
        raise ValueError("proxy source-index maps do not match the mask geometry")
    # Conservative target coverage around nearest-neighbour sample boundaries.
    tx0 = max(0, int(np.floor(dirty_rect.x0 * tw / sw)) - 1)
    ty0 = max(0, int(np.floor(dirty_rect.y0 * th / sh)) - 1)
    tx1 = min(tw, int(np.ceil(dirty_rect.x1 * tw / sw)) + 1)
    ty1 = min(th, int(np.ceil(dirty_rect.y1 * th / sh)) + 1)
    dx0 = max(0, proxy.x + tx0)
    dy0 = max(0, proxy.y + ty0)
    dx1 = min(prefix.shape[1], proxy.x + tx1)
    dy1 = min(prefix.shape[0], proxy.y + ty1)
    if dx0 >= dx1 or dy0 >= dy1:
        return (dx0, dy0, dx0, dy0), np.empty((0, 0, 4), np.uint8)
    sx0, sy0 = dx0 - proxy.x, dy0 - proxy.y
    sx1, sy1 = dx1 - proxy.x, dy1 - proxy.y
    layer_roi = np.array(selected[sy0:sy1, sx0:sx1], copy=True)
    coverage = mask_pixels[
        np.ix_(map_y[sy0:sy1], map_x[sx0:sx1])
    ].astype(np.uint16)
    np.subtract(255, coverage, out=coverage)
    np.multiply(coverage, density, out=coverage)
    np.add(coverage, 50, out=coverage)
    np.floor_divide(coverage, 100, out=coverage)
    np.subtract(255, coverage, out=coverage)
    np.multiply(coverage, layer_roi[..., 3], out=coverage)
    np.add(coverage, 127, out=coverage)
    np.floor_divide(coverage, 255, out=coverage)
    layer_roi[..., 3] = coverage
    canvas = np.array(prefix[dy0:dy1, dx0:dx1], copy=True)
    if proxy.visible and proxy.opacity:
        if (
            proxy.blend_mode == "normal" and proxy.opacity == 100
            and not np.any(canvas[..., 3])
        ):
            canvas = layer_roi
        else:
            canvas = blend_layer(
                canvas, layer_roi, proxy.blend_mode, proxy.opacity)
    for rendered, x, y, mode, opacity in proxy.above_layers:
        left, top = max(dx0, x), max(dy0, y)
        right = min(dx1, x + rendered.shape[1])
        bottom = min(dy1, y + rendered.shape[0])
        if left >= right or top >= bottom:
            continue
        cy0, cx0 = top - dy0, left - dx0
        cy1, cx1 = bottom - dy0, right - dx0
        source = rendered[top - y:bottom - y, left - x:right - x]
        canvas[cy0:cy1, cx0:cx1] = blend_layer(
            canvas[cy0:cy1, cx0:cx1], source, mode, opacity)
    return (dx0, dy0, dx1, dy1), canvas


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


def _resize_channel_nearest(
    channel: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    if channel.shape == shape:
        return channel
    height, width = shape
    return np.asarray(
        Image.fromarray(channel, mode="L").resize(
            (width, height), resample=Image.Resampling.NEAREST),
        dtype=np.uint8,
    )


def _nearest_source_index_map(source_size: int, target_size: int) -> np.ndarray:
    """Freeze Pillow NEAREST's exact target-to-source index phase."""
    if source_size <= 0 or target_size <= 0:
        raise ValueError("map dimensions must be positive")
    ramp = np.arange(source_size, dtype=np.int32).reshape(1, source_size)
    mapped = np.asarray(
        Image.fromarray(ramp, mode="I").resize(
            (target_size, 1), resample=Image.Resampling.NEAREST),
        dtype=np.int32,
    )[0].copy()
    mapped.flags.writeable = False
    return mapped


def _normalize_layer_alpha(
    rendered, source_rgba, target_shape: tuple[int, int]
) -> np.ndarray:
    """Normalize a completed Look to target-sized straight RGBA.

    Smart Mask RGBA already contains its resolved Original/Transparent alpha,
    so it is preserved. Historical RGB Looks receive source alpha sampled
    directly from source coordinates to the final layer target.
    """
    rendered = np.asarray(rendered)
    if rendered.ndim != 3 or rendered.shape[2] not in (3, 4):
        raise ValueError("renderer output must be RGB or RGBA")
    rendered = _resize_rgba(rendered, target_shape)
    if rendered.shape[2] == 4:
        return rendered
    alpha = _resize_channel_nearest(source_rgba[..., 3], target_shape)
    result = np.empty((*target_shape, 4), dtype=np.uint8)
    result[..., :3] = rendered
    result[..., 3] = alpha
    return result


def _resize_rgba(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == shape:
        return image
    height, width = shape
    mode = "RGBA" if image.shape[2] == 4 else "RGB"
    return np.asarray(
        Image.fromarray(image, mode=mode).resize(
            (width, height), resample=Image.Resampling.NEAREST),
        dtype=np.uint8,
    )


def _apply_raster_mask(
    rendered, raster_mask, target_shape, *, is_cancelled=None
) -> np.ndarray:
    """Apply post-Look raster coverage with exact half-up byte arithmetic."""
    if (
        raster_mask is None
        or not raster_mask.enabled
        or raster_mask.density == 0
    ):
        return rendered
    _cancel(is_cancelled)
    minimum = int(raster_mask.pixels.min())
    if minimum == 255:
        return rendered
    _cancel(is_cancelled)

    result = np.array(rendered, dtype=np.uint8, order="C", copy=True)
    if raster_mask.density == 100 and int(raster_mask.pixels.max()) == 0:
        result[..., 3] = 0
        _cancel(is_cancelled)
        return result
    _cancel(is_cancelled)

    mask = _resize_channel_nearest(raster_mask.pixels, target_shape)
    _cancel(is_cancelled)
    # One reusable uint16 workspace is sufficient: the largest intermediate is
    # 255*255 + 127 = 65152. Keep the RGBA result as the only full-frame copy.
    workspace = mask.astype(np.uint16)
    np.subtract(255, workspace, out=workspace)
    np.multiply(workspace, raster_mask.density, out=workspace)
    np.add(workspace, 50, out=workspace)
    np.floor_divide(workspace, 100, out=workspace)
    np.subtract(255, workspace, out=workspace)
    np.multiply(workspace, result[..., 3], out=workspace)
    np.add(workspace, 127, out=workspace)
    np.floor_divide(workspace, 255, out=workspace)
    result[..., 3] = workspace
    _cancel(is_cancelled)
    return result


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
    look_cache: LayerLookRenderCache | None = None,
    mask_caches=None,
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
    proxy_pre_mask = None
    proxy_prefix = None
    proxy_above = []
    proxy_seen = False
    proxy_metadata = None
    proxy_maps = None
    for layer in document.layers:
        _cancel(is_cancelled)
        is_proxy_layer = layer.id == proxy_layer_id
        if (not layer.visible or layer.opacity == 0) and not is_proxy_layer:
            continue
        source = layer.source
        source_h, source_w = source.gray.shape
        target_shape = (
            max(1, int(round(
                source_h * layer.transform.scale_y * scale_y))),
            max(1, int(round(
                source_w * layer.transform.scale_x * scale_x))),
        )
        layer_cap = None if target_max_side is None else max(target_shape)
        effective_smart = (
            replace(layer.look.smart_mask, enabled=False)
            if (
                allow_pending_masks
                and source.probability is None
                and layer.look.smart_mask.enabled
            )
            else layer.look.smart_mask
        )
        look_target_shape = (
            (source_h, source_w)
            if layer_cap is None
            else preview_target_size(source_h, source_w, layer_cap)
        )
        look_key = RenderedLookKey(
            source.source_identity,
            layer.look.signature,
            look_target_shape,
            LookInputsKey(
                effective_smart,
                None if source.probability is None
                else source.probability.identity,
            ),
        )

        def render_look():
            cache_args = {} if mask_caches is None else {"caches": mask_caches}
            renderer = LookRenderer(
                registry,
                source.gray,
                source.rgba,
                probability=source.probability,
                **cache_args,
            )
            return renderer.render(
                layer.look,
                smart_mask=effective_smart,
                target_max_side=layer_cap,
                is_cancelled=is_cancelled,
            )

        rendered = (
            render_look()
            if look_cache is None
            else look_cache.get_or_render(
                look_key, render_look, is_cancelled=is_cancelled)
        )
        rendered = _normalize_layer_alpha(rendered, source.rgba, target_shape)
        normalized = rendered
        rendered = _apply_raster_mask(
            rendered, layer.raster_mask, target_shape,
            is_cancelled=is_cancelled)
        _cancel(is_cancelled)
        x = int(round(layer.transform.x * scale_x))
        y = int(round(layer.transform.y * scale_y))
        if is_proxy_layer:
            proxy_prefix = np.array(canvas, copy=True, order="C")
            proxy_pre_mask = np.array(normalized, copy=True, order="C")
            proxy_seen = True
            proxy_rendered = rendered
            proxy_metadata = (
                x, y, layer.blend_mode, layer.opacity, layer.visible)
            proxy_maps = (
                _nearest_source_index_map(source_w, target_shape[1]),
                _nearest_source_index_map(source_h, target_shape[0]),
            )
        if layer.visible and layer.opacity:
            canvas = _place_layer(
                canvas, rendered, x, y, layer.blend_mode, layer.opacity)
        if proxy_seen and not is_proxy_layer and layer.visible and layer.opacity:
            replay = np.array(rendered, copy=True, order="C")
            replay.flags.writeable = False
            proxy_above.append(
                (replay, x, y, layer.blend_mode, layer.opacity))
        if proxy_canvas is not None and not is_proxy_layer:
            proxy_canvas = _place_layer(
                proxy_canvas, rendered, x, y, layer.blend_mode, layer.opacity)
        _cancel(is_cancelled)
    _cancel(is_cancelled)
    proxy = None
    if proxy_layer_id is not None:
        if proxy_rendered is None or proxy_metadata is None or proxy_maps is None:
            raise ValueError("layer_id must identify a live document layer")
        x, y, blend_mode, opacity, visible = proxy_metadata
        background = np.array(proxy_canvas, copy=True, order="C")
        foreground = np.array(proxy_rendered, copy=True, order="C")
        prefix = np.array(proxy_prefix, copy=True, order="C")
        pre_mask = np.array(proxy_pre_mask, copy=True, order="C")
        background.flags.writeable = False
        foreground.flags.writeable = False
        prefix.flags.writeable = False
        pre_mask.flags.writeable = False
        proxy = LayerRenderProxy(
            proxy_layer_id,
            background,
            foreground,
            x,
            y,
            blend_mode,
            opacity,
            visible,
            prefix,
            pre_mask,
            tuple(proxy_above),
            proxy_maps[0],
            proxy_maps[1],
        )
    return canvas, proxy


def render_layer_document(
    document: LayerDocument,
    registry,
    *,
    target_max_side=None,
    is_cancelled=None,
    allow_pending_masks: bool = False,
    look_cache: LayerLookRenderCache | None = None,
    mask_caches=None,
) -> np.ndarray:
    composite, _proxy = _render_layer_document(
        document,
        registry,
        target_max_side=target_max_side,
        is_cancelled=is_cancelled,
        allow_pending_masks=allow_pending_masks,
        look_cache=look_cache,
        mask_caches=mask_caches,
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
    look_cache: LayerLookRenderCache | None = None,
    mask_caches=None,
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
        look_cache=look_cache,
        mask_caches=mask_caches,
    )
    assert proxy is not None
    return LayerDocumentRender(composite, proxy)
