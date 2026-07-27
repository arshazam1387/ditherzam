from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
import os
from typing import Any

import numpy as np


try:
    if os.environ.get("DITHERZAM_DISABLE_NATIVE") == "1":
        raise ImportError("native extensions disabled by environment")
    _composite = importlib.import_module("ditherzam._native._composite")
except ImportError:
    _composite = None


def _as_rgba(value: Any) -> np.ndarray:
    """Return an owned, C-contiguous canonical RGBA image."""
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint8
        or value.ndim != 3
        or value.shape[0] == 0
        or value.shape[1] == 0
        or value.shape[2] not in (3, 4)
    ):
        raise ValueError("image must be a non-empty uint8 (H,W,3|4) array")

    if value.shape[2] == 4:
        return np.array(value, dtype=np.uint8, order="C", copy=True)

    rgba = np.empty((*value.shape[:2], 4), dtype=np.uint8, order="C")
    rgba[..., :3] = value
    rgba[..., 3] = 255
    return rgba


def _canonical_pair(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray]:
    a_rgba = _as_rgba(a)
    b_rgba = _as_rgba(b)
    if a_rgba.shape[:2] != b_rgba.shape[:2]:
        raise ValueError("images must have matching height and width")
    return a_rgba, b_rgba


def _weight_array(weight: Any, shape: tuple[int, int]) -> tuple[float | np.ndarray, bool]:
    if np.isscalar(weight):
        if isinstance(weight, (str, bytes, complex)):
            raise ValueError("weight must be a real scalar or an (H,W) array")
        try:
            scalar = float(weight)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("weight must be a real scalar or an (H,W) array") from exc
        if not np.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
            raise ValueError("weight must be within [0,1]")
        return scalar, True

    weights = np.asarray(weight)
    if weights.shape != shape or not np.issubdtype(weights.dtype, np.number):
        raise ValueError("weight array must have shape (H,W)")
    if np.iscomplexobj(weights):
        raise ValueError("weight must be real")
    weights = weights.astype(np.float64, copy=False)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or np.any(weights > 1.0):
        raise ValueError("weight must be within [0,1]")
    return weights, False


def _blend_straight_rgba_reference(a: Any, b: Any, weight: Any) -> np.ndarray:
    """Independently callable NumPy reference for exactness verification."""
    a_rgba, b_rgba = _canonical_pair(a, b)
    normalized_weight, is_scalar = _weight_array(weight, a_rgba.shape[:2])

    if is_scalar and normalized_weight == 0.0:
        return a_rgba
    if is_scalar and normalized_weight == 1.0:
        return b_rgba

    if is_scalar:
        w = normalized_weight
    else:
        w = normalized_weight[..., None]

    a_float = a_rgba.astype(np.float64)
    b_float = b_rgba.astype(np.float64)
    a_alpha = a_float[..., 3:4]
    b_alpha = b_float[..., 3:4]
    out_alpha = a_alpha * (1.0 - w) + b_alpha * w
    premul = (
        a_float[..., :3] * a_alpha * (1.0 - w)
        + b_float[..., :3] * b_alpha * w
    )
    hidden_rgb = a_float[..., :3] * (1.0 - w) + b_float[..., :3] * w
    out_rgb = np.empty_like(premul)
    np.divide(premul, out_alpha, out=out_rgb, where=out_alpha > 0.0)
    np.copyto(out_rgb, hidden_rgb, where=np.broadcast_to(out_alpha <= 0.0, out_rgb.shape))

    out = np.concatenate((out_rgb, out_alpha), axis=2)
    return np.clip(np.floor(out + 0.5), 0.0, 255.0).astype(np.uint8)


def _blend_straight_rgba_native(a: Any, b: Any, weight: Any) -> np.ndarray:
    """Independently callable native seam; raises instead of falling back."""
    a_rgba, b_rgba = _canonical_pair(a, b)
    normalized_weight, is_scalar = _weight_array(weight, a_rgba.shape[:2])
    if _composite is None:
        raise RuntimeError("native compositor extension is unavailable")
    if is_scalar and normalized_weight == 0.0:
        return a_rgba
    if is_scalar and normalized_weight == 1.0:
        return b_rgba
    a_rgba = np.ascontiguousarray(a_rgba)
    b_rgba = np.ascontiguousarray(b_rgba)
    if is_scalar:
        return _composite.blend_transition_scalar_u8(
            a_rgba, b_rgba, normalized_weight
        )
    return _composite.blend_transition_plane_u8(
        a_rgba, b_rgba, np.ascontiguousarray(normalized_weight)
    )


def blend_straight_rgba(a: Any, b: Any, weight: Any) -> np.ndarray:
    """Blend straight-alpha pixels using premultiplication for the calculation."""
    if _composite is None:
        return _blend_straight_rgba_reference(a, b, weight)
    return _blend_straight_rgba_native(a, b, weight)


def select_rgba(a: Any, b: Any, choose_b: Any) -> np.ndarray:
    """Select complete canonical pixels from A or B using a boolean mask."""
    a_rgba, b_rgba = _canonical_pair(a, b)
    if (
        not isinstance(choose_b, np.ndarray)
        or choose_b.dtype != np.bool_
        or choose_b.shape != a_rgba.shape[:2]
    ):
        raise ValueError("choose_b must be a boolean (H,W) array")
    return np.where(choose_b[..., None], b_rgba, a_rgba)


def _normalized_t(t: Any) -> float:
    if isinstance(t, (str, bytes, complex)) or not np.isscalar(t):
        raise ValueError("t must be a finite scalar")
    try:
        value = float(t)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("t must be a finite scalar") from exc
    if not np.isfinite(value):
        raise ValueError("t must be a finite scalar")
    return float(np.clip(value, 0.0, 1.0))


def _endpoint(t: Any, a: Any, b: Any) -> tuple[float, np.ndarray, np.ndarray, np.ndarray | None]:
    value = _normalized_t(t)
    a_rgba, b_rgba = _canonical_pair(a, b)
    if value == 0.0:
        return value, a_rgba, b_rgba, a_rgba
    if value == 1.0:
        return value, a_rgba, b_rgba, b_rgba
    return value, a_rgba, b_rgba, None


class Transition(ABC):
    @abstractmethod
    def render(
        self,
        t: Any,
        a_rgba: Any,
        b_rgba: Any,
        base_gray: Any,
        params: dict,
        *,
        seed: int = 0,
    ) -> np.ndarray:
        raise NotImplementedError


class CrossfadeTransition(Transition):
    def render(self, t, a_rgba, b_rgba, base_gray, params, *, seed=0):
        value = _normalized_t(t)
        return blend_straight_rgba(a_rgba, b_rgba, value)


def _bayer_matrix(size: int) -> np.ndarray:
    matrix = np.array([[0, 2], [3, 1]], dtype=np.float32)
    while matrix.shape[0] < size:
        matrix = np.block(
            [
                [4.0 * matrix, 4.0 * matrix + 2.0],
                [4.0 * matrix + 3.0, 4.0 * matrix + 1.0],
            ]
        )
    return matrix / float(size * size)


_BAYER_8 = _bayer_matrix(8)


class DitherDissolveTransition(Transition):
    def render(self, t, a_rgba, b_rgba, base_gray, params, *, seed=0):
        value, a, b, endpoint = _endpoint(t, a_rgba, b_rgba)
        if endpoint is not None:
            return endpoint
        params = params or {}
        mask_kind = params.get("mask", "bayer")
        height, width = a.shape[:2]
        if mask_kind == "bayer":
            mask = np.tile(
                _BAYER_8,
                ((height + 7) // 8, (width + 7) // 8),
            )[:height, :width]
        elif mask_kind == "noise":
            mask = np.random.default_rng(seed).random(
                (height, width), dtype=np.float32
            )
        else:
            raise ValueError("mask must be 'bayer' or 'noise'")
        return select_rgba(a, b, mask < value)


class SpatialWipeTransition(Transition):
    def render(self, t, a_rgba, b_rgba, base_gray, params, *, seed=0):
        value, a, b, endpoint = _endpoint(t, a_rgba, b_rgba)
        if endpoint is not None:
            return endpoint
        params = params or {}
        mode = params.get("mode", "linear")
        softness = params.get("softness", 0)
        if isinstance(softness, (str, bytes, complex)) or not np.isscalar(softness):
            raise ValueError("softness must be within [0,1]")
        try:
            softness = float(softness)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("softness must be within [0,1]") from exc
        if not np.isfinite(softness) or not 0.0 <= softness <= 1.0:
            raise ValueError("softness must be within [0,1]")

        height, width = a.shape[:2]
        if mode == "linear":
            direction = params.get("direction", "left")
            if direction not in ("left", "right"):
                raise ValueError("direction must be 'left' or 'right'")
            coord = np.arange(width, dtype=np.float64) / max(width - 1, 1)
            if direction == "right":
                coord = 1.0 - coord
            coord = np.broadcast_to(coord, (height, width))
        elif mode == "radial":
            yy, xx = np.indices((height, width), dtype=np.float64)
            center_y = (height - 1) / 2.0
            center_x = (width - 1) / 2.0
            coord = np.hypot(yy - center_y, xx - center_x)
            max_distance = max(
                np.hypot(y - center_y, x - center_x)
                for y in (0, height - 1)
                for x in (0, width - 1)
            )
            if max_distance > 0.0:
                coord /= max_distance
            else:
                coord.fill(0.0)
        elif mode == "luma":
            gray = np.asarray(base_gray)
            if gray.shape != (height, width) or not np.issubdtype(gray.dtype, np.number):
                raise ValueError("base_gray must be a numeric matching (H,W) array")
            if np.iscomplexobj(gray):
                raise ValueError("base_gray must be real")
            coord = np.clip(gray.astype(np.float64) / 255.0, 0.0, 1.0)
        else:
            raise ValueError("mode must be 'linear', 'radial', or 'luma'")

        if softness == 0.0:
            return select_rgba(a, b, coord <= value)
        weight = np.clip((value - coord) / softness + 0.5, 0.0, 1.0)
        return blend_straight_rgba(a, b, weight)


class ParamMorphTransition(Transition):
    def render(self, t, a_rgba, b_rgba, base_gray, params, *, seed=0):
        value = _normalized_t(t)
        if not hasattr(a_rgba, "frame") or not hasattr(b_rgba, "frame"):
            raise RuntimeError("param-morph requires compositor contexts")
        if value == 0.0:
            return a_rgba.frame()
        if value == 1.0:
            return b_rgba.frame()

        a_settings = a_rgba.settings
        b_settings = b_rgba.settings
        compatible = (
            a_settings.style == b_settings.style
            and a_settings.color_mapping == b_settings.color_mapping
            and a_settings.preview_disabled == b_settings.preview_disabled
            and a_settings.params == b_settings.params
            and a_rgba.normalized_color_effects == b_rgba.normalized_color_effects
            and a_rgba.smart_mask.enabled == b_rgba.smart_mask.enabled
            and a_rgba.smart_mask.target == b_rgba.smart_mask.target
            and a_rgba.smart_mask.invert == b_rgba.smart_mask.invert
            and a_rgba.smart_mask.outside == b_rgba.smart_mask.outside
            and a_rgba.smart_mask.bake_fill == b_rgba.smart_mask.bake_fill
            and a_rgba.source_identity == b_rgba.source_identity
            and a_rgba.inference_identity == b_rgba.inference_identity
        )
        if not compatible:
            return blend_straight_rgba(a_rgba.frame(), b_rgba.frame(), value)

        from dataclasses import replace

        numeric_fields = (
            "contrast",
            "midtones",
            "highlights",
            "blur",
            "luminance_threshold",
            "saturation",
        )
        updates = {
            field: getattr(a_settings, field)
            + (getattr(b_settings, field) - getattr(a_settings, field)) * value
            for field in numeric_fields
        }
        for field in ("scale", "depth"):
            interpolated = (
                getattr(a_settings, field)
                + (getattr(b_settings, field) - getattr(a_settings, field)) * value
            )
            updates[field] = max(1, round(interpolated))
        settings = replace(b_settings, **updates)

        mask_updates = {}
        for field in ("sensitivity", "feather_px", "expansion_px"):
            interpolated = (
                getattr(a_rgba.smart_mask, field)
                + (
                    getattr(b_rgba.smart_mask, field)
                    - getattr(a_rgba.smart_mask, field)
                )
                * value
            )
            mask_updates[field] = round(interpolated)
        smart_mask = replace(b_rgba.smart_mask, **mask_updates)
        return b_rgba.render(settings, smart_mask)


TRANSITIONS = {
    "crossfade": CrossfadeTransition(),
    "dither-dissolve": DitherDissolveTransition(),
    "spatial-wipe": SpatialWipeTransition(),
    "param-morph": ParamMorphTransition(),
}
