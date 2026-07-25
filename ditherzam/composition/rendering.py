from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..masking.cache import MaskCaches
from ..masking.contracts import ProbabilityMap, SourceIdentity, source_identity
from ..masking.render import render_with_mask
from ..masking.settings import MaskTarget, SmartMaskSettings
from ..preview import preview_target_size, render_preview
from ..render import (
    RenderPipeline,
    RenderSettings,
    render_context_signature,
    render_settings_signature,
)
from .look import Look


class CompositionMaskError(RuntimeError):
    pass


@dataclass(frozen=True)
class _MaskContext:
    source: SourceIdentity
    source_rgba: np.ndarray
    probability: ProbabilityMap
    settings: SmartMaskSettings


class LookRenderer:
    def __init__(self, registry, source_gray, source_rgba, *,
                 probability=None, caches=None):
        if not isinstance(source_gray, np.ndarray) or source_gray.dtype != np.float32:
            raise ValueError("source_gray must be an ndarray with dtype float32")
        if not isinstance(source_rgba, np.ndarray) or source_rgba.dtype != np.uint8:
            raise ValueError("source_rgba must be an ndarray with dtype uint8")
        gray = np.array(source_gray, dtype=np.float32, order="C", copy=True)
        rgba = np.array(source_rgba, dtype=np.uint8, order="C", copy=True)
        if gray.ndim != 2 or not gray.size:
            raise ValueError("source_gray must be a non-empty float32 2-D array")
        if rgba.ndim != 3 or rgba.shape[2] != 4 or not rgba.shape[0] or not rgba.shape[1]:
            raise ValueError("source_rgba must be a non-empty uint8 RGBA array")
        if gray.shape != rgba.shape[:2]:
            raise ValueError("source_gray and source_rgba shapes must match")
        gray.flags.writeable = False
        rgba.flags.writeable = False
        identity = source_identity(rgba)
        if probability is not None:
            if not isinstance(probability, ProbabilityMap):
                raise ValueError("probability must be a ProbabilityMap")
            if probability.identity.source != identity:
                raise ValueError("probability source identity does not match source_rgba")
        if caches is not None and not isinstance(caches, MaskCaches):
            raise ValueError("caches must be MaskCaches")
        self.registry = registry
        self.source_gray = gray
        self.source_rgba = rgba
        self.source_identity = identity
        self.probability = probability
        self.caches = caches if caches is not None else MaskCaches()
        self._pipelines = {}

    @property
    def pipeline_count(self) -> int:
        return len(self._pipelines)

    def _pipeline(self, look: Look):
        source_rgb = self.source_rgba[..., :3]
        engine = look.build_color_engine(source_rgb=source_rgb)
        stack = look.build_effect_stack()
        source_sig = self.source_identity if getattr(engine, "mode", None) == "source" else None
        key = (render_context_signature(engine, stack), source_sig)
        pipeline = self._pipelines.get(key)
        if pipeline is None:
            pipeline = RenderPipeline(self.registry, engine, stack)
            self._pipelines[key] = pipeline
        return pipeline, engine, stack

    def render(self, look, *, settings=None, smart_mask=None,
               target_max_side=None, temporal_field=None,
               is_cancelled=None) -> np.ndarray:
        if not isinstance(look, Look):
            raise ValueError("look must be a Look")
        settings = look.settings if settings is None else settings
        smart_mask = look.smart_mask if smart_mask is None else smart_mask
        if type(settings) is not RenderSettings:
            raise ValueError("settings must be RenderSettings")
        if type(smart_mask) is not SmartMaskSettings:
            raise ValueError("smart_mask must be SmartMaskSettings")
        if target_max_side is not None and (
            isinstance(target_max_side, bool)
            or not isinstance(target_max_side, int)
            or target_max_side <= 0
        ):
            raise ValueError("target_max_side must be a positive non-bool int")
        pipeline, engine, stack = self._pipeline(look)
        masked = smart_mask.enabled and smart_mask.target is not MaskTarget.WHOLE_IMAGE
        if masked and self.probability is None:
            raise CompositionMaskError("matching probability is required for masked Look")
        context = None
        if masked:
            context = _MaskContext(
                self.source_identity, self.source_rgba, self.probability, smart_mask)
        target_shape = (
            self.source_gray.shape if target_max_side is None
            else preview_target_size(*self.source_gray.shape, target_max_side)
        )
        rendered_identity = (
            self.source_identity,
            render_settings_signature(settings),
            render_context_signature(engine, stack),
            target_shape,
            "composition-look-v1",
        )
        if target_max_side is not None:
            return render_preview(
                pipeline, self.source_gray, settings, target_max_side,
                is_cancelled=is_cancelled, temporal_field=temporal_field,
                mask_context=context, mask_caches=self.caches,
                rendered_identity=rendered_identity,
            )

        def render_complete(bake=None):
            base = self.source_gray if bake is None else bake(self.source_gray)
            return pipeline.render(
                base, settings, temporal_field=temporal_field,
                is_cancelled=is_cancelled)

        return render_with_mask(
            render_complete, context, caches=self.caches,
            rendered_identity=rendered_identity, is_cancelled=is_cancelled,
            target_shape=target_shape)
