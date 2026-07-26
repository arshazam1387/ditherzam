from __future__ import annotations

from dataclasses import dataclass, replace

from ..render import RenderCancelled
from .model import Composition, LookClip
from .rendering import LookRenderer
from .transitions import TRANSITIONS


def _automated_settings(clip: LookClip, idx: int):
    settings = clip.look.settings
    if clip.automation is not None:
        settings = clip.automation.settings_at(settings, idx - clip.start)
    return settings


@dataclass
class _LookContext:
    renderer: LookRenderer
    clip: LookClip
    settings: object
    smart_mask: object
    _frame: object = None

    @property
    def look(self):
        return self.clip.look

    @property
    def source_identity(self):
        return self.renderer.source_identity

    @property
    def inference_identity(self):
        probability = self.renderer.probability
        return None if probability is None else probability.identity

    @property
    def normalized_color_effects(self):
        normalized = dict(self.look.signature)
        return normalized.get("color"), normalized.get("effects")

    def frame(self):
        if self._frame is None:
            self._frame = self.render(self.settings, self.smart_mask)
        return self._frame

    def render(self, settings, smart_mask):
        return self.renderer.render(
            self.look,
            settings=settings,
            smart_mask=smart_mask,
        )


class Compositor:
    def __init__(self, composition, renderer, *, seed=0):
        if not isinstance(composition, Composition):
            raise ValueError("composition must be a Composition")
        if not isinstance(renderer, LookRenderer):
            raise ValueError("renderer must be a LookRenderer")
        composition.validate()
        self.composition = composition
        self.renderer = renderer
        self.seed = seed

    def _context(self, clip: LookClip, idx: int) -> _LookContext:
        return _LookContext(
            self.renderer,
            clip,
            _automated_settings(clip, idx),
            clip.look.smart_mask,
        )

    def render_frame(self, idx):
        resolution = self.composition.resolve(idx)
        if resolution.kind == "hold":
            settings = _automated_settings(resolution.clip, idx)
            return self.renderer.render(resolution.clip.look, settings=settings)

        context_a = self._context(resolution.clip, idx)
        context_b = self._context(resolution.clip_b, idx)
        transition = TRANSITIONS[resolution.spec.kind]
        if resolution.spec.kind == "param-morph":
            return transition.render(
                resolution.t,
                context_a,
                context_b,
                self.renderer.source_gray,
                resolution.spec.params,
                seed=self.seed,
            )
        return transition.render(
            resolution.t,
            context_a.frame(),
            context_b.frame(),
            self.renderer.source_gray,
            resolution.spec.params,
            seed=self.seed,
        )


def render_frames(compositor, length=None, *, is_cancelled=None):
    if length is None:
        if not isinstance(compositor, Compositor):
            raise ValueError("length is required for a non-Compositor")
        length = compositor.composition.length
    if isinstance(length, bool) or not isinstance(length, int):
        raise ValueError("length must be an int")
    maximum = (
        compositor.composition.length
        if isinstance(compositor, Compositor)
        else length
    )
    if length < 0 or length > maximum:
        raise ValueError("length must be within composition length")

    for idx in range(length):
        if is_cancelled is not None and is_cancelled():
            raise RenderCancelled
        frame = compositor.render_frame(idx)
        if is_cancelled is not None and is_cancelled():
            raise RenderCancelled
        yield frame
