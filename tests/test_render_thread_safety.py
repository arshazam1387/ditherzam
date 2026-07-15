"""render()/render_cached() must snapshot color_engine/effect_stack exactly once.

Root cause of the "app gets stuck the more you use it" throw: a TOCTOU race on
the shared pipeline. The GUI thread's ``ImageEditor._sync_pipeline()`` reassigns
``pipeline.color_engine`` / ``pipeline.effect_stack`` while a background render
worker is mid-render. render()/render_cached() read each attribute *several*
times per call (``is not None`` check, then dereference), so a reassignment to
``None`` between two reads raises ``AttributeError`` on the render thread
(``'NoneType' object has no attribute 'map'`` / ``'apply'``). Pre-fix that
exception wedged the render coalescer permanently.

Reading each attribute once into a local makes the read atomic under the GIL, so
a concurrent reassignment can never split a single render. These tests pin that
invariant deterministically (no timing) by counting attribute reads.
"""
import numpy as np

from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.color.palette import Palette
from ditherzam.color.engine import ColorEngine
from ditherzam.effects.stack import EffectStack

BASE = np.tile(np.linspace(0, 255, 24, np.float32), (24, 1))
SETTINGS = RenderSettings(style="None", scale=1)


def _duo_engine():
    return ColorEngine(Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]]), "nearest")


def _sharpen_stack():
    s = EffectStack()
    s.add("Sharpen", amount=1.0)
    return s


class _CountingPipeline(RenderPipeline):
    """Counts reads of ``color_engine`` / ``effect_stack`` and, after the first
    read of the watched attribute, returns ``None`` -- exactly what a concurrent
    ``_sync_pipeline()`` reassignment looks like to a mid-flight render. A
    race-free render reads the watched attribute once, so it never sees the
    ``None`` and never raises."""

    def __init__(self, watch, *, color_engine=None, effect_stack=None):
        self._watch = watch
        self._reads = {"color_engine": 0, "effect_stack": 0}
        self._store = {"color_engine": None, "effect_stack": None}
        # Let the base __init__ populate _store via the setters below.
        super().__init__(registry, color_engine=color_engine, effect_stack=effect_stack)

    def _get(self, name):
        self._reads[name] += 1
        if name == self._watch and self._reads[name] > 1:
            return None
        return self._store[name]

    @property
    def color_engine(self):
        return self._get("color_engine")

    @color_engine.setter
    def color_engine(self, v):
        self._store["color_engine"] = v

    @property
    def effect_stack(self):
        return self._get("effect_stack")

    @effect_stack.setter
    def effect_stack(self, v):
        self._store["effect_stack"] = v


def test_render_reads_color_engine_once():
    p = _CountingPipeline("color_engine", color_engine=_duo_engine())
    out = p.render(BASE, SETTINGS)  # must not raise on the flip-to-None
    assert out.shape == (24, 24, 3)
    assert p._reads["color_engine"] == 1


def test_render_reads_effect_stack_once():
    p = _CountingPipeline("effect_stack", effect_stack=_sharpen_stack())
    out = p.render(BASE, SETTINGS)
    assert out.shape == (24, 24, 3)
    assert p._reads["effect_stack"] == 1


def test_render_cached_reads_color_engine_once():
    p = _CountingPipeline("color_engine", color_engine=_duo_engine())
    out = p.render_cached(BASE, SETTINGS)
    assert out.shape == (24, 24, 3)
    assert p._reads["color_engine"] == 1


def test_render_cached_reads_effect_stack_once():
    p = _CountingPipeline("effect_stack", effect_stack=_sharpen_stack())
    out = p.render_cached(BASE, SETTINGS)
    assert out.shape == (24, 24, 3)
    assert p._reads["effect_stack"] == 1
