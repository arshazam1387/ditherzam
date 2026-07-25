from __future__ import annotations

from dataclasses import replace

import numpy as np

from ditherzam.composition.compositor import Compositor, render_frames
from ditherzam.composition.look import Look
from ditherzam.composition.model import Composition, LookClip, TransitionSpec
from ditherzam.composition.rendering import LookRenderer
from ditherzam.dithering.registry import DitherRegistry


def _preset(*, contrast=50, sensitivity=50, style="None", color=None):
    preset = {
        "adjustments": {
            "contrast": contrast,
            "midtones": 50,
            "highlights": 50,
            "blur": 0,
            "luminance_threshold": 50,
            "saturation": 50,
            "invert": False,
        },
        "dither": {
            "style": style,
            "scale": 1,
            "depth": 2,
            "color_mapping": "match",
            "preview_disabled": False,
            "params": {},
        },
        "smart_mask": {
            "enabled": False,
            "target": "subject",
            "sensitivity": sensitivity,
            "feather_px": 0,
            "expansion_px": 0,
            "invert": False,
            "outside": "original",
            "bake_fill": False,
        },
    }
    if color is not None:
        preset["color"] = color
    return preset


def _renderer():
    gray = np.array([[0, 64], [128, 255]], dtype=np.float32)
    rgba = np.empty((2, 2, 4), dtype=np.uint8)
    rgba[..., :3] = gray[..., None]
    rgba[..., 3] = 255
    return LookRenderer(DitherRegistry(), gray, rgba)


def test_single_clip_hold_matches_look_renderer_exactly():
    renderer = _renderer()
    look = Look("hold", _preset(contrast=63))
    composition = Composition(2)
    composition.add_clip(LookClip(look, 0, 2))
    compositor = Compositor(composition, renderer)
    assert np.array_equal(compositor.render_frame(0), renderer.render(look))


def test_compatible_param_morph_renders_once_through_b_context(monkeypatch):
    renderer = _renderer()
    look_a = Look("a", _preset(contrast=20, sensitivity=20))
    look_b = Look("b", _preset(contrast=80, sensitivity=80))
    composition = Composition(4)
    composition.add_clip(LookClip(look_a, 0, 2))
    composition.add_clip(LookClip(look_b, 2, 4))
    composition.set_transition(1, TransitionSpec("param-morph", 2, {}))
    calls = []
    real_render = renderer.render

    def counted(look, **kwargs):
        calls.append((look, kwargs))
        return real_render(look, **kwargs)

    monkeypatch.setattr(renderer, "render", counted)
    Compositor(composition, renderer).render_frame(3)
    assert len(calls) == 1
    assert calls[0][0] is look_b
    assert calls[0][1]["settings"].contrast == 50
    assert calls[0][1]["smart_mask"].sensitivity == 50


def test_incompatible_param_morph_falls_back_to_completed_frame_crossfade(monkeypatch):
    renderer = _renderer()
    look_a = Look("a", _preset(style="None"))
    look_b = Look("b", _preset(style="Floyd-Steinberg"))
    composition = Composition(4)
    composition.add_clip(LookClip(look_a, 0, 2))
    composition.add_clip(LookClip(look_b, 2, 4))
    composition.set_transition(1, TransitionSpec("param-morph", 2, {}))
    calls = []
    real_render = renderer.render

    def counted(look, **kwargs):
        calls.append(look)
        return real_render(look, **kwargs)

    monkeypatch.setattr(renderer, "render", counted)
    Compositor(composition, renderer).render_frame(3)
    assert calls == [look_a, look_b]


def test_param_morph_endpoints_return_completed_a_and_b():
    renderer = _renderer()
    a = Look("a", _preset(contrast=20))
    b = Look("b", _preset(contrast=80))
    composition = Composition(4)
    composition.add_clip(LookClip(a, 0, 2))
    composition.add_clip(LookClip(b, 2, 4))
    composition.set_transition(1, TransitionSpec("param-morph", 2, {}))
    compositor = Compositor(composition, renderer)
    assert np.array_equal(compositor.render_frame(2), renderer.render(a))


def test_render_frames_defaults_to_composition_length_and_allows_zero():
    renderer = _renderer()
    look = Look("hold", _preset())
    composition = Composition(2)
    composition.add_clip(LookClip(look, 0, 2))
    compositor = Compositor(composition, renderer)
    assert len(list(render_frames(compositor))) == 2
    assert list(render_frames(compositor, 0)) == []
