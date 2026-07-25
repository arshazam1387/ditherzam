from __future__ import annotations

import copy

import numpy as np
import pytest

from ditherzam.masking.settings import MaskTarget, OutsideMode


def _preset(*, enabled=True, target="subject", outside="transparent", bake_fill=False):
    return {
        "adjustments": {
            "contrast": 50, "midtones": 50, "highlights": 50,
            "luminance_threshold": 50, "blur": 0, "saturation": 50,
            "invert": False,
        },
        "dither": {
            "style": "None", "scale": 1, "depth": 2,
            "color_mapping": "match", "preview_disabled": False,
            "params": {"mix": 100},
        },
        "color": {
            "mode": "nearest",
            "source_dither": 37,
            "source_dither_brighten": True,
            "palette": {
                "name": "two", "category": "test",
                "colors": [[0, 0, 0], [255, 255, 255]],
            },
        },
        "effects": [{"name": "Sharpen", "params": {"amount": 0}}],
        "smart_mask": {
            "enabled": enabled,
            "target": target,
            "sensitivity": 61,
            "feather_px": 7,
            "expansion_px": -2,
            "invert": True,
            "outside": outside,
            "bake_fill": bake_fill,
        },
    }


def test_look_defensively_owns_preset_and_returns_fresh_settings():
    from ditherzam.composition import Look

    raw = _preset()
    original = copy.deepcopy(raw)
    look = Look("A", raw)
    raw["dither"]["style"] = "Atkinson"
    raw["dither"]["params"]["mix"] = 0

    assert look.preset == original
    leaked = look.preset
    leaked["dither"]["style"] = "Floyd-Steinberg"
    assert look.preset == original

    settings = look.settings
    settings.params["mix"] = 12
    assert look.settings.params == {"mix": 100}


def test_look_exposes_all_eight_mask_fields_and_legacy_defaults():
    from ditherzam.composition import Look

    mask = Look("masked", _preset()).smart_mask
    assert (
        mask.enabled, mask.target, mask.sensitivity, mask.feather_px,
        mask.expansion_px, mask.invert, mask.outside, mask.bake_fill,
    ) == (
        True, MaskTarget.SUBJECT, 61, 7, -2, True,
        OutsideMode.TRANSPARENT, False,
    )

    legacy = Look("legacy", {"dither": {"style": "None"}}).smart_mask
    assert legacy.enabled is False
    assert legacy.bake_fill is False


def test_equivalent_looks_have_equal_stable_signatures_not_identity_signatures():
    from ditherzam.composition import Look

    a = Look("display A", _preset())
    b = Look("display B", copy.deepcopy(_preset()))
    assert a is not b
    assert a.signature == b.signature
    assert hash(a.signature) == hash(b.signature)

    changed = _preset()
    changed["smart_mask"]["sensitivity"] = 62
    assert Look("C", changed).signature != a.signature


def test_look_builds_current_color_engine_and_effect_stack():
    from ditherzam.composition import Look

    look = Look("A", _preset())
    engine = look.build_color_engine()
    stack = look.build_effect_stack()
    assert engine.mode == "nearest"
    assert engine.source_dither == 37
    assert engine.source_dither_brighten is True
    assert np.array_equal(
        engine.palette.colors,
        np.array([[0, 0, 0], [255, 255, 255]], np.float32),
    )
    assert stack.items == [("Sharpen", {"amount": 0})]


def test_source_color_controls_affect_signature_and_roundtrip():
    from ditherzam.composition import Look

    raw = _preset()
    look = Look("A", raw)
    assert look.preset["color"]["source_dither"] == 37
    assert look.preset["color"]["source_dither_brighten"] is True

    changed = copy.deepcopy(raw)
    changed["color"]["source_dither"] = 82
    changed["color"]["source_dither_brighten"] = False
    assert Look("B", changed).signature != look.signature


def test_look_rejects_empty_name_and_non_mapping_preset():
    from ditherzam.composition import Look

    with pytest.raises(ValueError, match="name"):
        Look(" ", _preset())
    with pytest.raises(ValueError, match="preset"):
        Look("A", [])


def test_composition_roundtrip_preserves_mask_settings_but_no_runtime_mask_data():
    from ditherzam.composition import (
        Composition, Look, LookClip, composition_from_dict, composition_to_dict,
    )

    comp = Composition(length=12)
    comp.add_clip(LookClip(Look("A", _preset()), 0, 12))
    payload = composition_to_dict(comp)
    text = repr(payload).lower()
    for forbidden in ("probability", "source_identity", "generation", "overlay"):
        assert forbidden not in text

    restored = composition_from_dict(payload)
    assert restored.clips[0].look.smart_mask == comp.clips[0].look.smart_mask
    assert restored.clips[0].look.preset == comp.clips[0].look.preset


def test_composition_validation_and_half_open_resolution_are_exact():
    from ditherzam.composition import Composition, Look, LookClip, TransitionSpec

    comp = Composition(length=10)
    comp.add_clip(LookClip(Look("A", _preset(enabled=False)), 0, 5))
    comp.add_clip(LookClip(Look("B", _preset(enabled=False)), 5, 10))
    comp.set_transition(1, TransitionSpec("crossfade", 2, {}))
    comp.validate()

    assert comp.resolve(4).kind == "hold"
    start = comp.resolve(5)
    assert start.kind == "transition" and start.t == 0.0
    assert comp.resolve(6).t == 0.5
    with pytest.raises(IndexError):
        comp.resolve(10)

    broken = Composition(length=10)
    broken.add_clip(LookClip(Look("A", _preset(enabled=False)), 0, 4))
    broken.add_clip(LookClip(Look("B", _preset(enabled=False)), 5, 10))
    with pytest.raises(ValueError, match="contiguous"):
        broken.validate()
