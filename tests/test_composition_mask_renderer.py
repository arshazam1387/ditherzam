from __future__ import annotations

import numpy as np
import pytest

from ditherzam.masking.contracts import (
    InferenceIdentity, ModelIdentity, ProbabilityMap, source_identity,
)


def _preset(*, enabled=False, target="subject", outside="transparent", bake=False,
            threshold=50):
    return {
        "adjustments": {
            "contrast": 50, "midtones": 50, "highlights": 50,
            "luminance_threshold": threshold, "blur": 0, "saturation": 50,
            "invert": False,
        },
        "dither": {
            "style": "None", "scale": 1, "depth": 2,
            "color_mapping": "match", "preview_disabled": False, "params": {},
        },
        "smart_mask": {
            "enabled": enabled, "target": target, "sensitivity": 50,
            "feather_px": 0, "expansion_px": 0, "invert": False,
            "outside": outside, "bake_fill": bake,
        },
    }


def _source():
    gray = np.array([[0, 64], [192, 255]], np.float32)
    rgba = np.dstack((
        gray.astype(np.uint8), gray.astype(np.uint8), gray.astype(np.uint8),
        np.full(gray.shape, 255, np.uint8),
    ))
    rgba.flags.writeable = False
    return gray, rgba


def _probability(rgba):
    sid = source_identity(rgba)
    model = ModelIdentity("test", "1", "1" * 64)
    identity = InferenceIdentity(sid, model, "test-v1", "primary")
    return ProbabilityMap(identity, np.array([[1, 0], [1, 0]], np.float32))


def test_unmasked_look_renderer_is_byte_identical_to_direct_pipeline():
    from ditherzam.composition import Look, LookRenderer
    from ditherzam.dithering import registry
    from ditherzam.render import RenderPipeline

    gray, rgba = _source()
    look = Look("plain", _preset(enabled=False))
    renderer = LookRenderer(registry, gray, rgba)
    got = renderer.render(look)
    expected = RenderPipeline(
        registry, look.build_color_engine(source_rgb=rgba[..., :3]),
        look.build_effect_stack(),
    ).render(gray, look.settings)
    assert np.array_equal(got, expected)


def test_masked_look_matches_existing_mask_render_integration():
    from types import SimpleNamespace

    from ditherzam.composition import Look, LookRenderer
    from ditherzam.dithering import registry
    from ditherzam.masking.render import render_with_mask
    from ditherzam.render import RenderPipeline

    gray, rgba = _source()
    probability = _probability(rgba)
    look = Look("masked", _preset(enabled=True))
    renderer = LookRenderer(registry, gray, rgba, probability=probability)
    got = renderer.render(look)

    pipeline = RenderPipeline(
        registry, look.build_color_engine(source_rgb=rgba[..., :3]),
        look.build_effect_stack(),
    )
    context = SimpleNamespace(
        source=probability.identity.source, source_rgba=rgba,
        probability=probability, settings=look.smart_mask,
    )
    expected = render_with_mask(
        lambda: pipeline.render(gray, look.settings), context,
        target_shape=gray.shape,
    )
    assert np.array_equal(got, expected)


def test_whole_image_bypasses_probability_requirement_and_mask_work():
    from ditherzam.composition import Look, LookRenderer
    from ditherzam.dithering import registry

    gray, rgba = _source()
    renderer = LookRenderer(registry, gray, rgba)
    got = renderer.render(Look("whole", _preset(enabled=True, target="whole_image")))
    expected = renderer.render(Look("off", _preset(enabled=False)))
    assert np.array_equal(got, expected)


def test_subject_mask_without_matching_probability_fails_closed():
    from ditherzam.composition import CompositionMaskError, Look, LookRenderer
    from ditherzam.dithering import registry

    gray, rgba = _source()
    renderer = LookRenderer(registry, gray, rgba)
    with pytest.raises(CompositionMaskError, match="probability"):
        renderer.render(Look("masked", _preset(enabled=True)))


def test_equivalent_looks_reuse_value_keyed_pipeline():
    from ditherzam.composition import Look, LookRenderer
    from ditherzam.dithering import registry

    gray, rgba = _source()
    renderer = LookRenderer(registry, gray, rgba)
    renderer.render(Look("A", _preset(enabled=False)))
    renderer.render(Look("B", _preset(enabled=False)))
    assert renderer.pipeline_count == 1


def test_mask_setting_change_does_not_create_a_second_pipeline():
    from ditherzam.composition import Look, LookRenderer
    from ditherzam.dithering import registry

    gray, rgba = _source()
    probability = _probability(rgba)
    renderer = LookRenderer(registry, gray, rgba, probability=probability)
    renderer.render(Look("A", _preset(enabled=True)))
    changed = _preset(enabled=True)
    changed["smart_mask"]["sensitivity"] = 75
    renderer.render(Look("B", changed))
    assert renderer.pipeline_count == 1
