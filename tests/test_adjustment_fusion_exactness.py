"""Frozen dither-facing float32 output for future adjustment fusion work."""
import hashlib

import numpy as np

from ditherzam.adjustments import apply_contrast, apply_highlights, apply_midtones


def _chain(image, contrast, midtones, highlights):
    return apply_highlights(
        apply_midtones(apply_contrast(image, contrast), midtones), highlights)


def _fixture():
    rng = np.random.default_rng(20260709)
    boundaries = np.array(
        [0.0, np.nextafter(np.float32(0), np.float32(1)), 1.0,
         127.5, 254.99998, 255.0], dtype=np.float32)
    random = rng.uniform(0, 255, 4090).astype(np.float32)
    return np.concatenate((boundaries, random)).reshape(64, 64)


def _sha256(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def test_tonal_chain_reference_hash_primary_settings():
    result = _chain(_fixture(), 70.0, 30.0, 80.0)
    assert result.dtype == np.float32
    assert _sha256(result) == "2ab563dd74e3f662e722cfd3788664b56b1449c5efbe2c7efd5376afdf07da20"


def test_tonal_chain_reference_hash_fraction_sensitive_settings():
    result = _chain(_fixture(), 13.0, 91.0, 47.0)
    assert result.dtype == np.float32
    assert _sha256(result) == "cecd580a09d8f6c9788167dccb27bfe4598bf5a5f173dc0346e28e82ea85e154"


# ---------------------------------------------------------------- Task 3.2 ----
# render.py now routes contrast/midtones/highlights through ONE shared buffer
# (out=) instead of three allocations. Prove that route is byte-identical to
# the allocating chain above, including against these same frozen hashes.

def _buffer_chain(image, contrast, midtones, highlights):
    """Mirrors render.py's/render_cached's one-buffer routing exactly."""
    buf = np.empty_like(np.asarray(image, dtype=np.float32))
    g = apply_contrast(image, contrast, out=buf)
    g = apply_midtones(g, midtones, out=buf)
    g = apply_highlights(g, highlights, out=buf)
    return g


PARAM_SETS = (
    (70.0, 30.0, 80.0),   # primary settings (frozen hash)
    (13.0, 91.0, 47.0),   # fraction-sensitive settings (frozen hash)
    (50.0, 50.0, 50.0),   # identity
    (0.0, 100.0, 0.0),    # extreme low/high
    (100.0, 0.0, 100.0),
    (37.5, 62.5, 12.5),   # fractional settings
)


def _adversarial_corpus():
    rng = np.random.default_rng(20260709)
    boundaries = np.array(
        [-255.0, -0.0, 0.0, np.nextafter(np.float32(0), np.float32(1)),
         1.0, 127.5, 254.99998, 255.0, 511.0], dtype=np.float32)
    random = rng.uniform(0, 255, 100_000).astype(np.float32)
    return np.concatenate((boundaries, random)).reshape(1, -1)


def test_buffer_route_matches_frozen_hashes():
    fixture = _fixture()
    assert _sha256(_buffer_chain(fixture, 70.0, 30.0, 80.0)) == \
        "2ab563dd74e3f662e722cfd3788664b56b1449c5efbe2c7efd5376afdf07da20"
    assert _sha256(_buffer_chain(fixture, 13.0, 91.0, 47.0)) == \
        "cecd580a09d8f6c9788167dccb27bfe4598bf5a5f173dc0346e28e82ea85e154"


def test_buffer_route_matches_allocating_chain_adversarial_sweep():
    corpus = _adversarial_corpus()
    for contrast, midtones, highlights in PARAM_SETS:
        expected = _chain(corpus, contrast, midtones, highlights)
        actual = _buffer_chain(corpus, contrast, midtones, highlights)
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected)


def test_buffer_route_does_not_mutate_source_array():
    corpus = _adversarial_corpus()
    original = corpus.copy()
    _buffer_chain(corpus, 70.0, 30.0, 80.0)
    np.testing.assert_array_equal(corpus, original)


def test_render_end_to_end_matches_manual_unfused_chain_adversarial():
    """render()'s fused-buffer tonal stage must not change any downstream byte
    versus computing the same pipeline with the original three allocations."""
    from ditherzam.adjustments import apply_blur, apply_saturation
    from ditherzam.dithering import registry
    from ditherzam.dithering.pipeline import apply_dither
    from ditherzam.render import RenderPipeline, RenderSettings

    h, w = 32, 32
    rng = np.random.default_rng(20260709)
    base = rng.uniform(0, 255, size=(h, w)).astype(np.float32)
    boundary = np.array(
        [-255.0, -0.0, 0.0, np.nextafter(np.float32(0), np.float32(1)),
         1.0, 127.5, 254.99998, 255.0, 511.0], dtype=np.float32)
    base.flat[:boundary.size] = boundary

    pipe = RenderPipeline(registry)
    for contrast, midtones, highlights in PARAM_SETS:
        settings = RenderSettings(
            contrast=contrast, midtones=midtones, highlights=highlights, style="None")
        actual = pipe.render(base, settings)

        g = apply_contrast(base, contrast)
        g = apply_midtones(g, midtones)
        g = apply_highlights(g, highlights)
        g = apply_blur(g, settings.blur)
        d = apply_dither(
            g, style=settings.style, scale=settings.scale,
            luminance_threshold=settings.luminance_threshold,
            params=settings.params, registry=registry,
            preview_disabled=settings.preview_disabled,
            threshold_field=None, levels=settings.depth)
        rgb = np.asarray(d, np.float32)
        rgb_u8 = apply_saturation(rgb, settings.saturation, output_u8=True)
        expected = np.asarray(rgb_u8, np.uint8)

        np.testing.assert_array_equal(actual, expected)
