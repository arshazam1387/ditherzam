"""Task 3.4: invert-stage (render L9 / render_cached L6) exact scratch-buffer
reuse. ``clamp_u8(apply_invert(rgb_u8.astype(np.float32), True))`` allocated
FOUR full-image temporaries per call (uint8->float32 cast, subtraction, a
redundant same-dtype astype copy inside apply_invert, and clip's own copy)
before the mandatory final uint8 output. apply_invert(out=) and
clamp_u8(inplace=) collapse that to ONE call-private float32 scratch buffer
plus the mandatory final allocation -- never retained, never cached, never
returned as itself (the returned array is always the fresh uint8 result of
.astype), so it can never alias a value ``render_cached`` stores in ``c`` or
hands back to a caller. Byte-identical to the pre-3.4 allocating path."""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ditherzam.adjustments import apply_invert
from ditherzam.imaging import clamp_u8
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.color.palette import Palette
from ditherzam.color.engine import ColorEngine


def _sha256(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _adversarial_rgb_u8(h=37, w=53, seed=20260710):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    a.flat[:6] = [0, 1, 127, 128, 254, 255]
    return a


# --------------------------------------------------------- unit: adjustments.py

def test_apply_invert_out_matches_default_adversarial():
    img = _adversarial_rgb_u8()
    expected = apply_invert(img.astype(np.float32), True)
    buf = np.empty_like(img, dtype=np.float32)
    actual = apply_invert(img, True, out=buf)
    assert actual is buf
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


def test_apply_invert_out_does_not_mutate_source():
    img = _adversarial_rgb_u8()
    original = img.copy()
    buf = np.empty_like(img, dtype=np.float32)
    apply_invert(img, True, out=buf)
    np.testing.assert_array_equal(img, original)


def test_apply_invert_disabled_ignores_out_and_returns_input_unchanged():
    img = _adversarial_rgb_u8()
    buf = np.full(img.shape, -999.0, dtype=np.float32)
    result = apply_invert(img, False, out=buf)
    assert result is img            # identity path, buf untouched
    np.testing.assert_array_equal(buf, np.full(img.shape, -999.0, np.float32))


# ------------------------------------------------------------- unit: imaging.py

def test_clamp_u8_inplace_matches_default_adversarial():
    rng = np.random.default_rng(20260710)
    src = np.concatenate((
        np.array([-255.0, -0.0, 0.0, 254.99998, 255.0, 300.0], dtype=np.float32),
        rng.uniform(-50, 305, 10_000).astype(np.float32),
    ))
    expected = clamp_u8(src)
    scratch = src.copy()
    actual = clamp_u8(scratch, inplace=True)
    np.testing.assert_array_equal(actual, expected)
    assert actual is not scratch    # final uint8 alloc is always fresh


def test_clamp_u8_inplace_mutates_the_passed_buffer_in_place():
    scratch = np.array([-5.0, 300.0, 100.0], dtype=np.float32)
    clamp_u8(scratch, inplace=True)
    np.testing.assert_array_equal(scratch, np.array([0.0, 255.0, 100.0], np.float32))


def test_clamp_u8_default_still_never_mutates():
    src = np.array([-5.0, 300.0, 100.0], dtype=np.float32)
    original = src.copy()
    clamp_u8(src)
    np.testing.assert_array_equal(src, original)


# --------------------------------------------------------- render() end-to-end

def _manual_old_invert_path(rgb_u8):
    """Exactly what render.py/render_cached.py did before 3.4 (allocating)."""
    return clamp_u8(apply_invert(rgb_u8.astype(np.float32), True))


def test_render_invert_matches_manual_unfused_chain_adversarial():
    base = np.linspace(0, 255, 24 * 24, dtype=np.float32).reshape(24, 24)
    eng = ColorEngine(Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]]), "nearest")
    for style in ("None", "Floyd-Steinberg"):
        p = RenderPipeline(registry, color_engine=eng)
        settings_noinv = RenderSettings(style=style, scale=1, invert=False)
        rgb_u8_noinv = p.render(base, settings_noinv)
        expected = _manual_old_invert_path(rgb_u8_noinv)

        settings_inv = RenderSettings(style=style, scale=1, invert=True)
        actual = p.render(base, settings_inv)
        np.testing.assert_array_equal(actual, expected)


def test_render_cached_invert_matches_render_adversarial():
    base = np.linspace(0, 255, 20 * 30, dtype=np.float32).reshape(20, 30)
    eng = ColorEngine(Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]]), "nearest")
    p_ref = RenderPipeline(registry, color_engine=eng)
    p_cached = RenderPipeline(registry, color_engine=eng)
    for saturation in (10, 50, 90):
        settings = RenderSettings(style="Atkinson", scale=1, saturation=saturation, invert=True)
        expected = p_ref.render(base, settings)
        actual = p_cached.render_cached(base, settings)
        np.testing.assert_array_equal(actual, expected)


# ----------------------------------------------------- consecutive-call stability

def test_render_consecutive_invert_calls_do_not_leak_state():
    """A reused invert scratch buffer must never let call N corrupt a
    previously-returned array from call N-1 (buf is call-private and freed
    at return, never retained)."""
    base = np.linspace(0, 255, 16 * 16, dtype=np.float32).reshape(16, 16)
    eng = ColorEngine(Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]]), "nearest")
    p = RenderPipeline(registry, color_engine=eng)

    settings_seq = [
        RenderSettings(style="None", invert=True, contrast=70),
        RenderSettings(style="None", invert=False, contrast=70),
        RenderSettings(style="None", invert=True, contrast=20),
        RenderSettings(style="None", invert=True, midtones=80),
    ]
    outputs = [p.render(base, s) for s in settings_seq]
    snapshots = [o.copy() for o in outputs]
    # Re-run the same sequence fresh; each call's output must be unaffected
    # by having previously produced (and since discarded) earlier outputs.
    for out, snap, s in zip(outputs, snapshots, settings_seq):
        np.testing.assert_array_equal(out, snap)
        np.testing.assert_array_equal(out, RenderPipeline(registry, color_engine=eng).render(base, s))


def test_render_cached_consecutive_invert_toggle_does_not_corrupt_fx_cache():
    base = np.linspace(0, 255, 16 * 16, dtype=np.float32).reshape(16, 16)
    eng = ColorEngine(Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]]), "nearest")
    p = RenderPipeline(registry, color_engine=eng)
    s = RenderSettings(style="None", saturation=60)

    out_noinv_1 = p.render_cached(base, s)
    fx_before = p._cache.get(id(base))["fx"].copy()

    out_inv = p.render_cached(base, RenderSettings(style="None", saturation=60, invert=True))
    fx_after = p._cache.get(id(base))["fx"]
    # invert must never write through into the cached fx group
    np.testing.assert_array_equal(fx_after, fx_before)

    out_noinv_2 = p.render_cached(base, s)
    np.testing.assert_array_equal(out_noinv_2, out_noinv_1)
    np.testing.assert_array_equal(out_inv, clamp_u8(apply_invert(fx_before.astype(np.float32), True)))


# -------------------------------------------------------------- concurrency

def test_render_concurrent_invert_and_plain_do_not_cross_contaminate():
    base = np.linspace(0, 255, 24 * 24, dtype=np.float32).reshape(24, 24)
    eng = ColorEngine(Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]]), "nearest")
    p = RenderPipeline(registry, color_engine=eng)

    settings_inv = RenderSettings(style="Floyd-Steinberg", scale=1, invert=True)
    settings_noinv = RenderSettings(style="Floyd-Steinberg", scale=1, invert=False)
    expected_inv = p.render(base, settings_inv)
    expected_noinv = p.render(base, settings_noinv)

    jobs = [settings_inv, settings_noinv] * 16
    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(lambda s: p.render(base, s), jobs))

    for s, out in zip(jobs, outputs):
        expected = expected_inv if s is settings_inv else expected_noinv
        np.testing.assert_array_equal(out, expected)


def test_render_cached_concurrent_invert_and_plain_do_not_cross_contaminate():
    base = np.linspace(0, 255, 24 * 24, dtype=np.float32).reshape(24, 24)
    eng = ColorEngine(Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]]), "nearest")
    p = RenderPipeline(registry, color_engine=eng)

    settings_inv = RenderSettings(style="Floyd-Steinberg", scale=1, invert=True)
    settings_noinv = RenderSettings(style="Floyd-Steinberg", scale=1, invert=False)
    expected_inv = p.render(base, settings_inv)
    expected_noinv = p.render(base, settings_noinv)

    jobs = [settings_inv, settings_noinv] * 16
    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(lambda s: p.render_cached(base, s), jobs))

    for s, out in zip(jobs, outputs):
        expected = expected_inv if s is settings_inv else expected_noinv
        np.testing.assert_array_equal(out, expected)


def test_render_and_render_cached_interleaved_concurrently_agree():
    base = np.linspace(0, 255, 24 * 24, dtype=np.float32).reshape(24, 24)
    eng = ColorEngine(Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]]), "nearest")
    p = RenderPipeline(registry, color_engine=eng)
    settings = RenderSettings(style="Floyd-Steinberg", scale=1, invert=True)
    expected = p.render(base, settings)

    def call(i):
        return p.render(base, settings) if i % 2 == 0 else p.render_cached(base, settings)

    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(call, range(32)))

    for out in outputs:
        np.testing.assert_array_equal(out, expected)
