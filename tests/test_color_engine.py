import numpy as np
import pytest
from ditherzam.color.palette import Palette, builtin_palettes
from ditherzam.color.engine import ColorEngine, nearest_indices

DUO = Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]])
QUAD = Palette.from_list("quad", [[0, 0, 0], [128, 0, 0], [0, 128, 0], [255, 255, 255]])


def test_off_mode_passthrough_to_rgb():
    eng = ColorEngine(DUO, mode="off")
    gray = np.full((4, 4), 100.0, np.float32)
    out = eng.map(gray)
    assert out.shape == (4, 4, 3)
    assert out.dtype == np.uint8
    assert np.all(out == 100)


def test_off_mode_clamps():
    eng = ColorEngine(DUO, mode="off")
    rgb = np.array([[[-20.0, 300.0, 128.0]]], np.float32)
    out = eng.map(rgb)
    assert out[0, 0].tolist() == [0, 255, 128]


def test_nearest_snaps_gray_to_palette():
    eng = ColorEngine(DUO, mode="nearest")
    gray = np.array([[10.0, 240.0]], np.float32)
    out = eng.map(gray)
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[0, 1].tolist() == [255, 255, 255]


def test_nearest_output_only_palette_colors():
    eng = ColorEngine(QUAD, mode="nearest")
    img = np.random.RandomState(1).randint(0, 256, (8, 8, 3)).astype(np.float32)
    out = eng.map(img)
    uniq = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    allowed = {tuple(int(round(v)) for v in c) for c in QUAD.colors}
    assert uniq <= allowed


def test_source_mode_preserves_spatial_hues_while_simplifying():
    source = np.array([
        [[240, 20, 20], [20, 30, 235]],
        [[225, 35, 25], [25, 20, 220]],
    ], np.float32)
    palette = Palette.from_list("source", [[255, 0, 0], [0, 0, 255]])
    out = ColorEngine(palette, mode="source", source_rgb=source,
                      source_dither=0).map(
        np.full((2, 2), 127.5, np.float32))
    np.testing.assert_array_equal(out[:, 0], np.array([[255, 0, 0], [255, 0, 0]]))
    np.testing.assert_array_equal(out[:, 1], np.array([[0, 0, 255], [0, 0, 255]]))


def test_source_mode_resizes_source_to_preview_shape():
    source = np.zeros((4, 4, 3), np.float32)
    source[:, 2:] = [255, 255, 255]
    palette = Palette.from_list("bw", [[0, 0, 0], [255, 255, 255]])
    out = ColorEngine(palette, "source", source_rgb=source,
                      source_dither=0).map(
        np.zeros((2, 2), np.float32))
    assert out.shape == (2, 2, 3)
    np.testing.assert_array_equal(out[:, 0], 0)
    np.testing.assert_array_equal(out[:, 1], 255)


def test_source_dither_adds_texture_without_swapping_spatial_hues():
    source = np.array([[[220, 30, 30], [25, 30, 220]]], np.float32)
    palette = Palette.from_list("rb", [[220, 30, 30], [25, 30, 220]])
    dither = np.array([[0.0, 255.0]], np.float32)
    plain = ColorEngine(palette, "source", source_rgb=source,
                        source_dither=0).map(dither)
    textured = ColorEngine(palette, "source", source_rgb=source,
                           source_dither=50).map(dither)
    assert not np.array_equal(textured, plain)
    assert textured[0, 0, 0] > textured[0, 0, 2]  # left remains red
    assert textured[0, 1, 2] > textured[0, 1, 0]  # right remains blue


def test_full_source_dither_colors_the_marks_themselves():
    source = np.array([[[220, 30, 30], [25, 30, 220]]], np.float32)
    palette = Palette.from_list("rb", [[220, 30, 30], [25, 30, 220]])
    dither = np.array([[64.0, 192.0]], np.float32)
    out = ColorEngine(palette, "source", source_rgb=source,
                      source_dither=100).map(dither)
    # The dither value becomes each mark's peak channel; local source hue picks
    # which channel carries it. This distinguishes foreground-colored marks
    # from a monochrome mask composited over/under a source-color layer.
    assert out[0, 0, 0] == 64 and out[0, 0, 0] > out[0, 0, 2]
    assert out[0, 1, 2] == 192 and out[0, 1, 2] > out[0, 1, 0]


def test_nearest_indices_helper():
    pal = np.array([[0, 0, 0], [255, 255, 255]], np.float32)
    rgb = np.array([[[10, 10, 10], [200, 200, 200]]], np.float32)
    idx = nearest_indices(rgb, pal)
    assert idx.tolist() == [[0, 1]]


def _ref_nearest_indices(rgb_f32, palette_f32):
    """The original broadcast implementation, kept as the equivalence oracle."""
    diff = rgb_f32[:, :, None, :] - palette_f32[None, None, :, :]
    dist = (diff * diff).sum(axis=-1)
    return dist.argmin(axis=-1)


@pytest.mark.parametrize("pal_name", list(builtin_palettes().keys()))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nearest_indices_bit_identical_to_reference(pal_name, seed):
    pal = builtin_palettes()[pal_name].colors.astype(np.float32)
    rng = np.random.default_rng(seed)
    # non-integer float values to stress tie-breaking / rounding
    rgb = rng.uniform(-5, 260, size=(23, 29, 3)).astype(np.float32)
    got = nearest_indices(rgb, pal)
    ref = _ref_nearest_indices(rgb, pal)
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("mode", ["nearest", "ordered"])
def test_map_bit_identical_across_palettes(mode):
    rng = np.random.default_rng(9)
    rgb = rng.uniform(0, 255, size=(31, 17, 3)).astype(np.float32)
    for name, pal in builtin_palettes().items():
        eng = ColorEngine(pal, mode)
        # reference map: same body but with the oracle nearest-indices
        got = eng.map(rgb)
        assert got.dtype == np.uint8 and got.shape == (31, 17, 3)
        # snap-only invariant: every output color is a palette color
        allowed = {tuple(int(v) for v in c) for c in pal.colors}
        uniq = {tuple(c) for c in got.reshape(-1, 3).tolist()}
        assert uniq <= allowed


def test_ordered_output_only_palette_colors():
    eng = ColorEngine(QUAD, mode="ordered")
    img = np.random.RandomState(2).randint(0, 256, (8, 8, 3)).astype(np.float32)
    out = eng.map(img)
    uniq = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    allowed = {tuple(int(round(v)) for v in c) for c in QUAD.colors}
    assert uniq <= allowed


def test_ordered_dithers_flat_midgray():
    # nearest would make a solid fill; ordered must mix both palette colors
    eng = ColorEngine(DUO, mode="ordered")
    gray = np.full((8, 8), 127.0, np.float32)
    out = eng.map(gray)
    uniq = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    assert (0, 0, 0) in uniq
    assert (255, 255, 255) in uniq


def test_ordered_is_deterministic():
    eng = ColorEngine(DUO, mode="ordered")
    gray = np.full((8, 8), 127.0, np.float32)
    np.testing.assert_array_equal(eng.map(gray), eng.map(gray))


def test_diffused_output_only_palette_colors():
    eng = ColorEngine(QUAD, mode="diffused")
    img = np.random.RandomState(4).randint(0, 256, (8, 8, 3)).astype(np.float32)
    out = eng.map(img)
    uniq = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    allowed = {tuple(int(round(v)) for v in c) for c in QUAD.colors}
    assert uniq <= allowed


def test_diffused_dithers_flat_midgray():
    eng = ColorEngine(DUO, mode="diffused")
    gray = np.full((8, 8), 127.0, np.float32)
    out = eng.map(gray)
    uniq = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    assert (0, 0, 0) in uniq and (255, 255, 255) in uniq


def test_diffused_preserves_average():
    # error diffusion of mid-gray on a black/white palette ~= 50% each
    eng = ColorEngine(DUO, mode="diffused")
    gray = np.full((16, 16), 127.0, np.float32)
    out = eng.map(gray).astype(np.float32)
    assert 100.0 < out.mean() < 155.0


def test_diffused_shape_and_dtype():
    eng = ColorEngine(QUAD, mode="diffused")
    out = eng.map(np.full((5, 6), 60.0, np.float32))
    assert out.shape == (5, 6, 3) and out.dtype == np.uint8
