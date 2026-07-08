import numpy as np
import pytest
from ditherzam.color.palette import Palette
from ditherzam.color.ramp import build_ramp, RAMP_MODES

DUO = Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]])
# deliberately NOT luminance-ordered, to exercise sorting/glitch:
TRI = Palette.from_list("tri", [[255, 255, 255], [0, 0, 0], [200, 50, 50]])

_LUMW = np.array([0.299, 0.587, 0.114], np.float32)
def _lum(rows):
    return (np.asarray(rows, np.float32) @ _LUMW)


@pytest.mark.parametrize("mode", RAMP_MODES)
@pytest.mark.parametrize("depth", [1, 2, 5, 8, 64])
def test_shape_dtype_range(mode, depth):
    r = build_ramp(TRI, depth, mode)
    assert r.shape == (depth, 3)
    assert r.dtype == np.float32
    assert r.min() >= 0.0 and r.max() <= 255.0


def test_depth_clamped():
    assert build_ramp(TRI, 0, "match").shape[0] == 1
    assert build_ramp(TRI, 999, "match").shape[0] == 64


@pytest.mark.parametrize("mode", ["match", "interpolated"])
def test_luminance_monotonic(mode):
    r = build_ramp(TRI, 16, mode)
    lum = _lum(r)
    assert np.all(np.diff(lum) >= -1e-3)  # non-decreasing


def test_match_uses_only_palette_colors():
    r = build_ramp(TRI, 10, "match")
    allowed = {tuple(np.round(c).astype(int)) for c in TRI.colors}
    got = {tuple(np.round(c).astype(int)) for c in r}
    assert got <= allowed


def test_interpolated_blends_between_anchors():
    # a value strictly between the two DUO endpoints must appear
    r = build_ramp(DUO, 5, "interpolated")
    mids = r[(r[:, 0] > 5) & (r[:, 0] < 250)]
    assert mids.shape[0] >= 1


def test_glitch_is_raw_order_modulo_k():
    r = build_ramp(TRI, 7, "glitch")
    for i in range(7):
        np.testing.assert_array_equal(r[i], TRI.colors[i % 3])


def test_reverse_is_match_flipped():
    m = build_ramp(TRI, 9, "match")
    rev = build_ramp(TRI, 9, "reverse")
    np.testing.assert_array_equal(rev, m[::-1])


def test_hue_cycle_is_palette_independent_and_colorful():
    a = build_ramp(TRI, 12, "hue_cycle")
    b = build_ramp(DUO, 12, "hue_cycle")
    np.testing.assert_array_equal(a, b)          # ignores palette
    assert a.std(axis=0).sum() > 10.0            # actual color variation


def test_banded_repeats_every_k():
    r = build_ramp(TRI, 9, "banded")             # K=3
    np.testing.assert_array_equal(r[0], r[3])
    np.testing.assert_array_equal(r[3], r[6])


def test_deterministic():
    np.testing.assert_array_equal(build_ramp(TRI, 20, "glitch"),
                                  build_ramp(TRI, 20, "glitch"))


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_ramp(TRI, 4, "nope")
