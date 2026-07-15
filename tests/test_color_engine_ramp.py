import numpy as np
from ditherzam.color.palette import Palette
from ditherzam.color.engine import ColorEngine
from ditherzam.color.ramp import build_ramp

DUO = Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]])
QUAD = Palette.from_list("quad", [[0, 0, 0], [90, 0, 0], [0, 160, 0], [255, 255, 255]])


def test_ramp_mode_recolors_levels():
    eng = ColorEngine(DUO, mode="ramp", depth=2, mapping="match")
    gray = np.array([[0.0, 255.0]], np.float32)
    out = eng.map(gray)
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[0, 1].tolist() == [255, 255, 255]


def test_ramp_depth_limits_distinct_tones():
    eng = ColorEngine(QUAD, mode="ramp", depth=3, mapping="match")
    grad = np.tile(np.linspace(0, 255, 64, dtype=np.float32), (4, 1))
    out = eng.map(grad)
    uniq = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    assert len(uniq) <= 3


def test_ramp_output_matches_builder():
    eng = ColorEngine(QUAD, mode="ramp", depth=4, mapping="interpolated")
    ramp = build_ramp(QUAD, 4, "interpolated")
    gray = np.array([[0.0, 85.0, 170.0, 255.0]], np.float32)
    out = eng.map(gray).astype(np.float32)
    # each pixel maps to its binned ramp entry
    levels = np.clip(np.round(gray / 255.0 * 3), 0, 3).astype(int)
    expected = np.round(ramp[levels[0]]).astype(np.uint8)
    np.testing.assert_array_equal(out[0], expected)


def test_ramp_depth_one_is_single_color():
    eng = ColorEngine(QUAD, mode="ramp", depth=1, mapping="match")
    out = eng.map(np.tile(np.linspace(0, 255, 20, np.float32), (3, 1)))
    uniq = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    assert len(uniq) == 1


def test_ramp_cache_rebuilds_on_param_change():
    eng = ColorEngine(QUAD, mode="ramp", depth=2, mapping="match")
    gray = np.tile(np.linspace(0, 255, 16, np.float32), (2, 1))
    a = eng.map(gray)
    eng.depth = 5
    b = eng.map(gray)
    assert not np.array_equal(a, b)  # changing depth changes output


def test_existing_modes_untouched():
    eng = ColorEngine(DUO, mode="nearest")
    out = eng.map(np.array([[10.0, 240.0]], np.float32))
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[0, 1].tolist() == [255, 255, 255]
