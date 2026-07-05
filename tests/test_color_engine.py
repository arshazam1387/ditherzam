import numpy as np
from ditherzam.color.palette import Palette
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


def test_nearest_indices_helper():
    pal = np.array([[0, 0, 0], [255, 255, 255]], np.float32)
    rgb = np.array([[[10, 10, 10], [200, 200, 200]]], np.float32)
    idx = nearest_indices(rgb, pal)
    assert idx.tolist() == [[0, 1]]


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
