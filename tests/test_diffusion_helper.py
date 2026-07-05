import numpy as np
from ditherzam.dithering.kernels.error_diffusion import _diffuse, _diffuse_row


def test_diffuse_matches_floyd_shape_and_binary():
    img = np.full((3, 3), 100.0, dtype=np.float32)
    offs = np.array([[0, 1], [1, -1], [1, 0], [1, 1]], dtype=np.int64)
    wts = np.array([7.0, 3.0, 5.0, 1.0], dtype=np.float32)
    out = _diffuse(img.copy(), 128.0, offs, wts, 16.0)
    assert out.shape == (3, 3)
    assert out.dtype == np.float32
    assert set(np.unique(out).tolist()) <= {0.0, 255.0}


def test_diffuse_all_white_stays_white():
    img = np.full((4, 4), 255.0, dtype=np.float32)
    offs = np.array([[0, 1], [1, 0]], dtype=np.int64)
    wts = np.array([1.0, 1.0], dtype=np.float32)
    out = _diffuse(img.copy(), 128.0, offs, wts, 2.0)
    assert np.all(out == 255.0)


def test_diffuse_row_is_binary():
    img = np.tile(np.linspace(0, 255, 6, dtype=np.float32), (2, 1))
    out = _diffuse_row(img.copy(), 128.0, 0.9)
    assert out.shape == (2, 6)
    assert set(np.unique(out).tolist()) <= {0.0, 255.0}
