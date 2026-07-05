import numpy as np
import pytest

pytest.importorskip("PySide6")
from ditherzam.ui.convert import numpy_to_qimage, qimage_to_numpy


def test_roundtrip_rgb():
    a = np.random.RandomState(0).randint(0, 256, (5, 7, 3), np.uint8)
    q = numpy_to_qimage(a)
    assert q.width() == 7 and q.height() == 5
    b = qimage_to_numpy(q)
    np.testing.assert_array_equal(a, b)


def test_grayscale_2d_is_broadcast_to_rgb():
    g = np.array([[0, 128, 255]], np.uint8)
    q = numpy_to_qimage(g)
    assert q.width() == 3 and q.height() == 1
    b = qimage_to_numpy(q)
    assert b.shape == (1, 3, 3)
    np.testing.assert_array_equal(b[0, :, 0], b[0, :, 1])
    np.testing.assert_array_equal(b[0, :, 0], [0, 128, 255])


def test_non_contiguous_input_is_handled():
    a = np.random.RandomState(1).randint(0, 256, (4, 6, 3), np.uint8)
    view = a[::1, ::1, :]  # keep, then force a non-contiguous slice below
    sliced = np.ascontiguousarray(a)[:, ::2, :]  # width becomes 3, non-standard stride
    q = numpy_to_qimage(sliced)
    b = qimage_to_numpy(q)
    np.testing.assert_array_equal(sliced, b)
