from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ditherzam.ui.main_window import ImageEditor, _DecodeWorker


class _FakeSettings:
    def value(self, key, defaultValue=None, type=None):
        return defaultValue

    def setValue(self, key, value):
        pass


def _editor(qapp_fixture) -> ImageEditor:
    return ImageEditor(preference_store=_FakeSettings())


def test_transparent_png_decode_retains_exact_straight_rgba(tmp_path, qapp_fixture):
    expected = np.array(
        [[[11, 22, 33, 0], [44, 55, 66, 127]],
         [[77, 88, 99, 200], [111, 122, 133, 255]]],
        dtype=np.uint8,
    )
    path = tmp_path / "alpha.png"
    Image.fromarray(expected, "RGBA").save(path)
    captured = []
    worker = _DecodeWorker(str(path))
    worker.signals.finished.connect(lambda gray, rgb, rgba: captured.append((gray, rgb, rgba)))
    worker.run()

    assert len(captured) == 1
    gray, rgb, rgba = captured[0]
    assert np.array_equal(rgba, expected)
    assert np.array_equal(rgb, expected[..., :3])
    assert gray.dtype == np.float32
    assert not rgba.flags.writeable


def test_load_rgb_synthesizes_owned_opaque_rgba(qapp_fixture):
    editor = _editor(qapp_fixture)
    gray = np.array([[1.5, 2.5]], dtype=np.float32)
    rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    editor.load_array(gray, rgb)

    rgba = editor._base_rgba
    assert rgba is not None
    assert np.array_equal(rgba[..., :3], rgb)
    assert np.all(rgba[..., 3] == 255)
    assert rgba.flags.c_contiguous
    assert not rgba.flags.writeable
    rgb[:] = 0
    assert np.array_equal(rgba[0, 0], [10, 20, 30, 255])
    assert editor._base_gray is gray
    assert editor._base_rgb is not None


def test_load_grayscale_synthesizes_rgba_without_changing_render_input(qapp_fixture):
    editor = _editor(qapp_fixture)
    gray = np.array([[-2.0, 12.9, 300.0]], dtype=np.float32)
    editor.load_array(gray)

    assert editor._base_gray is gray
    assert editor._base_rgb is None
    assert np.array_equal(
        editor._base_rgba,
        np.array([[[0, 0, 0, 255], [12, 12, 12, 255], [255, 255, 255, 255]]], dtype=np.uint8),
    )


@pytest.mark.parametrize(
    ("gray", "rgb", "rgba"),
    [
        (np.zeros((2, 2, 1)), None, None),
        (np.zeros((2, 2)), np.zeros((2, 3, 3)), None),
        (np.zeros((2, 2)), None, np.zeros((2, 2, 3))),
        (np.zeros((2, 2)), np.zeros((2, 2, 3)), np.zeros((3, 2, 4))),
    ],
)
def test_invalid_source_shapes_fail_atomically(qapp_fixture, gray, rgb, rgba):
    editor = _editor(qapp_fixture)
    old_gray = np.ones((1, 1), dtype=np.float32)
    editor.load_array(old_gray)
    old_rgb = editor._base_rgb
    old_rgba = editor._base_rgba

    with pytest.raises(ValueError):
        editor.load_array(gray, rgb, rgba)

    assert editor._base_gray is old_gray
    assert editor._base_rgb is old_rgb
    assert editor._base_rgba is old_rgba


def test_explicit_rgba_is_owned_and_rgb_must_match(qapp_fixture):
    editor = _editor(qapp_fixture)
    gray = np.zeros((1, 1), dtype=np.float32)
    rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)
    rgba = np.array([[[1, 2, 3, 4]]], dtype=np.uint8)
    editor.load_array(gray, rgb, rgba)
    rgba[:] = 255
    assert np.array_equal(editor._base_rgba, [[[1, 2, 3, 4]]])

    with pytest.raises(ValueError, match="must match"):
        editor.load_array(gray, rgb, np.array([[[9, 2, 3, 4]]], dtype=np.uint8))
