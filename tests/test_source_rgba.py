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
    gray = np.array([[0.0, 12.9, 255.0]], dtype=np.float32)
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
        (np.zeros((2, 2, 1), dtype=np.float32), None, None),
        (np.zeros((2, 2), dtype=np.float32), np.zeros((2, 3, 3), dtype=np.uint8), None),
        (np.zeros((2, 2), dtype=np.float32), None, np.zeros((2, 2, 3), dtype=np.uint8)),
        (np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((3, 2, 4), dtype=np.uint8)),
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


@pytest.mark.parametrize(
    ("gray", "rgb", "rgba", "error"),
    [
        (np.zeros((1, 1), dtype=np.float64), None, None, TypeError),
        ([[0.0]], None, None, TypeError),
        (np.array([[np.nan]], dtype=np.float32), None, None, ValueError),
        (np.array([[np.inf]], dtype=np.float32), None, None, ValueError),
        (np.array([[-0.01]], dtype=np.float32), None, None, ValueError),
        (np.array([[255.01]], dtype=np.float32), None, None, ValueError),
        (np.zeros((0, 1), dtype=np.float32), None, None, ValueError),
        (np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1, 3), dtype=np.int16), None, TypeError),
        (np.zeros((1, 1), dtype=np.float32), None, np.zeros((1, 1, 4), dtype=np.float32), TypeError),
    ],
)
def test_noncanonical_values_and_dtypes_fail_atomically(
    qapp_fixture, gray, rgb, rgba, error
):
    editor = _editor(qapp_fixture)
    old_gray = np.ones((1, 1), dtype=np.float32)
    editor.load_array(old_gray)
    old_rgb = editor._base_rgb
    old_rgba = editor._base_rgba

    with pytest.raises(error):
        editor.load_array(gray, rgb, rgba)

    assert editor._base_gray is old_gray
    assert editor._base_rgb is old_rgb
    assert editor._base_rgba is old_rgba


def test_public_owned_readonly_rgba_is_still_defensively_copied(qapp_fixture):
    editor = _editor(qapp_fixture)
    gray = np.zeros((2, 2), dtype=np.float32)
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba.setflags(write=False)

    editor.load_array(gray, rgba[..., :3].copy(), rgba)

    assert editor._base_rgba is not rgba
    rgba.setflags(write=True)
    rgba[:] = 255
    assert np.all(editor._base_rgba == 0)


def test_private_decode_receiver_adopts_worker_owned_rgba(qapp_fixture, monkeypatch):
    editor = _editor(qapp_fixture)
    gray = np.zeros((2, 2), dtype=np.float32)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba.setflags(write=False)
    monkeypatch.setattr(editor, "schedule_render", lambda: None)

    editor._on_image_decoded(gray, rgb, rgba)

    assert editor._base_rgba is rgba


@pytest.mark.parametrize("kind", ["mutable", "borrowed", "strided"])
def test_programmatic_rgba_without_exclusive_canonical_ownership_is_copied(
    qapp_fixture, kind
):
    editor = _editor(qapp_fixture)
    gray = np.zeros((2, 2), dtype=np.float32)
    backing = np.zeros((2, 4, 4), dtype=np.uint8)
    if kind == "mutable":
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    elif kind == "borrowed":
        rgba = backing[:, :2, :]
        rgba.setflags(write=False)
    else:
        rgba = backing[:, ::2, :]
        rgba.setflags(write=False)
    rgb = np.array(rgba[..., :3], copy=True)

    editor.load_array(gray, rgb, rgba)

    assert editor._base_rgba is not rgba
    assert editor._base_rgba.flags.owndata
    assert editor._base_rgba.flags.c_contiguous
    assert not editor._base_rgba.flags.writeable
