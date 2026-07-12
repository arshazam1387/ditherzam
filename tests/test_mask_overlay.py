import numpy as np

from ditherzam.ui.mask_overlay import apply_mask_overlay


def test_overlay_is_display_copy_and_preserves_alpha():
    image = np.zeros((2, 2, 4), np.uint8); image[..., 3] = 91
    mask = np.array([[0, 1], [0.5, 0]], np.float32); mask.flags.writeable = False
    before = image.copy()
    out = apply_mask_overlay(image, mask)
    assert np.array_equal(image, before)
    assert out is not image and np.array_equal(out[..., 3], image[..., 3])
    assert np.array_equal(out[0, 0], image[0, 0])
    assert not np.array_equal(out[0, 1, :3], image[0, 1, :3])
