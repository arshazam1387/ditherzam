import numpy as np


def test_normal_half_opacity_source_over():
    from ditherzam.layers import blend_layer
    back = np.array([[[0, 0, 255, 255]]], np.uint8)
    src = np.array([[[255, 0, 0, 255]]], np.uint8)
    assert np.array_equal(blend_layer(back, src, "normal", 50),
                          np.array([[[128, 0, 127, 255]]], np.uint8))


def test_blend_modes_are_distinct():
    from ditherzam.layers import blend_layer
    back = np.array([[[80, 120, 200, 255]]], np.uint8)
    src = np.array([[[200, 100, 40, 255]]], np.uint8)
    results = {m: blend_layer(back, src, m, 100).tobytes()
               for m in ("normal", "multiply", "screen", "overlay", "difference")}
    assert len(set(results.values())) == 5
