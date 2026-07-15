import numpy as np
from ditherzam.dithering import registry

def test_all_three_registered():
    for n in ("Floyd-Steinberg", "Atkinson", "Bayer-Matrix 4x4"):
        assert registry.get_entry(n) is not None

def test_output_is_binary_0_255():
    reg = registry
    img = np.tile(np.linspace(0, 255, 8, dtype=np.float32), (8, 1))
    out = reg.get_entry("Floyd-Steinberg").func(img.copy(), 0, 128.0)
    vals = set(np.unique(out).tolist())
    assert vals <= {0.0, 255.0}
    assert out.shape == img.shape

def test_all_black_input_stays_black():
    img = np.zeros((8, 8), dtype=np.float32)
    out = registry.get_entry("Atkinson").func(img, 0, 128.0)
    assert out.sum() == 0.0

def test_all_white_input_stays_white():
    img = np.full((8, 8), 255.0, dtype=np.float32)
    out = registry.get_entry("Bayer-Matrix 4x4").func(img, 0, 128.0)
    assert np.all(out == 255.0)
