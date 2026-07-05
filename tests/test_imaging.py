import numpy as np
from PIL import Image
from ditherzam.imaging import to_gray_f32, clamp_u8, nearest_downscale, nearest_upscale_to

def test_to_gray_f32_from_rgb():
    img = Image.new("RGB", (4, 2), (255, 255, 255))
    g = to_gray_f32(img)
    assert g.shape == (2, 4) and g.dtype == np.float32
    assert np.allclose(g, 255.0)

def test_clamp_u8():
    a = np.array([[-5.0, 128.0, 300.0]], dtype=np.float32)
    assert clamp_u8(a).tolist() == [[0, 128, 255]]

def test_downscale_then_upscale_roundtrips_size():
    g = np.arange(64, dtype=np.float32).reshape(8, 8)
    small = nearest_downscale(g, 4)          # 8//4 = 2
    assert small.shape == (2, 2)
    back = nearest_upscale_to(small, (8, 8))
    assert back.shape == (8, 8)
