import numpy as np
import pytest
from ditherzam.color.palette import Palette


def test_from_list_shape_and_dtype():
    p = Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]])
    assert p.name == "duo"
    assert p.colors.shape == (2, 3)
    assert p.colors.dtype == np.float32


def test_from_list_values_preserved():
    p = Palette.from_list("t", [[10, 20, 30], [40, 50, 60]])
    np.testing.assert_array_equal(p.colors, np.array([[10, 20, 30], [40, 50, 60]], np.float32))


def test_roundtrip_yaml(tmp_path):
    p = Palette.from_list("mypal", [[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    f = tmp_path / "mypal.yaml"
    p.to_yaml(f)
    assert f.is_file()
    q = Palette.load(f)
    assert q.name == "mypal"
    assert q.colors.dtype == np.float32
    np.testing.assert_array_equal(q.colors, p.colors)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Palette.load(tmp_path / "nope.yaml")
