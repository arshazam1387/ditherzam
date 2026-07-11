import numpy as np
import pytest
from ditherzam.color.palette import Palette, builtin_palettes, extract_palette, source_palette


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


def test_builtins_all_present():
    b = builtin_palettes()
    for name in ("grayscale", "gameboy", "cga", "pico8", "sepia"):
        assert name in b, f"missing built-in palette: {name}"


def test_builtin_counts_and_shape():
    b = builtin_palettes()
    assert b["grayscale"].colors.shape == (4, 3)
    assert b["gameboy"].colors.shape == (4, 3)
    assert b["cga"].colors.shape == (16, 3)
    assert b["pico8"].colors.shape == (16, 3)
    assert b["sepia"].colors.shape == (4, 3)
    for p in b.values():
        assert p.colors.dtype == np.float32


def test_builtin_exact_values():
    b = builtin_palettes()
    np.testing.assert_array_equal(
        b["gameboy"].colors,
        np.array([[15, 56, 15], [48, 98, 48], [139, 172, 15], [155, 188, 15]], np.float32),
    )
    # PICO-8 index 8 is the signature red (#FF004D)
    np.testing.assert_array_equal(b["pico8"].colors[8], np.array([255, 0, 77], np.float32))
    # CGA index 14 is yellow
    np.testing.assert_array_equal(b["cga"].colors[14], np.array([255, 255, 85], np.float32))


def test_added_builtin_palettes_present_and_valid():
    b = builtin_palettes()
    added = {"c64": "retro", "zxspectrum": "retro", "nord": "cool",
             "solarized": "cool", "greencrt": "mono", "ambercrt": "mono"}
    for name, category in added.items():
        assert name in b, f"missing added palette: {name}"
        p = b[name]
        assert p.category == category
        assert p.colors.ndim == 2 and p.colors.shape[1] == 3
        assert p.colors.shape[0] >= 2
        assert p.colors.dtype == np.float32
        assert float(p.colors.min()) >= 0.0 and float(p.colors.max()) <= 255.0


def test_twenty_curated_builtin_palettes_are_valid_and_tonally_useful():
    b = builtin_palettes()
    added = {
        "midnight_bloom": "cinematic", "desert_film": "cinematic",
        "noir_teal": "cinematic", "velvet_gold": "cinematic",
        "forest_mist": "nature", "ocean_glass": "nature",
        "alpine_lake": "nature", "autumn_earth": "nature",
        "lavender_milk": "pastel", "peach_sorbet": "pastel",
        "mint_cloud": "pastel", "berry_cream": "pastel",
        "electric_night": "neon", "laser_lime": "neon",
        "ultraviolet": "neon", "arctic_signal": "cool",
        "blue_hour": "cool", "coral_sunset": "warm",
        "honey_ink": "warm", "rosewood": "warm",
    }
    assert added.keys() <= b.keys()
    for name, category in added.items():
        palette = b[name]
        assert palette.category == category
        assert palette.colors.shape == (6, 3)
        luminance = palette.colors @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        assert float(luminance.max() - luminance.min()) >= 120.0


def test_extract_returns_k_colors():
    img = np.random.RandomState(0).randint(0, 256, (32, 32, 3), dtype=np.uint8)
    p = extract_palette(img, k=8)
    assert p.colors.shape == (8, 3)
    assert p.colors.dtype == np.float32
    assert p.name == "source"


def test_extract_default_name_and_k():
    img = np.random.RandomState(3).randint(0, 256, (16, 16, 3), dtype=np.uint8)
    p = extract_palette(img)
    assert p.colors.shape == (16, 3)


def test_extract_two_color_image():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:, :5] = [255, 0, 0]
    img[:, 5:] = [0, 0, 255]
    p = extract_palette(img, k=2)
    got = [tuple(int(round(v)) for v in c) for c in p.colors]
    reds = [c for c in got if c[0] > 200 and c[2] < 55]
    blues = [c for c in got if c[2] > 200 and c[0] < 55]
    assert reds and blues


def test_extract_k_larger_than_unique_colors():
    img = np.zeros((8, 8, 3), np.uint8)
    img[:, :4] = [10, 10, 10]
    img[:, 4:] = [200, 200, 200]
    p = extract_palette(img, k=4)
    # still returns exactly k rows even when the image has < k distinct colors
    assert p.colors.shape == (4, 3)


def test_source_complete_is_256():
    img = np.random.RandomState(5).randint(0, 256, (48, 48, 3), dtype=np.uint8)
    p = source_palette(img, completeness=1.0)
    assert p.colors.shape == (256, 3)


def test_source_minimal_is_2():
    img = np.random.RandomState(6).randint(0, 256, (48, 48, 3), dtype=np.uint8)
    p = source_palette(img, completeness=0.0)
    assert p.colors.shape == (2, 3)


def test_source_midpoint_between_bounds():
    img = np.random.RandomState(7).randint(0, 256, (48, 48, 3), dtype=np.uint8)
    p = source_palette(img, completeness=0.5)
    k = p.colors.shape[0]
    assert 2 < k < 256


def test_source_clamps_out_of_range():
    img = np.random.RandomState(8).randint(0, 256, (24, 24, 3), dtype=np.uint8)
    assert source_palette(img, completeness=5.0).colors.shape == (256, 3)
    assert source_palette(img, completeness=-1.0).colors.shape == (2, 3)


def test_shuffle_keeps_locked_and_shape():
    p = Palette.from_list("t", [[0, 0, 0], [64, 64, 64], [128, 128, 128], [255, 255, 255]])
    rng = np.random.default_rng(0)
    q = p.shuffle(locked={0, 3}, rng=rng)
    assert q.colors.shape == (4, 3)
    assert q.name == "t"
    # locked rows identical
    np.testing.assert_array_equal(q.colors[0], p.colors[0])
    np.testing.assert_array_equal(q.colors[3], p.colors[3])


def test_shuffle_changes_unlocked():
    p = Palette.from_list("t", [[0, 0, 0], [64, 64, 64], [128, 128, 128], [255, 255, 255]])
    rng = np.random.default_rng(1)
    q = p.shuffle(locked={0}, rng=rng)
    # at least one unlocked row differs from the original
    changed = any(not np.array_equal(q.colors[i], p.colors[i]) for i in (1, 2, 3))
    assert changed


def test_shuffle_is_immutable_on_original():
    p = Palette.from_list("t", [[0, 0, 0], [64, 64, 64]])
    before = p.colors.copy()
    p.shuffle(locked=set(), rng=np.random.default_rng(2))
    np.testing.assert_array_equal(p.colors, before)


def test_shuffle_values_in_range():
    p = Palette.from_list("t", [[0, 0, 0], [64, 64, 64], [128, 128, 128]])
    q = p.shuffle(locked=set(), rng=np.random.default_rng(3))
    assert q.colors.min() >= 0 and q.colors.max() <= 255


def test_generate_palette_k_gives_exact_count():
    from ditherzam.color.palette import generate_palette
    img = np.random.default_rng(0).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    p = generate_palette(img, "k", 8)
    assert p.colors.shape == (8, 3)
    assert p.name == "from image"


def test_generate_palette_pct_maps_to_source_palette():
    from ditherzam.color.palette import generate_palette, source_palette
    img = np.random.default_rng(1).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    got = generate_palette(img, "pct", 50)
    expect = source_palette(img, completeness=0.5)
    assert got.colors.shape == expect.colors.shape


def test_generate_palette_bad_unit_raises():
    from ditherzam.color.palette import generate_palette
    img = np.zeros((4, 4, 3), np.uint8)
    with pytest.raises(ValueError):
        generate_palette(img, "nonsense", 4)


def test_category_defaults_empty():
    from ditherzam.color.palette import Palette
    p = Palette.from_list("x", [[1, 2, 3]])
    assert p.category == ""


def test_category_yaml_roundtrip(tmp_path):
    from ditherzam.color.palette import Palette
    p = Palette.from_list("x", [[1, 2, 3]], category="retro")
    dest = tmp_path / "x.yaml"
    p.to_yaml(dest)
    assert "category: retro" in dest.read_text(encoding="utf-8")
    assert Palette.load(dest).category == "retro"


def test_empty_category_not_written(tmp_path):
    from ditherzam.color.palette import Palette
    p = Palette.from_list("x", [[1, 2, 3]])
    dest = tmp_path / "x.yaml"
    p.to_yaml(dest)
    assert "category" not in dest.read_text(encoding="utf-8")


def test_load_legacy_yaml_has_empty_category(tmp_path):
    from ditherzam.color.palette import Palette
    dest = tmp_path / "legacy.yaml"
    dest.write_text("name: legacy\ncolors:\n  - [1, 2, 3]\n", encoding="utf-8")
    assert Palette.load(dest).category == ""


def test_shuffle_preserves_category():
    import numpy as np
    from ditherzam.color.palette import Palette
    p = Palette.from_list("x", [[1, 2, 3], [4, 5, 6]], category="retro")
    out = p.shuffle(locked={0}, rng=np.random.default_rng(0))
    assert out.category == "retro"


def test_extract_palette_is_user_category():
    import numpy as np
    from ditherzam.color.palette import extract_palette
    rgb = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    assert extract_palette(rgb, k=4).category == "user"
