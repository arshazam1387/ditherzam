def test_prereqs_import():
    from ditherzam.render import RenderSettings, RenderPipeline
    from ditherzam.color.palette import Palette
    from ditherzam.effects.stack import EffectStack
    s = RenderSettings()
    assert s.contrast == 50 and s.scale == 5 and s.style == "None"
    assert s.saturation == 50 and s.invert is False
    assert isinstance(s.params, dict)


import numpy as np
import pytest
from ditherzam.render import RenderSettings
from ditherzam.presets import settings_to_preset, preset_to_settings


def test_roundtrip_preserves_values():
    s = RenderSettings(contrast=70, midtones=30, highlights=60,
                       luminance_threshold=40, blur=10, saturation=80,
                       invert=True, style="Atkinson", scale=3,
                       preview_disabled=True, params={"dither_parameter": 2})
    d = settings_to_preset(s)
    s2, pal, fx = preset_to_settings(d)
    assert s2.contrast == 70 and s2.midtones == 30 and s2.highlights == 60
    assert s2.luminance_threshold == 40 and s2.blur == 10 and s2.saturation == 80
    assert s2.invert is True and s2.style == "Atkinson" and s2.scale == 3
    assert s2.preview_disabled is True
    assert s2.params == {"dither_parameter": 2}
    assert pal is None and fx == []


def test_roundtrip_clamps_out_of_range():
    s = RenderSettings(contrast=70, style="Atkinson", scale=3)
    d = settings_to_preset(s)
    d["adjustments"]["contrast"] = 9999          # out of range high
    d["adjustments"]["blur"] = -50               # out of range low
    d["dither"]["scale"] = 0                      # below 1
    s2, pal, fx = preset_to_settings(d)
    assert 0 <= s2.contrast <= 100 and s2.contrast == 100
    assert 0 <= s2.blur <= 100 and s2.blur == 0
    assert 1 <= s2.scale <= 20 and s2.scale == 1
    assert s2.style == "Atkinson"


def test_missing_sections_fall_back_to_defaults():
    s2, pal, fx = preset_to_settings({"dither": {"style": "Floyd-Steinberg"}})
    d = RenderSettings()
    assert s2.contrast == d.contrast and s2.saturation == d.saturation
    assert s2.style == "Floyd-Steinberg" and s2.scale == d.scale


def test_non_mapping_preset_raises():
    with pytest.raises(ValueError):
        preset_to_settings("just a string")


def test_palette_and_effects_roundtrip():
    from ditherzam.color.palette import Palette
    from ditherzam.effects.stack import EffectStack
    pal = Palette(name="mini", colors=np.array([[0, 0, 0], [255, 255, 255]], np.float32))
    stack = EffectStack()
    stack.add("Blur", radius=2)
    stack.add("Sharpen", amount=1)
    d = settings_to_preset(RenderSettings(), palette=pal, effect_stack=stack, color_mode="nearest")
    assert d["color"]["mode"] == "nearest"
    assert d["color"]["palette"]["name"] == "mini"
    assert d["color"]["palette"]["colors"] == [[0, 0, 0], [255, 255, 255]]
    assert d["effects"] == [{"name": "Blur", "params": {"radius": 2}},
                            {"name": "Sharpen", "params": {"amount": 1}}]
    s2, pal2, fx = preset_to_settings(d)
    assert pal2 is not None and pal2.name == "mini"
    assert pal2.colors.shape == (2, 3) and pal2.colors.dtype == np.float32
    assert fx == [("Blur", {"radius": 2}), ("Sharpen", {"amount": 1})]


from ditherzam.presets import PresetManager


def test_preset_manager_save_list_load(tmp_path):
    m = PresetManager(tmp_path)
    p = m.save("mine", {"adjustments": {}, "dither": {"style": "None"}})
    assert p.exists() and p.suffix == ".yaml"
    assert "mine" in m.list()
    assert m.load("mine")["dither"]["style"] == "None"


def test_preset_manager_list_is_sorted(tmp_path):
    m = PresetManager(tmp_path)
    m.save("zeta", {"dither": {}})
    m.save("alpha", {"dither": {}})
    assert m.list() == ["alpha", "zeta"]


def test_preset_manager_delete(tmp_path):
    m = PresetManager(tmp_path)
    m.save("temp", {"dither": {}})
    assert m.delete("temp") is True
    assert "temp" not in m.list()
    assert m.delete("temp") is False          # already gone


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PresetManager(tmp_path).load("nope")


def test_import_valid_yaml_returns_name(tmp_path):
    src = tmp_path / "cool.yaml"
    src.write_text("dither:\n  style: Atkinson\n", encoding="utf-8")
    m = PresetManager(tmp_path / "store")
    name = m.import_file(src)
    assert name == "cool"
    assert m.load("cool")["dither"]["style"] == "Atkinson"


def test_import_invalid_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("just a string", encoding="utf-8")
    with pytest.raises(ValueError):
        PresetManager(tmp_path / "store2").import_file(bad)


def test_import_wrong_extension_raises(tmp_path):
    bad = tmp_path / "notpreset.txt"
    bad.write_text("dither: {}", encoding="utf-8")
    with pytest.raises(ValueError):
        PresetManager(tmp_path / "store3").import_file(bad)


def test_preset_roundtrips_palette_category():
    from ditherzam.color.palette import Palette
    pal = Palette.from_list("mine", [[1, 2, 3], [4, 5, 6]], category="retro")
    preset = settings_to_preset(RenderSettings(), pal, None, "ramp")
    _settings, out_pal, _effects = preset_to_settings(preset)
    assert out_pal.category == "retro"
