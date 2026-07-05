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
