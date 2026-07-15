from ditherzam.render import RenderSettings
from ditherzam.ui.settings_map import settings_from_controls, build_dither_rows


def test_maps_dict_to_render_settings():
    state = {
        "contrast": 70, "midtones": 40, "highlights": 55,
        "blur": 10, "luminance_threshold": 60, "saturation": 80,
        "invert": True, "style": "Atkinson", "scale": 3,
        "preview_disabled": False, "params": {"Line Count": 4},
    }
    s = settings_from_controls(state)
    assert isinstance(s, RenderSettings)
    assert s.contrast == 70 and s.midtones == 40 and s.highlights == 55
    assert s.blur == 10 and s.luminance_threshold == 60 and s.saturation == 80
    assert s.invert is True
    assert s.style == "Atkinson" and s.scale == 3
    assert s.preview_disabled is False
    assert s.params == {"Line Count": 4}


def test_defaults_fill_missing_keys():
    s = settings_from_controls({})
    assert s.contrast == 50 and s.midtones == 50 and s.highlights == 50
    assert s.blur == 50 and s.luminance_threshold == 50 and s.saturation == 50
    assert s.style == "None" and s.scale == 5
    assert s.invert is False and s.preview_disabled is False
    assert s.params == {}


def test_params_is_copied_not_aliased():
    src = {"style": "Glitch", "params": {"Glitch Intensity": 5}}
    s = settings_from_controls(src)
    s.params["Glitch Intensity"] = 999
    assert src["params"]["Glitch Intensity"] == 5


def test_build_dither_rows_headers_and_items():
    by_cat = {
        "Default": ["None"],
        "Error Diffusion": ["Floyd-Steinberg", "Atkinson"],
    }
    rows = build_dither_rows(by_cat)
    assert rows == [
        (True, "Default", None),
        (False, "None", "None"),
        (True, "Error Diffusion", None),
        (False, "Floyd-Steinberg", "Floyd-Steinberg"),
        (False, "Atkinson", "Atkinson"),
    ]
