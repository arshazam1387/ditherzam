from ditherzam.ui.settings_map import settings_from_controls
from ditherzam.render import RenderSettings
from ditherzam.presets import settings_to_preset, preset_to_settings


def test_settings_map_reads_depth_and_mapping():
    s = settings_from_controls({"depth": 12, "color_mapping": "glitch"})
    assert s.depth == 12 and s.color_mapping == "glitch"


def test_settings_map_defaults():
    s = settings_from_controls({})
    assert s.depth == 2 and s.color_mapping == "match"


def test_preset_round_trip_depth_mapping():
    s = RenderSettings(depth=9, color_mapping="hue_cycle")
    back, _palette, _effects = preset_to_settings(settings_to_preset(s))
    assert back.depth == 9 and back.color_mapping == "hue_cycle"
