from ditherzam.masking.settings import MaskTarget, OutsideMode, SmartMaskSettings
from ditherzam.presets import preset_to_settings, settings_to_preset
from ditherzam.render import RenderSettings


def test_exactly_eight_reusable_mask_settings_round_trip():
    mask = SmartMaskSettings(True, MaskTarget.BACKGROUND, 73, 19, -7, True,
                             OutsideMode.WHITE, True)
    preset = settings_to_preset(RenderSettings(), smart_mask=mask)
    assert preset["smart_mask"] == {
        "enabled": True, "target": "background", "sensitivity": 73,
        "feather_px": 19, "expansion_px": -7, "invert": True,
        "outside": "white", "bake_fill": True,
    }
    assert preset_to_settings(preset).smart_mask == mask


def test_old_and_malformed_mask_presets_load_safe_defaults_and_clamp():
    assert preset_to_settings({}).smart_mask == SmartMaskSettings()
    contents = preset_to_settings({"smart_mask": {
        "enabled": "yes", "target": "bogus", "sensitivity": 999,
        "feather_px": -2, "expansion_px": -999, "invert": 1,
        "outside": "bogus",
    }})
    assert contents.smart_mask == SmartMaskSettings(sensitivity=100, feather_px=0,
                                                    expansion_px=-64)


def test_nonfinite_and_bad_mask_numbers_use_field_defaults():
    for bad in (float("nan"), float("inf"), float("-inf"), "bad", None):
        contents = preset_to_settings({"smart_mask": {
            "sensitivity": bad, "feather_px": bad, "expansion_px": bad,
        }})
        assert contents.smart_mask == SmartMaskSettings()


def test_session_and_asset_fields_are_never_persisted():
    preset = settings_to_preset(RenderSettings(), smart_mask=SmartMaskSettings(True))
    forbidden = {"array", "source", "candidate", "overlay", "progress", "error",
                 "model_path", "model_hash", "probability", "mask"}
    assert forbidden.isdisjoint(preset["smart_mask"])


def test_legacy_three_value_unpacking_is_preserved():
    settings, palette, effects = preset_to_settings({})
    assert isinstance(settings, RenderSettings)
    assert palette is None and effects == []
