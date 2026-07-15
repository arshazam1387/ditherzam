from ditherzam.presets import settings_to_preset
from ditherzam.render import RenderSettings
from ditherzam.ui.preview_preferences import (
    PreviewPreferences,
    load_preview_preferences,
    normalize_preview_resolution,
    normalize_rerender_on_zoom,
    save_preview_preferences,
)


class FakeSettings:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def value(self, key, defaultValue=None):
        return self.values.get(key, defaultValue)

    def setValue(self, key, value):
        self.values[key] = value


def test_defaults_are_auto_and_zoom_rerender_off():
    assert load_preview_preferences(FakeSettings()) == PreviewPreferences()
    assert PreviewPreferences().resolution == "Auto"
    assert PreviewPreferences().rerender_on_zoom is False


def test_resolution_normalizes_supported_and_corrupt_values():
    expected = {
        "auto": "Auto",
        " AUTO ": "Auto",
        480: "480",
        "720": "720",
        1080.0: "1080",
        "1440": "1440",
        2160: "2160",
        " full ": "Full",
    }
    for raw, canonical in expected.items():
        assert normalize_preview_resolution(raw) == canonical

    for raw in (None, "", "480p", 481, 720.5, True, object()):
        assert normalize_preview_resolution(raw) == "Auto"


def test_zoom_flag_normalizes_qsettings_shaped_values():
    for raw in (True, 1, "1", "true", "TRUE", "yes", "on"):
        assert normalize_rerender_on_zoom(raw) is True
    for raw in (False, 0, "0", "false", "FALSE", "no", "off", None, "bad", 2):
        assert normalize_rerender_on_zoom(raw) is False


def test_load_normalizes_corrupt_stored_values():
    store = FakeSettings({
        "preview/resolution": "gigantic",
        "preview/rerender_on_zoom": "yes",
    })
    assert load_preview_preferences(store) == PreviewPreferences("Auto", True)


def test_save_writes_canonical_qsettings_values_and_round_trips():
    store = FakeSettings()
    save_preview_preferences(store, PreviewPreferences(" 1080 ", True))

    assert store.values == {
        "preview/resolution": "1080",
        "preview/rerender_on_zoom": True,
    }
    assert load_preview_preferences(store) == PreviewPreferences("1080", True)


def test_real_qsettings_round_trip(qapp_fixture, tmp_path):
    from PySide6.QtCore import QSettings

    store = QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat)
    save_preview_preferences(store, PreviewPreferences("Full", True))
    store.sync()

    reopened = QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat)
    assert load_preview_preferences(reopened) == PreviewPreferences("Full", True)


def test_preview_preferences_are_not_creative_preset_fields():
    preset = settings_to_preset(RenderSettings())
    serialized = repr(preset).lower()

    assert "rerender_on_zoom" not in serialized
    assert "preview/resolution" not in serialized
    assert "preview_resolution" not in serialized
