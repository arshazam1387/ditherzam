import pytest

pytest.importorskip("PySide6")


class FakeSettings:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def value(self, key, defaultValue=None):
        return self.values.get(key, defaultValue)

    def setValue(self, key, value):
        self.values[key] = value


def test_view_menu_loads_exclusive_preview_resolution(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor

    store = FakeSettings({"preview/resolution": "1080"})
    editor = ImageEditor(preference_store=store)

    assert editor.preview_resolution_menu.title() == "Preview Resolution"
    assert tuple(editor.preview_resolution_actions) == (
        "Auto", "480", "720", "1080", "1440", "2160", "Full"
    )
    assert editor.preview_resolution_group.isExclusive()
    assert editor.preview_resolution_actions["1080"].isChecked()
    assert sum(a.isChecked() for a in editor.preview_resolution_actions.values()) == 1


def test_resolution_choice_persists_and_schedules(qapp_fixture, monkeypatch):
    from ditherzam.ui.main_window import ImageEditor

    store = FakeSettings()
    editor = ImageEditor(preference_store=store)
    scheduled = []
    monkeypatch.setattr(editor, "schedule_render", lambda: scheduled.append(True))
    editor._full_preview_requested = True

    editor.preview_resolution_actions["1440"].trigger()

    assert store.values["preview/resolution"] == "1440"
    assert editor.preview_preferences.resolution == "1440"
    assert editor._full_preview_requested is False
    assert scheduled == [True]


def test_zoom_rerender_toggle_loads_and_persists(qapp_fixture, monkeypatch):
    from ditherzam.ui.main_window import ImageEditor

    store = FakeSettings({"preview/rerender_on_zoom": "true"})
    editor = ImageEditor(preference_store=store)
    assert editor.rerender_on_zoom_action.isCheckable()
    assert editor.rerender_on_zoom_action.isChecked()

    scheduled = []
    monkeypatch.setattr(editor, "schedule_render", lambda: scheduled.append(True))
    editor._full_preview_requested = True
    editor.rerender_on_zoom_action.trigger()

    assert store.values["preview/rerender_on_zoom"] is False
    assert editor.preview_preferences.rerender_on_zoom is False
    assert editor._full_preview_requested is False
    assert scheduled == [True]


def test_corrupt_preference_is_reflected_as_safe_defaults(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor(preference_store=FakeSettings({
        "preview/resolution": "potato",
        "preview/rerender_on_zoom": "potato",
    }))

    assert editor.preview_resolution_actions["Auto"].isChecked()
    assert not editor.rerender_on_zoom_action.isChecked()
