from pathlib import Path
import pytest
from ditherzam.ui.theme import find_themes, load_theme, ThemeData

ROOT = Path("themes")


def test_find_themes_lists_default():
    themes = find_themes(ROOT)
    assert "default" in themes


def test_find_themes_missing_root_is_empty():
    assert find_themes(Path("no/such/dir")) == []


def test_load_default_theme_qss_and_glow():
    td = load_theme(ROOT, "default")
    assert isinstance(td, ThemeData)
    assert td.name == "default"
    assert "#1f1f1f" in td.stylesheet          # panel background
    assert "QSlider::sub-page:horizontal" in td.stylesheet
    assert td.glow_color == "#5e89ed"
    assert td.labels.get("Contrast") is True


def test_load_missing_theme_raises():
    with pytest.raises(FileNotFoundError):
        load_theme(ROOT, "does_not_exist")
