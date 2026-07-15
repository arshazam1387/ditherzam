"""Tests for ditherzam.masking.settings: MaskTarget, OutsideMode, SmartMaskSettings.

Exact defaults are recorded verbatim from the approved design spec's state
table: disabled, Subject, sensitivity 50, feather 8, expansion 0, invert
false, Original.
"""
from __future__ import annotations

import dataclasses

import pytest

from ditherzam.masking.settings import (
    EXPANSION_MAX_PX,
    EXPANSION_MIN_PX,
    FEATHER_MIN_PX,
    MaskSettingsError,
    MaskTarget,
    OutsideMode,
    SENSITIVITY_MAX,
    SENSITIVITY_MIN,
    SmartMaskSettings,
)


# -- Enums -----------------------------------------------------------------


def test_mask_target_members():
    assert {t.name for t in MaskTarget} == {"SUBJECT", "BACKGROUND", "WHOLE_IMAGE"}


def test_outside_mode_members():
    assert {m.name for m in OutsideMode} == {"ORIGINAL", "TRANSPARENT", "WHITE", "BLACK"}


def test_enum_members_are_distinct():
    assert len({MaskTarget.SUBJECT, MaskTarget.BACKGROUND, MaskTarget.WHOLE_IMAGE}) == 3
    assert len({OutsideMode.ORIGINAL, OutsideMode.TRANSPARENT, OutsideMode.WHITE, OutsideMode.BLACK}) == 4


# -- Exact defaults ----------------------------------------------------------


def test_exact_defaults():
    settings = SmartMaskSettings()
    assert settings.enabled is False
    assert settings.target is MaskTarget.SUBJECT
    assert settings.sensitivity == 50
    assert settings.feather_px == 8
    assert settings.expansion_px == 0
    assert settings.invert is False
    assert settings.outside is OutsideMode.ORIGINAL


def test_range_constants_match_spec():
    assert (SENSITIVITY_MIN, SENSITIVITY_MAX) == (0, 100)
    assert FEATHER_MIN_PX == 0
    assert (EXPANSION_MIN_PX, EXPANSION_MAX_PX) == (-64, 64)


# -- Frozen / immutable --------------------------------------------------


def test_settings_are_frozen():
    settings = SmartMaskSettings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.sensitivity = 10  # type: ignore[misc]


# -- Equality / hash stability -----------------------------------------------


def test_equal_settings_are_equal_and_hash_equal():
    a = SmartMaskSettings(sensitivity=70, feather_px=4)
    b = SmartMaskSettings(sensitivity=70, feather_px=4)
    assert a == b
    assert hash(a) == hash(b)


def test_different_settings_are_not_equal():
    a = SmartMaskSettings(sensitivity=70)
    b = SmartMaskSettings(sensitivity=71)
    assert a != b


# -- Range / type validation -------------------------------------------------


@pytest.mark.parametrize("sensitivity", [0, 50, 100])
def test_sensitivity_boundaries_accepted(sensitivity):
    assert SmartMaskSettings(sensitivity=sensitivity).sensitivity == sensitivity


@pytest.mark.parametrize("sensitivity", [-1, 101, -1000, 1000])
def test_sensitivity_out_of_range_rejected(sensitivity):
    with pytest.raises(MaskSettingsError):
        SmartMaskSettings(sensitivity=sensitivity)


@pytest.mark.parametrize("expansion_px", [-64, 0, 64])
def test_expansion_boundaries_accepted(expansion_px):
    assert SmartMaskSettings(expansion_px=expansion_px).expansion_px == expansion_px


@pytest.mark.parametrize("expansion_px", [-65, 65, -1000, 1000])
def test_expansion_out_of_range_rejected(expansion_px):
    with pytest.raises(MaskSettingsError):
        SmartMaskSettings(expansion_px=expansion_px)


def test_feather_zero_accepted():
    assert SmartMaskSettings(feather_px=0).feather_px == 0


def test_negative_feather_rejected():
    with pytest.raises(MaskSettingsError):
        SmartMaskSettings(feather_px=-1)


def test_non_bool_enabled_rejected():
    with pytest.raises(MaskSettingsError):
        SmartMaskSettings(enabled=1)  # type: ignore[arg-type]


def test_non_bool_invert_rejected():
    with pytest.raises(MaskSettingsError):
        SmartMaskSettings(invert="yes")  # type: ignore[arg-type]


def test_wrong_type_target_rejected():
    with pytest.raises(MaskSettingsError):
        SmartMaskSettings(target="subject")  # type: ignore[arg-type]


def test_wrong_type_outside_rejected():
    with pytest.raises(MaskSettingsError):
        SmartMaskSettings(outside="original")  # type: ignore[arg-type]


def test_non_int_sensitivity_rejected():
    with pytest.raises(MaskSettingsError):
        SmartMaskSettings(sensitivity=50.5)  # type: ignore[arg-type]


# -- No Qt imports ------------------------------------------------------------


def test_settings_module_has_no_qt_import():
    import ditherzam.masking.settings as mod

    source = mod.__file__
    from pathlib import Path

    text = Path(source).read_text(encoding="utf-8")
    assert "PySide6" not in text
    assert "PyQt" not in text
