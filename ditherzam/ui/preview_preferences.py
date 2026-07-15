"""Persistent application preferences for interactive preview quality.

The normalization helpers are intentionally Qt-free.  Persistence only relies
on the small ``value``/``setValue`` interface exposed by ``QSettings`` so tests
and non-widget callers do not need to construct Qt objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


PREVIEW_RESOLUTIONS = ("Auto", "480", "720", "1080", "1440", "2160", "Full")
DEFAULT_PREVIEW_RESOLUTION = "Auto"
DEFAULT_RERENDER_ON_ZOOM = False

RESOLUTION_KEY = "preview/resolution"
RERENDER_ON_ZOOM_KEY = "preview/rerender_on_zoom"

_NUMERIC_RESOLUTIONS = frozenset(PREVIEW_RESOLUTIONS[1:-1])


class SettingsReader(Protocol):
    def value(self, key: str, defaultValue: Any = None) -> Any: ...


class SettingsWriter(Protocol):
    def setValue(self, key: str, value: Any) -> None: ...


def normalize_preview_resolution(value: Any) -> str:
    """Return a canonical resolution choice, defaulting corrupt input to Auto."""
    if isinstance(value, bool):
        return DEFAULT_PREVIEW_RESOLUTION

    if isinstance(value, int):
        candidate = str(value)
    elif isinstance(value, float):
        if not value.is_integer():
            return DEFAULT_PREVIEW_RESOLUTION
        candidate = str(int(value))
    elif isinstance(value, str):
        candidate = value.strip()
    else:
        return DEFAULT_PREVIEW_RESOLUTION

    lowered = candidate.casefold()
    if lowered == "auto":
        return "Auto"
    if lowered == "full":
        return "Full"
    if candidate in _NUMERIC_RESOLUTIONS:
        return candidate
    return DEFAULT_PREVIEW_RESOLUTION


def normalize_rerender_on_zoom(value: Any) -> bool:
    """Normalize common QSettings boolean representations; corrupt means Off."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return DEFAULT_RERENDER_ON_ZOOM


@dataclass(frozen=True)
class PreviewPreferences:
    resolution: str = DEFAULT_PREVIEW_RESOLUTION
    rerender_on_zoom: bool = DEFAULT_RERENDER_ON_ZOOM

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "resolution", normalize_preview_resolution(self.resolution)
        )
        object.__setattr__(
            self,
            "rerender_on_zoom",
            normalize_rerender_on_zoom(self.rerender_on_zoom),
        )


def load_preview_preferences(settings: SettingsReader) -> PreviewPreferences:
    """Load and normalize preview preferences from a QSettings-like object."""
    return PreviewPreferences(
        resolution=settings.value(RESOLUTION_KEY, DEFAULT_PREVIEW_RESOLUTION),
        rerender_on_zoom=settings.value(
            RERENDER_ON_ZOOM_KEY, DEFAULT_RERENDER_ON_ZOOM
        ),
    )


def save_preview_preferences(
    settings: SettingsWriter, preferences: PreviewPreferences
) -> None:
    """Write canonical values to a QSettings-like object."""
    normalized = PreviewPreferences(
        preferences.resolution, preferences.rerender_on_zoom
    )
    settings.setValue(RESOLUTION_KEY, normalized.resolution)
    settings.setValue(RERENDER_ON_ZOOM_KEY, normalized.rerender_on_zoom)
