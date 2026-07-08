from __future__ import annotations

from ditherzam.render import RenderSettings


def settings_from_controls(state: dict) -> RenderSettings:
    """Build a RenderSettings from a plain dict of control values.

    Every field falls back to its spec default so a partial dict is valid; params
    is deep-copied so later mutation of the RenderSettings never leaks back.
    """
    return RenderSettings(
        contrast=int(state.get("contrast", 50)),
        midtones=int(state.get("midtones", 50)),
        highlights=int(state.get("highlights", 50)),
        blur=int(state.get("blur", 50)),
        luminance_threshold=int(state.get("luminance_threshold", 50)),
        invert=bool(state.get("invert", False)),
        saturation=int(state.get("saturation", 50)),
        style=state.get("style", "None"),
        scale=int(state.get("scale", 5)),
        depth=int(state.get("depth", 2)),
        color_mapping=state.get("color_mapping", "match"),
        preview_disabled=bool(state.get("preview_disabled", False)),
        params=dict(state.get("params", {}) or {}),
    )


def build_dither_rows(by_category):
    """Flatten a {category: [style, ...]} map into ordered combo rows.

    Each row is (is_header, label, style_or_None). Header rows carry None style and
    are rendered non-selectable by the delegate.
    """
    rows: list[tuple[bool, str, str | None]] = []
    for category, styles in by_category.items():
        rows.append((True, category, None))
        for style in styles:
            rows.append((False, style, style))
    return rows
