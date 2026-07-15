from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import yaml

from .render import RenderSettings
from .color.palette import Palette
from .color.ramp import RAMP_MODES
from .masking.settings import (
    EXPANSION_MAX_PX, EXPANSION_MIN_PX, MaskTarget, OutsideMode,
    SmartMaskSettings,
)

# Allowed ranges used to clamp presets on load (spec §10.2).
_ADJ_RANGE: dict[str, tuple[int, int]] = {
    "contrast": (0, 100),
    "midtones": (0, 100),
    "highlights": (0, 100),
    "luminance_threshold": (0, 100),
    "blur": (0, 100),
    "saturation": (0, 100),
}
_SCALE_RANGE: tuple[int, int] = (1, 20)
_DEPTH_RANGE: tuple[int, int] = (1, 64)


def _clamp_int(value, lo: int, hi: int, default: int | None = None) -> int:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        v = lo if default is None else default
    return max(lo, min(hi, v))


def settings_to_preset(settings: RenderSettings, palette: Palette | None = None,
                       effect_stack=None, color_mode: str = "off",
                       smart_mask: SmartMaskSettings | None = None) -> dict:
    """Serialize a RenderSettings (+ optional palette/effect stack) to a preset dict."""
    preset: dict = {
        "adjustments": {
            "contrast": int(settings.contrast),
            "midtones": int(settings.midtones),
            "highlights": int(settings.highlights),
            "luminance_threshold": int(settings.luminance_threshold),
            "blur": int(settings.blur),
            "saturation": int(settings.saturation),
            "invert": bool(settings.invert),
        },
        "dither": {
            "style": str(settings.style),
            "scale": int(settings.scale),
            "depth": int(settings.depth),
            "color_mapping": str(settings.color_mapping),
            "preview_disabled": bool(settings.preview_disabled),
            "params": dict(settings.params),
        },
    }
    if palette is not None:
        preset["color"] = {
            "mode": str(color_mode),
            "palette": {
                "name": str(palette.name),
                "category": str(getattr(palette, "category", "") or ""),
                "colors": np.asarray(palette.colors, dtype=np.float32)
                            .round().astype(int).reshape(-1, 3).tolist(),
            },
        }
    if effect_stack is not None:
        preset["effects"] = [
            {"name": str(name), "params": dict(params)}
            for name, params in effect_stack.items
        ]
    if smart_mask is not None:
        preset["smart_mask"] = {
            "enabled": smart_mask.enabled,
            "target": smart_mask.target.value,
            "sensitivity": smart_mask.sensitivity,
            "feather_px": smart_mask.feather_px,
            "expansion_px": smart_mask.expansion_px,
            "invert": smart_mask.invert,
            "outside": smart_mask.outside.value,
            "bake_fill": smart_mask.bake_fill,
        }
    return preset


@dataclass(frozen=True)
class PresetContents:
    settings: RenderSettings
    palette: Palette | None
    effects: list[tuple[str, dict]]
    smart_mask: SmartMaskSettings

    def __iter__(self):
        """Keep the historic three-value unpacking API source-compatible."""
        return iter((self.settings, self.palette, self.effects))


def _enum_value(enum_type, value, default):
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def preset_to_settings(preset: dict) -> PresetContents:
    """Deserialize a preset dict into RenderSettings, clamping every value to range."""
    if not isinstance(preset, dict):
        raise ValueError("Not a valid preset file.")
    adj = preset.get("adjustments") or {}
    dit = preset.get("dither") or {}
    if not isinstance(adj, dict) or not isinstance(dit, dict):
        raise ValueError("Not a valid preset file.")

    defaults = RenderSettings()

    def adj_val(key: str) -> int:
        lo, hi = _ADJ_RANGE[key]
        return _clamp_int(adj.get(key, getattr(defaults, key)), lo, hi)

    settings = RenderSettings(
        contrast=adj_val("contrast"),
        midtones=adj_val("midtones"),
        highlights=adj_val("highlights"),
        luminance_threshold=adj_val("luminance_threshold"),
        blur=adj_val("blur"),
        saturation=adj_val("saturation"),
        invert=bool(adj.get("invert", defaults.invert)),
        style=str(dit.get("style", defaults.style)),
        scale=_clamp_int(dit.get("scale", defaults.scale), *_SCALE_RANGE),
        depth=_clamp_int(dit.get("depth", defaults.depth), *_DEPTH_RANGE),
        color_mapping=(str(dit.get("color_mapping", defaults.color_mapping))
                       if dit.get("color_mapping", defaults.color_mapping) in RAMP_MODES
                       else defaults.color_mapping),
        preview_disabled=bool(dit.get("preview_disabled", defaults.preview_disabled)),
        params=dict(dit.get("params", {}) or {}),
    )

    palette: Palette | None = None
    color = preset.get("color")
    if isinstance(color, dict) and isinstance(color.get("palette"), dict):
        pdata = color["palette"]
        colors = np.asarray(pdata.get("colors", []), dtype=np.float32)
        if colors.size:
            colors = colors.reshape(-1, 3)
        else:
            colors = colors.reshape(0, 3)
        palette = Palette(name=str(pdata.get("name", "preset")), colors=colors,
                          category=str(pdata.get("category", "")))

    effects: list[tuple[str, dict]] = []
    for item in preset.get("effects", []) or []:
        if isinstance(item, dict) and "name" in item:
            effects.append((str(item["name"]), dict(item.get("params", {}) or {})))

    mask_defaults = SmartMaskSettings()
    raw_mask = preset.get("smart_mask")
    mask = raw_mask if isinstance(raw_mask, dict) else {}
    smart_mask = SmartMaskSettings(
        enabled=_safe_bool(mask.get("enabled"), mask_defaults.enabled),
        target=_enum_value(MaskTarget, mask.get("target"), mask_defaults.target),
        sensitivity=_clamp_int(mask.get("sensitivity", mask_defaults.sensitivity), 0, 100,
                               mask_defaults.sensitivity),
        feather_px=_clamp_int(mask.get("feather_px", mask_defaults.feather_px), 0, 256,
                              mask_defaults.feather_px),
        expansion_px=_clamp_int(mask.get("expansion_px", mask_defaults.expansion_px),
                                EXPANSION_MIN_PX, EXPANSION_MAX_PX,
                                mask_defaults.expansion_px),
        invert=_safe_bool(mask.get("invert"), mask_defaults.invert),
        outside=_enum_value(OutsideMode, mask.get("outside"), mask_defaults.outside),
        bake_fill=_safe_bool(mask.get("bake_fill"), mask_defaults.bake_fill),
    )
    return PresetContents(settings, palette, effects, smart_mask)


class PresetManager:
    """Filesystem-backed store of preset YAML files (spec §10)."""

    def __init__(self, presets_dir) -> None:
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.presets_dir / f"{name}.yaml"

    def save(self, name: str, preset: dict) -> Path:
        path = self._path(name)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(preset, f, sort_keys=False, allow_unicode=True)
        return path

    def load(self, name: str) -> dict:
        path = self._path(name)
        if not path.is_file():
            raise FileNotFoundError(f"Preset not found: {name}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Not a valid preset file.")
        return data

    def list(self) -> list[str]:
        return sorted(p.stem for p in self.presets_dir.glob("*.yaml"))

    def delete(self, name: str) -> bool:
        path = self._path(name)
        if path.is_file():
            path.unlink()
            return True
        return False

    def import_file(self, src) -> str:
        src = Path(src)
        if src.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError("Not a valid preset file.")
        try:
            data = yaml.safe_load(src.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ValueError("Not a valid preset file.") from e
        if not isinstance(data, dict):
            raise ValueError("Not a valid preset file.")
        name = src.stem
        self.save(name, data)
        return name
