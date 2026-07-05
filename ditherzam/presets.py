from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .render import RenderSettings
from .color.palette import Palette

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


def _clamp_int(value, lo: int, hi: int) -> int:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        v = lo
    return max(lo, min(hi, v))


def settings_to_preset(settings: RenderSettings, palette: Palette | None = None,
                       effect_stack=None, color_mode: str = "off") -> dict:
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
            "preview_disabled": bool(settings.preview_disabled),
            "params": dict(settings.params),
        },
    }
    if palette is not None:
        preset["color"] = {
            "mode": str(color_mode),
            "palette": {
                "name": str(palette.name),
                "colors": np.asarray(palette.colors, dtype=np.float32)
                            .round().astype(int).reshape(-1, 3).tolist(),
            },
        }
    if effect_stack is not None:
        preset["effects"] = [
            {"name": str(name), "params": dict(params)}
            for name, params in effect_stack.items
        ]
    return preset


def preset_to_settings(preset: dict) -> tuple[RenderSettings, Palette | None, list[tuple[str, dict]]]:
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
        palette = Palette(name=str(pdata.get("name", "preset")), colors=colors)

    effects: list[tuple[str, dict]] = []
    for item in preset.get("effects", []) or []:
        if isinstance(item, dict) and "name" in item:
            effects.append((str(item["name"]), dict(item.get("params", {}) or {})))

    return settings, palette, effects


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
