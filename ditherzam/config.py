from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


class ConfigError(Exception):
    """Raised when configuration is missing or malformed."""


@dataclass
class AppConfig:
    default_dither_style: str
    default_dither_scale: int
    viewport_bg_color: str
    app_style: str
    enable_inertia: bool
    friction: float
    velocity_scale: float
    max_velocity: float
    debounce_ms: int
    loading_delay_ms: int


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Configuration file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing configuration: {e}") from e
    if not data:
        raise ConfigError("Configuration file is empty.")
    try:
        return AppConfig(
            default_dither_style=data["dither"]["default_style"],
            default_dither_scale=int(data["dither"]["default_scale"]),
            viewport_bg_color=data["style"]["viewport_bg_color"],
            app_style=data["style"]["app_style"],
            enable_inertia=bool(data["inertia"]["enable"]),
            friction=float(data["inertia"]["friction"]),
            velocity_scale=float(data["inertia"]["velocity_scale"]),
            max_velocity=float(data["inertia"]["max_velocity"]),
            debounce_ms=int(data["timing"]["debounce_ms"]),
            loading_delay_ms=int(data["timing"]["loading_delay_ms"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"Configuration validation failed: {e}") from e
