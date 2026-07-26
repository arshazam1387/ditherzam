from __future__ import annotations

import math

from ..composition import Look
from .model import Layer, LayerStack


def _yaml_safe(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("preset must contain finite YAML-safe values")
        return value
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("preset keys must be strings")
        return {key: _yaml_safe(item) for key, item in value.items()}
    raise ValueError("preset must contain only YAML-safe values")


def layer_stack_to_dict(stack: LayerStack) -> dict:
    if not isinstance(stack, LayerStack):
        raise ValueError("stack must be a LayerStack")
    return {
        "version": 1,
        "layers": [
            {
                "id": layer.id,
                "name": layer.name,
                "look": {
                    "name": layer.look.name,
                    "preset": _yaml_safe(layer.look.preset),
                },
                "visible": layer.visible,
                "opacity": layer.opacity,
                "blend_mode": layer.blend_mode,
            }
            for layer in stack.layers
        ],
    }


def layer_stack_from_dict(payload: dict) -> LayerStack:
    try:
        if (
            not isinstance(payload, dict)
            or type(payload.get("version")) is not int
            or payload["version"] != 1
        ):
            raise ValueError("layer stack version must be 1")
        if not isinstance(payload.get("layers"), list):
            raise ValueError("layers must be a list")
        layers = []
        for raw in payload["layers"]:
            if not isinstance(raw, dict) or not isinstance(raw.get("look"), dict):
                raise ValueError("layer must be a mapping")
            look_data = raw["look"]
            layers.append(Layer(
                raw["id"],
                raw["name"],
                Look(look_data["name"], _yaml_safe(look_data["preset"])),
                raw["visible"],
                raw["opacity"],
                raw["blend_mode"],
            ))
        return LayerStack(layers)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("malformed layer stack data") from exc
