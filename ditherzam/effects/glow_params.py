"""Pure (Qt-free) mapping between Glow-tab slider ints and epsilon_glow params."""
from __future__ import annotations

import math

GLOW_DEFAULTS: dict = {
    "glow_enabled": False,
    "glow_threshold": 64,     # 0..255  -> threshold
    "glow_smoothing": 32,     # 0..128  -> smoothing
    "glow_radius": 8,         # 0..200  -> radius
    "glow_intensity": 50,     # 0..100  -> intensity = v/50   (0..2)
    "glow_epsilon": 40,       # 0..100  -> epsilon   = v/100  (0..1)
    "glow_falloff": 50,       # 0..100  -> falloff   = v/100  (0..1)
    "glow_distance": 50,      # 1..100  -> distance_scale = v/50 (0.02..2)
    "glow_aspect": 50,        # 0..100  -> aspect = 2**((v-50)/25) (0.25..4)
}


def glow_params_from_state(state: dict) -> dict:
    g = lambda k: state.get(k, GLOW_DEFAULTS[k])
    return {
        "threshold": float(g("glow_threshold")),
        "smoothing": float(g("glow_smoothing")),
        "radius": float(g("glow_radius")),
        "intensity": g("glow_intensity") / 50.0,
        "epsilon": g("glow_epsilon") / 100.0,
        "falloff": g("glow_falloff") / 100.0,
        "distance_scale": g("glow_distance") / 50.0,
        "aspect": 2.0 ** ((g("glow_aspect") - 50) / 25.0),
    }


def glow_state_from_params(params: dict) -> dict:
    return {
        "glow_enabled": True,
        "glow_threshold": int(round(params["threshold"])),
        "glow_smoothing": int(round(params["smoothing"])),
        "glow_radius": int(round(params["radius"])),
        "glow_intensity": int(round(params["intensity"] * 50.0)),
        "glow_epsilon": int(round(params["epsilon"] * 100.0)),
        "glow_falloff": int(round(params["falloff"] * 100.0)),
        "glow_distance": int(round(params["distance_scale"] * 50.0)),
        "glow_aspect": int(round(50 + 25.0 * math.log2(max(params["aspect"], 1e-6)))),
    }
