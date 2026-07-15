import math
from ditherzam.effects.glow_params import (
    GLOW_DEFAULTS, glow_params_from_state, glow_state_from_params,
)


def test_defaults_map_to_neutral_params():
    p = glow_params_from_state(GLOW_DEFAULTS)
    assert p["threshold"] == 64.0
    assert p["intensity"] == 1.0            # glow_intensity 50 -> 50/50
    assert p["distance_scale"] == 1.0       # glow_distance 50 -> 50/50
    assert abs(p["aspect"] - 1.0) < 1e-6    # glow_aspect 50 -> 2**0
    assert p["epsilon"] == 0.4 and p["falloff"] == 0.5


def test_forward_scales_each_slider():
    state = dict(GLOW_DEFAULTS, glow_intensity=100, glow_epsilon=100,
                 glow_aspect=100, glow_radius=200)
    p = glow_params_from_state(state)
    assert p["intensity"] == 2.0
    assert p["epsilon"] == 1.0
    assert abs(p["aspect"] - 4.0) < 1e-6
    assert p["radius"] == 200.0


def test_roundtrip_state_params_state():
    state = dict(GLOW_DEFAULTS, glow_threshold=120, glow_radius=40,
                 glow_intensity=70, glow_aspect=75)
    back = glow_state_from_params(glow_params_from_state(state))
    for k in ("glow_threshold", "glow_radius", "glow_intensity", "glow_aspect"):
        assert abs(back[k] - state[k]) <= 1
    assert back["glow_enabled"] is True
