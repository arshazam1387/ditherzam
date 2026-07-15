import numpy as np
import pytest

from ditherzam.dithering import registry
from ditherzam.dithering.parameters import parameter_specs
from ditherzam.dithering.pipeline import apply_dither


FAMILY_STYLES = tuple(
    name
    for category in ("Glitch Effects", "Special Effects")
    for name in registry.by_category()[category]
    if registry.get_entry(name).func.__module__.endswith((".glitch", ".special"))
)


@pytest.fixture(scope="module")
def audit_images():
    y, x = np.mgrid[:48, :64]
    return (
        np.tile(np.linspace(0, 255, 64, dtype=np.float32), (48, 1)),
        np.full((48, 64), 127.0, np.float32),
        np.where(x < 32, 40.0, 215.0).astype(np.float32),
        ((x * 17 + y * 29 + (x * y) % 53) % 256).astype(np.float32),
    )


# Classic defaults leave a few controls dormant until a companion control is
# raised (e.g. Glitch at intensity 1 never shifts a row, so seed/hold/wrap
# have nothing to act on). Those controls are exercised from the companion
# baseline instead of the defaults.
COMPANION_BASELINES = {
    ("Glitch", "glitch_seed_slider"): ("dither_parameter_slider", 8),
    ("Glitch", "glitch_row_hold_slider"): ("dither_parameter_slider", 8),
    ("Glitch", "glitch_wrap_slider"): ("dither_parameter_slider", 8),
    ("Uniform Modulation X", "smoothing_factor_slider"): ("bleed_fraction_slider", 50),
}


@pytest.mark.parametrize("name", FAMILY_STYLES)
def test_each_native_control_changes_at_least_one_audit_fixture(name, audit_images):
    entry = registry.get_entry(name)
    specs = parameter_specs(entry)[6:]
    base_defaults = tuple(spec.default for spec in specs)
    keys = [spec.key for spec in specs]
    baselines = {}

    def baseline_for(defaults):
        if defaults not in baselines:
            baselines[defaults] = [entry.func(image.copy(), defaults, 127.5)
                                   for image in audit_images]
        return baselines[defaults]

    for index, spec in enumerate(specs):
        defaults = base_defaults
        companion = COMPANION_BASELINES.get((name, spec.key))
        if companion is not None:
            adjusted = list(base_defaults)
            adjusted[keys.index(companion[0])] = companion[1]
            defaults = tuple(adjusted)
        baseline = baseline_for(defaults)
        span = spec.maximum - spec.minimum
        candidates = {spec.minimum, spec.minimum + span // 4,
                      spec.minimum + span // 2, spec.minimum + 3 * span // 4,
                      spec.maximum}
        changed = False
        for value in candidates - {spec.default}:
            params = list(defaults)
            params[index] = value
            if any(
                np.any(entry.func(image.copy(), tuple(params), 127.5) != base)
                for image, base in zip(audit_images, baseline)
            ):
                changed = True
                break
        assert changed, f"{name}: {spec.label} is inert across its declared range"


def test_upgraded_defaults_are_not_duplicate_styles(audit_images):
    # Artifact Modulation / Waveform Alt and Modulated Diffuse X / Uniform
    # Modulation X share their classic default output by request (restored
    # pre-audit defaults); they diverge once native sliders move.
    pairs = (
        ("Modulated Diffuse Y", "Uniform Modulation Y"),
        ("Diagonal", "Wireframe Alt"),
    )
    texture = audit_images[-1]
    for left, right in pairs:
        le = registry.get_entry(left)
        re = registry.get_entry(right)
        lp = tuple(s.default for s in parameter_specs(le)[6:])
        rp = tuple(s.default for s in parameter_specs(re)[6:])
        assert np.any(le.func(texture.copy(), lp, 127.5) != re.func(texture.copy(), rp, 127.5))


def test_geometric_spacing_creates_literal_blank_scanlines(audit_images):
    image = audit_images[-1]
    for name, blank_axis in (("Modulated Diffuse Y", 1), ("Contrast Aware Y", 1),
                             ("Modulated Diffuse X", 0), ("Contrast Aware X", 0)):
        entry = registry.get_entry(name)
        params = [s.default for s in parameter_specs(entry)[6:]]
        params[-1] = 4
        out = entry.func(image.copy(), tuple(params), 127.5)
        if blank_axis == 1:
            assert np.all(out[1::4, :] == 255) and np.all(out[2::4, :] == 255)
        else:
            assert np.all(out[:, 1::4] == 255) and np.all(out[:, 2::4] == 255)


def test_contour_metadata_does_not_call_density_line_spacing():
    entry = registry.get_entry("Displace Contour")
    labels = [spec.label for spec in parameter_specs(entry)[6:]]
    assert labels == ["Tone Limit", "Line Thickness", "Contour Reach",
                      "Contour Density", "Contour Displacement"]


def test_uniform_smoothing_is_normalized_and_stays_binary(audit_images):
    entry = registry.get_entry("Uniform Modulation Y")
    defaults = [spec.default for spec in parameter_specs(entry)[6:]]
    for smoothing in (0, 50, 100):
        defaults[1] = smoothing
        out = entry.func(audit_images[-1].copy(), tuple(defaults), 127.5)
        assert np.isfinite(out).all()
        assert set(np.unique(out)).issubset({0.0, 255.0})


def test_edge_styles_are_distinct_and_noncollapsed_through_ui_default_pipeline():
    y, x = np.mgrid[:96, :128]
    fixtures = (
        np.tile(np.linspace(0, 255, 128, dtype=np.float32), (96, 1)),
        ((x * 3 + y * 5 + 32 * np.sin(x / 9) + 24 * np.cos(y / 7)) % 256).astype(np.float32),
    )
    rendered = {}
    for name in ("Diagonal", "Wireframe Alt"):
        entry = registry.get_entry(name)
        params = {spec.key: spec.default for spec in parameter_specs(entry)[6:]}
        rendered[name] = [
            apply_dither(image, style=name, scale=1, luminance_threshold=50,
                         params=params, registry=registry, levels=2)
            for image in fixtures
        ]
        # Classic edge detectors render a constant-slope ramp as blank white
        # (fixture 0); only the textured fixture must produce visible edges.
        ratio = np.mean(rendered[name][1] == 0)
        assert 0.01 < ratio < 0.99, (name, ratio)
    assert np.any(rendered["Diagonal"][1] != rendered["Wireframe Alt"][1])
