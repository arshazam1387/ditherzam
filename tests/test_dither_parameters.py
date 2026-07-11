from ditherzam.dithering import registry
from ditherzam.dithering.parameters import parameter_specs


def test_every_declared_parameter_has_valid_ui_metadata():
    for style in registry.list_dithers():
        entry = registry.get_entry(style)
        specs = parameter_specs(entry)
        keys = tuple(spec.key for spec in specs)
        if style == "None":
            assert keys == ()
        else:
            assert keys[-len(entry.param_sliders):] == entry.param_sliders if entry.param_sliders else True
        for spec in specs:
            assert spec.label
            assert spec.minimum <= spec.default <= spec.maximum


def test_style_primary_parameter_uses_artistic_name_and_real_range():
    entry = registry.get_entry("Flow Hatch")
    spec = {s.key: s for s in parameter_specs(entry)}["dither_parameter_slider"]
    assert spec.label == "Hatch Spacing"
    assert (spec.minimum, spec.maximum, spec.default) == (2, 24, 6)


def test_reused_legacy_keys_can_have_style_specific_ranges():
    smooth = {s.key: s for s in parameter_specs(
        registry.get_entry("Smooth Diffuse"))}["smoothness_slider"]
    contour = {s.key: s for s in parameter_specs(
        registry.get_entry("Displace Contour"))}["smoothing_slider"]
    assert (smooth.minimum, smooth.maximum, smooth.default) == (1, 10, 5)
    assert (contour.minimum, contour.maximum, contour.default) == (0, 5, 0)
