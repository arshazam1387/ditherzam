import numpy as np
from ditherzam.dithering.registry import DitherRegistry
from ditherzam.dithering.pipeline import apply_dither


def test_param_func_transforms_slider_params_before_kernel():
    reg = DitherRegistry()
    captured = {}

    def double_param(params):
        return params["strength"] * 2

    @reg.register("PF", "Error Diffusion", dims=2,
                  param_sliders=("strength",), param_func=double_param)
    def pf(image_array, parameter, luminance_threshold_value):
        captured["param"] = parameter
        return np.where(image_array >= parameter, 255.0, 0.0).astype(np.float32)

    img = np.full((4, 4), 100.0, dtype=np.float32)
    apply_dither(img, style="PF", scale=1, luminance_threshold=50,
                 params={"strength": 30}, registry=reg)
    # param_func doubled 30 -> 60 and that value reached the kernel
    assert captured["param"] == 60


def test_param_func_overrides_multi_slider_extraction():
    reg = DitherRegistry()
    seen = {}

    def constant_seven(params):
        return 7

    @reg.register("PF2", "Error Diffusion", dims=2,
                  param_sliders=("a", "b"), param_func=constant_seven)
    def pf2(image_array, parameter, luminance_threshold_value):
        seen["param"] = parameter
        return image_array

    img = np.zeros((3, 3), dtype=np.float32)
    apply_dither(img, style="PF2", scale=1, luminance_threshold=50,
                 params={"a": 1, "b": 2}, registry=reg)
    # param_func wins over the (a, b) tuple that param_sliders would have built
    assert seen["param"] == 7


def test_no_param_func_falls_back_to_slider_values():
    reg = DitherRegistry()
    seen = {}

    @reg.register("PF3", "Error Diffusion", dims=2, param_sliders=("k",))
    def pf3(image_array, parameter, luminance_threshold_value):
        seen["param"] = parameter
        return image_array

    img = np.zeros((3, 3), dtype=np.float32)
    apply_dither(img, style="PF3", scale=1, luminance_threshold=50,
                 params={"k": 42}, registry=reg)
    assert seen["param"] == 42  # single slider -> scalar, unchanged


def test_threshold_field_kwarg_is_accepted_and_inert():
    # Frozen-contract signature must accept threshold_field even though Phase 1 ignores it.
    reg = DitherRegistry()

    @reg.register("PF4", "Error Diffusion", dims=2)
    def pf4(image_array, parameter, luminance_threshold_value):
        return image_array

    img = np.full((3, 3), 10.0, dtype=np.float32)
    field = np.zeros((3, 3), dtype=np.float32)
    out = apply_dither(img, style="PF4", scale=1, luminance_threshold=50,
                       params={}, registry=reg, threshold_field=field)
    np.testing.assert_allclose(out, img)  # inert in Phase 1
