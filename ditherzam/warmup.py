"""Background JIT warmup (Qt-free).

The first render of each Numba kernel pays a compile cost (~hundreds of ms even
with ``cache=True`` linking). Running a few representative renders on a tiny
synthetic image at startup — off the GUI thread — compiles the common kernels and
the color/nearest path so the user's first real interaction is not janky.

Pure compute, no Qt, no output side effects. Safe to run in a daemon thread.
"""
from __future__ import annotations

import numpy as np

# Styles worth pre-compiling: one error-diffusion (serial), one Atkinson variant,
# one ordered/parallel kernel — covers the kernels most first interactions hit.
DEFAULT_WARMUP_STYLES = ("Floyd-Steinberg", "Atkinson", "Bayer-Matrix 4x4")

# Color modes whose mapping kernels the nearest-only style warmup leaves cold but
# a first real interaction may hit (Task 4.5): ordered (_ordered_rgb_njit), ramp
# (_ramp_gray/rgb_njit), diffused (_floyd_steinberg_rgb), plus nearest itself.
DEFAULT_WARMUP_COLOR_MODES = ("nearest", "ordered", "ramp", "diffused")


def warmup_render(registry=None, styles=DEFAULT_WARMUP_STYLES,
                  color_modes=DEFAULT_WARMUP_COLOR_MODES) -> None:
    from .dithering import registry as default_registry
    from .render import RenderPipeline, RenderSettings
    from .color.engine import ColorEngine
    from .color.palette import builtin_palettes

    reg = registry or default_registry
    base = np.linspace(0, 255, 48 * 48, dtype=np.float32).reshape(48, 48)
    palette = builtin_palettes()["gameboy"]

    # with a palette -> also compiles the nearest_indices njit kernel
    pipe = RenderPipeline(reg, ColorEngine(palette, "nearest"), None)
    for style in styles:
        entry = reg.get_entry(style)
        if entry is None:
            continue
        pipe.render(base, RenderSettings(style=style, scale=2))

    # Compile the remaining color-mapping kernels with one tiny render each; a
    # non-neutral saturation pass compiles apply_saturation's real path too.
    for mode in color_modes:
        cpipe = RenderPipeline(reg, ColorEngine(palette, mode, depth=4), None)
        cpipe.render(base, RenderSettings(
            style="Bayer-Matrix 4x4", scale=2, depth=4))
    spipe = RenderPipeline(reg, ColorEngine(palette, "nearest"), None)
    spipe.render(base, RenderSettings(
        style="Bayer-Matrix 4x4", scale=2, saturation=60))


def start_warmup_thread(registry=None, styles=DEFAULT_WARMUP_STYLES,
                        color_modes=DEFAULT_WARMUP_COLOR_MODES):
    """Start ``warmup_render`` on a daemon thread and return the Thread."""
    import threading

    t = threading.Thread(
        target=lambda: _safe(warmup_render, registry, styles, color_modes),
        name="ditherzam-warmup", daemon=True)
    t.start()
    return t


def _safe(fn, *args) -> None:
    try:
        fn(*args)
    except Exception:
        pass  # warmup is best-effort; never crash startup
