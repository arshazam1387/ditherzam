"""Frozen standard-build smoke check; invoked explicitly by release tooling."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication


def run(output_dir: str | Path) -> int:
    from ditherzam.export.raster import save_raster
    from ditherzam.export.vector import raster_to_svg
    from ditherzam.masking.ort_adapter import load_default_segmentation_adapter
    from ditherzam.ui.main_window import ImageEditor
    from ditherzam.video.ffmpeg import ffmpeg_bin, ffprobe_bin, have_ffmpeg

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Create, save, and reopen a real PNG so decoder and file I/O are exercised.
    y, x = np.mgrid[:96, :128]
    source_rgb = np.stack(((x * 2) % 256, (y * 3) % 256, (x + y) % 256), axis=2).astype(np.uint8)
    source_path = out / "smoke-input.png"
    Image.fromarray(source_rgb, "RGB").save(source_path)
    with Image.open(source_path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        gray = np.asarray(opened.convert("L"), dtype=np.float32)

    app = QApplication.instance() or QApplication([])
    adapter = load_default_segmentation_adapter()
    window = ImageEditor(mask_adapter=adapter, mask_model=None)
    window.show()
    app.processEvents()
    window.load_array(gray, rgb)
    window.panel.state["style"] = "Floyd-Steinberg"
    window.panel.state["color_mode"] = "nearest"
    window.panel.set_working_palette(window.panel.store.get("gameboy"))
    window._sync_pipeline()
    rendered = window._export_pipeline().render(gray, window._collect_settings())

    png_path = out / "smoke-export.png"
    svg_path = out / "smoke-export.svg"
    save_raster(rendered, png_path)
    svg_path.write_text(raster_to_svg(gray.astype(np.uint8), 128), encoding="utf-8")

    result = {
        "launched": window.isVisible(),
        "opened_png": source_path.is_file(),
        "style": window.panel.state["style"],
        "palette": window.panel.working_palette.name,
        "export_png": png_path.is_file() and png_path.stat().st_size > 0,
        "export_svg": svg_path.is_file() and "<svg" in svg_path.read_text(encoding="utf-8"),
        "smart_mask_disabled": adapter is None and not window.panel.smart_mask_panel.settings.enabled,
        "ffmpeg_detected": have_ffmpeg(),
        "ffmpeg": ffmpeg_bin(),
        "ffprobe": ffprobe_bin(),
        "numba_render_shape": list(rendered.shape),
    }
    window.close()
    app.processEvents()
    (out / "smoke-report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if all(result[k] for k in (
        "launched", "opened_png", "export_png", "export_svg",
        "smart_mask_disabled", "ffmpeg_detected")) else 1
