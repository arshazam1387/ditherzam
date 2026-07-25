"""Regenerate the README gallery in docs/images/ from the SVG sources in docs/demo/.

Usage (from the repo root):
    python tools/render_gallery.py

Rasterizes each SVG with Qt's SVG renderer (offscreen), pushes the result
through the real render pipeline at a handful of style/palette combinations,
and assembles the labeled contact sheets the README embeds.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEMO = ROOT / "docs" / "demo"
OUT = ROOT / "docs" / "images"

CELL_W = 560          # width every gallery cell is resized to
LABEL_H = 34


def rasterize_svgs() -> dict[str, Path]:
    from PySide6.QtGui import QGuiApplication, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    QGuiApplication.instance() or QGuiApplication([sys.argv[0]])
    out = {}
    for svg in sorted(DEMO.glob("*.svg")):
        renderer = QSvgRenderer(str(svg))
        size = renderer.defaultSize()
        image = QImage(size.width(), size.height(), QImage.Format_RGB32)
        image.fill(0xFFFFFFFF)
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()
        png = OUT / f"source-{svg.stem}.png"
        image.save(str(png))
        out[svg.stem] = png
    return out


def dithered(source_png: Path, style: str, palette: str | None = None,
             scale: int = 4, depth: int = 4) -> Image.Image:
    from ditherzam.color.engine import ColorEngine
    from ditherzam.color.palette import builtin_palettes
    from ditherzam.dithering import registry
    from ditherzam.imaging import to_gray_f32
    from ditherzam.render import RenderPipeline, RenderSettings

    gray = to_gray_f32(Image.open(source_png))
    engine = ColorEngine(builtin_palettes()[palette], "ramp") if palette else None
    pipeline = RenderPipeline(registry, engine)
    settings = RenderSettings(style=style, scale=scale, depth=depth)
    rgb = pipeline.render(gray, settings)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def _font(size: int):
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def labeled(img: Image.Image, text: str) -> Image.Image:
    h = int(img.height * CELL_W / img.width)
    cell = img.resize((CELL_W, h), Image.LANCZOS)
    canvas = Image.new("RGB", (CELL_W, h + LABEL_H), "#111110")
    canvas.paste(cell, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, h + LABEL_H // 2), text, fill="#e8e4da",
              font=_font(18), anchor="lm")
    return canvas


def sheet(cells: list[Image.Image], cols: int, path: Path, gap: int = 6) -> None:
    rows = (len(cells) + cols - 1) // cols
    w = cols * CELL_W + (cols - 1) * gap
    h = rows * cells[0].height + (rows - 1) * gap
    board = Image.new("RGB", (w, h), "#111110")
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        board.paste(cell, (c * (CELL_W + gap), r * (cell.height + gap)))
    board.save(path, optimize=True)
    print(f"wrote {path.relative_to(ROOT)}  {board.size}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = rasterize_svgs()

    # Hero: source next to a classic error-diffusion render.
    hero_src = Image.open(sources["orbit"])
    hero = [labeled(hero_src, "source (docs/demo/orbit.svg)"),
            labeled(dithered(sources["orbit"], "Floyd-Steinberg", "gameboy",
                             scale=4, depth=4), "Floyd-Steinberg / gameboy")]
    sheet(hero, 2, OUT / "hero.png")

    # One style per family, each with its own palette.
    combos = [
        ("Floyd-Steinberg", "grayscale", 4, 2),
        ("Atkinson", "sepia", 4, 2),
        ("Hilbert (Riemersma)", "greencrt", 4, 2),
        ("Bayer-Ordered", "c64", 4, 4),
        ("Halftone-Ordered", "nord", 4, 4),
        ("Hex Bayer", "ambercrt", 4, 4),
        ("Stippling", "noir_teal", 4, 4),
        ("Crosshatch", "honey_ink", 4, 4),
        ("Line Screen", "ultraviolet", 4, 4),
        ("Waveform", "zxspectrum", 4, 4),
        ("Glitch", "electric_night", 4, 4),
        ("Atkinson-VHS", "coral_sunset", 4, 4),
    ]
    cells = [labeled(dithered(sources["orbit"], s, p, sc, d), f"{s} / {p}")
             for s, p, sc, d in combos]
    sheet(cells, 3, OUT / "styles.png")

    # Signature generative styles on the waves source.
    special = [
        ("Echo Smear", "midnight_bloom", 4, 4),
        ("Feedback Smear", "greencrt", 4, 4),
        ("Reaction-Diffusion", "ocean_glass", 4, 4),
        ("Quasicrystal", "velvet_gold", 4, 4),
        ("Spiral Engrave", "rosewood", 4, 4),
        ("Topography", "alpine_lake", 4, 4),
    ]
    cells = [labeled(dithered(sources["waves"], s, p, sc, d), f"{s} / {p}")
             for s, p, sc, d in special]
    sheet(cells, 3, OUT / "special.png")

    # One style, many palettes, on the poster source.
    palettes = ["gameboy", "pico8", "cga", "solarized", "lavender_milk", "desert_film"]
    cells = [labeled(dithered(sources["poster"], "Bayer-Matrix 8x8", p, 4, 4),
                     f"Bayer-Matrix 8x8 / {p}") for p in palettes]
    sheet(cells, 3, OUT / "palettes.png")


if __name__ == "__main__":
    main()
