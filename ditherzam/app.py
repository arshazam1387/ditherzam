from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if "--release-smoke" in sys.argv:
        from ditherzam.release_smoke import run
        index = sys.argv.index("--release-smoke")
        output_dir = Path(sys.argv[index + 1]) if index + 1 < len(sys.argv) else Path.cwd()
        return run(output_dir)
    if "--offline-smoke" in sys.argv:
        from ditherzam.offline_smoke import run
        root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
        return run(root / "packaging" / "smart-mask-release.lock.json")
    from PySide6.QtWidgets import QApplication

    from ditherzam.ui.main_window import ImageEditor
    from ditherzam.ui.theme import find_themes, load_theme

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("ditherzam")

    themes_root = Path(__file__).resolve().parent.parent / "themes"
    if "default" in find_themes(themes_root):
        app.setStyleSheet(load_theme(themes_root, "default").stylesheet)

    # Build the Smart Mask adapter from a locally staged model, if one is present.
    # Fail-closed: with no staged model this returns None and Smart Mask stays
    # cleanly disabled (no network fetch ever happens here).
    from ditherzam.masking.ort_adapter import load_default_segmentation_adapter
    mask_adapter = load_default_segmentation_adapter()
    mask_model = mask_adapter.model_identity if mask_adapter is not None else None

    window = ImageEditor(mask_adapter=mask_adapter, mask_model=mask_model)
    window.resize(1100, 720)
    window.show()

    # Bound interactive renders to the measured thread budget before warming, so
    # the kernels compile at the same thread count they will run at.
    from ditherzam.threading_policy import install_interactive_budget
    install_interactive_budget()

    # Compile the common JIT kernels in the background so the first drag is snappy.
    from ditherzam.warmup import start_warmup_thread
    start_warmup_thread()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
