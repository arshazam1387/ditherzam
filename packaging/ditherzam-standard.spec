"""Standard Windows build; deliberately excludes Smart Mask model assets."""
from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata


ROOT = Path(SPECPATH).parent
NATIVE_MODULES = [
    "ditherzam._native._smoke",
    "ditherzam._native._composite",
    "ditherzam._native._selection",
    "ditherzam._native._brush",
]

datas = [
    (str(ROOT / "config" / "config.yaml"), "config"),
    (str(ROOT / "themes" / "default" / "theme.yaml"), "themes/default"),
    (str(ROOT / "ditherzam" / "color" / "builtin" / "*.yaml"),
     "ditherzam/color/builtin"),
    (str(ROOT / "assets" / "ffmpeg" / "*"), "assets/ffmpeg"),
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "THIRD_PARTY_NOTICES.md"), "."),
]
for distribution in (
    "numpy", "numba", "llvmlite", "Pillow", "PyYAML", "platformdirs",
    "PySide6", "PySide6_Addons", "PySide6_Essentials", "shiboken6",
):
    datas += copy_metadata(distribution)

hiddenimports = NATIVE_MODULES + [
    "numba.np.ufunc._internal",
    "numba.np.ufunc.omppool",
    "numba.np.ufunc.workqueue",
    "llvmlite.binding",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
]

a = Analysis(
    [str(ROOT / "ditherzam" / "app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["onnxruntime", "pytest", "tests", "benchmarks"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="ditherzam",
    console=False, contents_directory="_internal",
    version=str(ROOT / "packaging" / "version_info.txt"),
)
coll = COLLECT(
    exe, a.binaries, a.datas, strip=False, upx=False, name="ditherzam",
)
