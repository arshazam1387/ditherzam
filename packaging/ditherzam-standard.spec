"""Standard no-model Windows build; Smart Mask remains fail-closed."""
from pathlib import Path

ROOT = Path.cwd()
NATIVE_MODULES = [
    "ditherzam._native._smoke",
    "ditherzam._native._composite",
    "ditherzam._native._selection",
    "ditherzam._native._brush",
]

a = Analysis(
    [str(ROOT / "ditherzam" / "app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=NATIVE_MODULES,
    hookspath=[],
    runtime_hooks=[],
    excludes=["onnxruntime"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="ditherzam",
    console=False, contents_directory=".",
)
coll = COLLECT(exe, a.binaries, a.datas, name="ditherzam")
