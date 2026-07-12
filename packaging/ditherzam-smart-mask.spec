"""Minimal Windows/Python 3.12 PyInstaller recipe; run via build_smart_mask_release.py."""
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

ROOT = Path(SPECPATH).parent.parent
LOCK = ROOT / "packaging" / "smart-mask-release.lock.json"
from ditherzam.masking.release_gate import verify_release_bundle

bundle = verify_release_bundle(ROOT, LOCK)  # fail before Analysis/build
datas = [(str(bundle[name]), str(bundle[name].parent.relative_to(ROOT)))
         for name in ("model_manifest", "model", "license", "notice", "provenance")]
datas += collect_data_files("onnxruntime")
binaries = collect_dynamic_libs("onnxruntime")

a = Analysis([str(ROOT / "ditherzam" / "app.py")], pathex=[str(ROOT)],
             binaries=binaries, datas=datas, hiddenimports=["onnxruntime"],
             hookspath=[], runtime_hooks=[], excludes=[])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ditherzam",
          console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="ditherzam")
