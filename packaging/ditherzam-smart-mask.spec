"""Minimal Windows/Python 3.12 PyInstaller recipe; run via build_smart_mask_release.py."""
from pathlib import Path

ROOT = Path(SPECPATH).parent.parent
LOCK = ROOT / "packaging" / "smart-mask-release.lock.json"
from ditherzam.masking.release_gate import verify_release_bundle

bundle = verify_release_bundle(ROOT, LOCK)  # fail before Analysis/build
datas = [(str(bundle[name]), str(bundle[name].parent.relative_to(ROOT)))
         for name in ("model_manifest", "model", "license", "notice", "provenance", "smoke_fixture")]
datas.append((str(LOCK), "packaging"))
binaries = [(str(bundle[f"ort:{name}"]), "onnxruntime/capi") for name in
            ("onnxruntime.dll", "onnxruntime_providers_shared.dll")]

a = Analysis([str(ROOT / "ditherzam" / "app.py")], pathex=[str(ROOT)],
             binaries=binaries, datas=datas, hiddenimports=["onnxruntime"],
             hookspath=[], runtime_hooks=[], excludes=[])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ditherzam",
          console=False, contents_directory=".")
coll = COLLECT(exe, a.binaries, a.datas, name="ditherzam")
