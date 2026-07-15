"""Static, import-free guarantees that Smart Mask stays offline.

``ditherzam/masking/`` must never import a networking/downloader module and
application code must never reference the developer-only staging script. No
model weights may be committed under the asset root.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MASKING_DIR = REPO_ROOT / "ditherzam" / "masking"
DITHERZAM_DIR = REPO_ROOT / "ditherzam"
STAGE_SCRIPT = REPO_ROOT / "tools" / "stage_smart_mask_model.py"
ASSET_ROOT = REPO_ROOT / "assets" / "models" / "smart_mask"

# Substring scan per SM-01: any of these tokens in masking source indicates a
# networking or downloader dependency that must not exist in application code.
FORBIDDEN_TOKENS = ("urllib", "requests", "socket", "http", "download")

FORBIDDEN_WEIGHT_SUFFIXES = (".onnx", ".pt", ".pth", ".bin", ".ckpt", ".safetensors")


def _masking_source_files() -> list[Path]:
    assert MASKING_DIR.is_dir(), f"missing package: {MASKING_DIR}"
    files = sorted(MASKING_DIR.rglob("*.py"))
    assert files, f"no source files under {MASKING_DIR}"
    return files


def test_masking_package_has_no_network_or_downloader_tokens():
    offenders = []
    for path in _masking_source_files():
        text = path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN_TOKENS:
            if token in text:
                offenders.append((str(path.relative_to(REPO_ROOT)), token))
    assert not offenders, f"network/downloader tokens found in masking package: {offenders}"


def test_masking_package_imports_are_stdlib_or_local_only():
    """Every import in ditherzam/masking resolves to stdlib, yaml, or the package itself."""
    import ast

    allowed_top_levels = {
        "__future__", "collections", "dataclasses", "hashlib", "json", "pathlib", "threading", "types", "typing",
        "yaml", "ditherzam",
        # numpy is a pure offline compute library (SM-02's quality metric
        # oracle is required to be pure-NumPy); it does no networking.
        "numpy",
        # enum is stdlib, used for SM-03's MaskTarget/OutsideMode.
        "enum",
        # Pillow performs local-only mask feathering/resampling (SM-04).
        "PIL",
        # CPU-only local inference; imported lazily and pinned in release extras.
        "onnxruntime",
        # Local compiled numeric kernels used by the outer compositor (SM-05).
        "numba",
    }
    offenders = []
    for path in _masking_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in allowed_top_levels:
                        offenders.append((str(path.relative_to(REPO_ROOT)), alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                top = node.module.split(".")[0]
                if node.level == 0 and top not in allowed_top_levels:
                    offenders.append((str(path.relative_to(REPO_ROOT)), node.module))
    assert not offenders, f"unexpected imports in masking package: {offenders}"


def test_application_code_never_imports_or_invokes_the_staging_script():
    """A doc comment naming the script is fine; an import/subprocess call is not."""
    import ast

    assert STAGE_SCRIPT.is_file(), f"missing developer staging script: {STAGE_SCRIPT}"
    invocation_hints = ("subprocess", "popen", "system", "startfile", "import_module", "runpy")
    offenders = []
    for path in sorted(DITHERZAM_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", None) or ""
                names = [alias.name for alias in node.names]
                if "stage_smart_mask_model" in module or any(
                    "stage_smart_mask_model" in name for name in names
                ):
                    offenders.append(str(path.relative_to(REPO_ROOT)))
            elif isinstance(node, ast.Call):
                func_repr = ast.dump(node.func).lower()
                if any(hint in func_repr for hint in invocation_hints):
                    call_source = ast.get_source_segment(source, node) or ""
                    if "stage_smart_mask_model" in call_source:
                        offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"application code references the staging script: {offenders}"


def test_no_model_weights_or_binaries_are_committed():
    # "Committed" means tracked by git. The documented workflow stages weights
    # LOCALLY into this gitignored dir (tools/stage_smart_mask_model.py), so a
    # developer's locally-staged model must not trip this — check git, not disk.
    result = subprocess.run(
        ["git", "ls-files", "--", str(ASSET_ROOT)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    offenders = [
        line for line in result.stdout.splitlines()
        if line.strip() and Path(line).suffix.lower() in FORBIDDEN_WEIGHT_SUFFIXES
    ]
    assert not offenders, f"committed model weight/binary files found: {offenders}"
