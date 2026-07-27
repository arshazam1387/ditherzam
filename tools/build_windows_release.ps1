[CmdletBinding()]
param(
    [switch]$SkipPyInstaller,
    [string]$PythonPath
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = if ($PythonPath) { $PythonPath } else { Join-Path $ProjectRoot ".venv\Scripts\python.exe" }

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Expected a Python 3.12 virtual environment at $Python"
}

& $Python -c "import sys; assert sys.platform == 'win32' and sys.version_info[:2] == (3, 12)"
& $Python -m pip install --requirement (Join-Path $ProjectRoot "packaging\requirements-windows-build.txt")
& $Python -m pip install --no-build-isolation --editable $ProjectRoot
& $Python -c "from ditherzam._native import native_available, smoke_add; assert native_available() and smoke_add(20, 22) == 42"
& $Python -m pytest -q (Join-Path $ProjectRoot "tests\test_native_pixels_exactness.py")

if (-not $SkipPyInstaller) {
    $Lock = Join-Path $ProjectRoot "packaging\smart-mask-release.lock.json"
    if (-not (Test-Path -LiteralPath $Lock)) {
        throw "PyInstaller release requires the approved Smart Mask lock and local assets: $Lock"
    }
    & $Python (Join-Path $ProjectRoot "tools\build_smart_mask_release.py")
    $FrozenNative = Get-ChildItem -Path (Join-Path $ProjectRoot "dist\ditherzam") -Recurse -Filter "_smoke*.pyd"
    if (-not $FrozenNative) {
        throw "Frozen release did not collect ditherzam._native._smoke"
    }
}
