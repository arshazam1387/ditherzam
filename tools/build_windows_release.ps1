[CmdletBinding()]
param(
    [switch]$SkipPyInstaller,
    [switch]$SmartMask,
    [switch]$SkipTests,
    [string]$PythonPath,
    [string]$Iscc = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = if ($PythonPath) { $PythonPath } else { Join-Path $ProjectRoot ".venv\Scripts\python.exe" }
Set-Location $ProjectRoot

$FfmpegVersion = "8.1.2"
$FfmpegUrl = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-8.1.2-essentials_build.zip"
$FfmpegArchiveSha256 = "db580001caa24ac104c8cb856cd113a87b0a443f7bdf47d8c12b1d740584a2ec"
$Cache = Join-Path $ProjectRoot ".build-cache"
New-Item -ItemType Directory -Force $Cache | Out-Null

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "External command failed with exit code ${LASTEXITCODE}: $FilePath $($ArgumentList -join ' ')"
    }
}

function Remove-ReleaseOutput {
    param([Parameter(Mandatory = $true)][string]$Name)
    $Target = [IO.Path]::GetFullPath((Join-Path $ProjectRoot $Name))
    $Parent = [IO.Directory]::GetParent($Target).FullName
    if ($Parent -ne $ProjectRoot) {
        throw "Refusing to remove release output outside the project root: $Target"
    }
    if (Test-Path -LiteralPath $Target) {
        Remove-Item -LiteralPath $Target -Recurse -Force
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Expected a Python 3.12 virtual environment at $Python"
}
$ResolvedPython = (Resolve-Path -LiteralPath $Python).Path
$RunningSourceApp = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.ExecutablePath -eq $ResolvedPython -and
        $_.CommandLine -match '(?i)-m\s+ditherzam\.app'
    } |
    Select-Object -First 1
if ($RunningSourceApp) {
    throw "Close the ditherzam process using this build environment before rebuilding native extensions (PID $($RunningSourceApp.ProcessId))"
}

$Pyproject = Get-Content -LiteralPath (Join-Path $ProjectRoot "pyproject.toml") -Raw
$PackageInit = Get-Content -LiteralPath (Join-Path $ProjectRoot "ditherzam\__init__.py") -Raw
$VersionMatch = [regex]::Match($Pyproject, '(?m)^version = "([^"]+)"\r?$')
$InitMatch = [regex]::Match($PackageInit, '(?m)^__version__ = "([^"]+)"\r?$')
if (-not $VersionMatch.Success -or -not $InitMatch.Success) {
    throw "Could not read the release version from project metadata"
}
$Version = $VersionMatch.Groups[1].Value
if ($InitMatch.Groups[1].Value -ne $Version) {
    throw "pyproject.toml and ditherzam/__init__.py versions differ"
}
if ($Version -notmatch '^\d+\.\d+\.\d+$') {
    throw "Public Windows releases require a final X.Y.Z version; found $Version"
}

Invoke-External $Python @("-c", "import platform,sys; assert sys.platform == 'win32' and platform.python_version() == '3.12.13' and platform.architecture()[0] == '64bit'")
Invoke-External $Python @("-m", "pip", "install", "--requirement", (Join-Path $ProjectRoot "packaging\requirements-windows-build.txt"))
Invoke-External $Python @("-m", "pip", "install", "--no-build-isolation", "--editable", $ProjectRoot)
Invoke-External $Python @("-m", "ditherzam.app", "--native-smoke")
Invoke-External $Python @("-m", "pytest", "-q", (Join-Path $ProjectRoot "tests\test_native_pixels_exactness.py"))

if (-not $SkipTests) {
    $env:NUMBA_DISABLE_JIT = "1"
    $env:QT_QPA_PLATFORM = "offscreen"
    try {
        Invoke-External $Python @("-m", "pytest", "-q", "--basetemp=.build-cache\pytest-release")
    }
    finally {
        Remove-Item Env:NUMBA_DISABLE_JIT -ErrorAction SilentlyContinue
    }
    Invoke-External $Python @(
        "-m", "pytest", "-q",
        "tests\test_native_integration_contract.py",
        "tests\test_threading_policy.py",
        "tests\test_native_pixels_exactness.py"
    )
}

if ($SkipPyInstaller) {
    return
}

$Archive = Join-Path $Cache "ffmpeg-$FfmpegVersion-essentials_build.zip"
$Expanded = Join-Path $Cache "ffmpeg-$FfmpegVersion"
$FfmpegAssets = Join-Path $ProjectRoot "assets\ffmpeg"
if (-not (Test-Path -LiteralPath $Archive)) {
    Invoke-WebRequest -Uri $FfmpegUrl -OutFile $Archive
}
$ArchiveHash = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ArchiveHash -ne $FfmpegArchiveSha256) {
    throw "FFmpeg archive hash mismatch: $ArchiveHash"
}
if (-not (Test-Path -LiteralPath $Expanded)) {
    Expand-Archive -LiteralPath $Archive -DestinationPath $Expanded
}
$FfmpegRoot = Get-ChildItem -LiteralPath $Expanded -Directory |
    Where-Object Name -eq "ffmpeg-$FfmpegVersion-essentials_build" |
    Select-Object -First 1
if (-not $FfmpegRoot) {
    throw "Expected FFmpeg $FfmpegVersion directory not found"
}
New-Item -ItemType Directory -Force $FfmpegAssets | Out-Null
Copy-Item -LiteralPath (Join-Path $FfmpegRoot.FullName "bin\ffmpeg.exe") -Destination $FfmpegAssets -Force
Copy-Item -LiteralPath (Join-Path $FfmpegRoot.FullName "bin\ffprobe.exe") -Destination $FfmpegAssets -Force
Copy-Item -LiteralPath (Join-Path $FfmpegRoot.FullName "LICENSE") -Destination (Join-Path $FfmpegAssets "FFMPEG_LICENSE.txt") -Force
Copy-Item -LiteralPath (Join-Path $FfmpegRoot.FullName "README.txt") -Destination (Join-Path $FfmpegAssets "FFMPEG_README.txt") -Force

Remove-ReleaseOutput "build"
Remove-ReleaseOutput "dist"
Remove-ReleaseOutput "release"

if ($SmartMask) {
    $Lock = Join-Path $ProjectRoot "packaging\smart-mask-release.lock.json"
    if (-not (Test-Path -LiteralPath $Lock)) {
        throw "Smart Mask release requires the approved lock and local assets: $Lock"
    }
    Invoke-External $Python @((Join-Path $ProjectRoot "tools\build_smart_mask_release.py"))
}
else {
    Invoke-External $Python @(
        "-m", "PyInstaller", "--clean", "--noconfirm",
        (Join-Path $ProjectRoot "packaging\ditherzam-standard.spec")
    )
}

$FrozenRoot = Join-Path $ProjectRoot "dist\ditherzam"
foreach ($Module in @("_smoke", "_composite", "_selection", "_brush")) {
    $FrozenNative = Get-ChildItem -Path $FrozenRoot -Recurse -Filter "${Module}*.pyd"
    if (-not $FrozenNative) {
        throw "Frozen release did not collect ditherzam._native.$Module"
    }
}
foreach ($RelativePath in @(
    "_internal\config\config.yaml",
    "_internal\themes\default\theme.yaml",
    "_internal\ditherzam\color\builtin\gameboy.yaml",
    "_internal\assets\ffmpeg\ffmpeg.exe",
    "_internal\assets\ffmpeg\ffprobe.exe"
)) {
    if (-not (Test-Path -LiteralPath (Join-Path $FrozenRoot $RelativePath))) {
        throw "Frozen release is missing required runtime data: $RelativePath"
    }
}
if (-not $SmartMask) {
    $Forbidden = Get-ChildItem -Path $FrozenRoot -Recurse -File |
        Where-Object { $_.Extension -eq ".onnx" -or $_.FullName -match "onnxruntime" }
    if ($Forbidden) {
        throw "Standard release unexpectedly contains Smart Mask model/runtime files"
    }
}
Invoke-External (Join-Path $FrozenRoot "ditherzam.exe") @("--native-smoke")
Invoke-External (Join-Path $FrozenRoot "_internal\assets\ffmpeg\ffmpeg.exe") @("-version")
Invoke-External (Join-Path $FrozenRoot "_internal\assets\ffmpeg\ffprobe.exe") @("-version")

Copy-Item -LiteralPath (Join-Path $ProjectRoot "LICENSE") -Destination (Join-Path $FrozenRoot "LICENSE") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "THIRD_PARTY_NOTICES.md") -Destination (Join-Path $FrozenRoot "THIRD_PARTY_NOTICES.md") -Force
$PortableReadme = (Get-Content -LiteralPath (Join-Path $ProjectRoot "packaging\PORTABLE_README.txt") -Raw).Replace("{{VERSION}}", $Version)
Set-Content -LiteralPath (Join-Path $FrozenRoot "README.txt") -Value $PortableReadme -Encoding ascii

$ReleaseRoot = Join-Path $ProjectRoot "release"
New-Item -ItemType Directory -Force $ReleaseRoot | Out-Null
$Edition = if ($SmartMask) { "smart-mask-" } else { "" }
$ArtifactStem = "ditherzam-$Version-${Edition}windows-x64"
$Portable = Join-Path $ReleaseRoot "$ArtifactStem-portable.zip"
Compress-Archive -Path (Join-Path $FrozenRoot "*") -DestinationPath $Portable -CompressionLevel Optimal

if (-not $Iscc) {
    $Candidates = @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )
    $Iscc = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $Iscc -or -not (Test-Path -LiteralPath $Iscc)) {
    throw "Inno Setup 6 ISCC.exe not found"
}
Invoke-External $Iscc @(
    "/DMyAppVersion=$Version",
    "/DMyArtifactStem=$ArtifactStem",
    (Join-Path $ProjectRoot "packaging\ditherzam.iss")
)

$Setup = Join-Path $ReleaseRoot "$ArtifactStem-setup.exe"
$Hashes = @($Portable, $Setup) | ForEach-Object {
    $Hash = Get-FileHash -LiteralPath $_ -Algorithm SHA256
    "$($Hash.Hash.ToLowerInvariant())  $([IO.Path]::GetFileName($_))"
}
$Hashes | Set-Content -LiteralPath (Join-Path $ReleaseRoot "SHA256SUMS.txt") -Encoding ascii
$Hashes
