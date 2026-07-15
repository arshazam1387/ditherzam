[CmdletBinding()]
param(
    [string]$Python = ".venv-release\Scripts\python.exe",
    [string]$Iscc = "",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$Version = "0.1.0"
$FfmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$FfmpegArchiveSha256 = "db580001caa24ac104c8cb856cd113a87b0a443f7bdf47d8c12b1d740584a2ec"
$Cache = Join-Path $Root ".build-cache"
$Archive = Join-Path $Cache "ffmpeg-release-essentials.zip"
$Expanded = Join-Path $Cache "ffmpeg"
$FfmpegAssets = Join-Path $Root "assets\ffmpeg"

if (-not (Test-Path $Python)) { throw "Python 3.12 environment not found: $Python" }
$ActualPython = & $Python -c "import platform; print(platform.python_version()); print(platform.architecture()[0])"
if ($ActualPython[0] -ne "3.12.13" -or $ActualPython[1] -ne "64bit") {
    throw "Release requires CPython 3.12.13 x64; found $($ActualPython -join ' ')"
}

New-Item -ItemType Directory -Force $Cache | Out-Null
if (-not (Test-Path $Archive)) {
    Invoke-WebRequest -Uri $FfmpegUrl -OutFile $Archive
}
$ArchiveHash = (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ArchiveHash -ne $FfmpegArchiveSha256) {
    throw "FFmpeg archive hash mismatch: $ArchiveHash"
}
if (-not (Test-Path $Expanded)) {
    Expand-Archive -LiteralPath $Archive -DestinationPath $Expanded
}
$FfmpegRoot = Get-ChildItem $Expanded -Directory | Where-Object Name -eq "ffmpeg-8.1.2-essentials_build" | Select-Object -First 1
if (-not $FfmpegRoot) { throw "Expected FFmpeg 8.1.2 directory not found" }

New-Item -ItemType Directory -Force $FfmpegAssets | Out-Null
Copy-Item (Join-Path $FfmpegRoot.FullName "bin\ffmpeg.exe") $FfmpegAssets -Force
Copy-Item (Join-Path $FfmpegRoot.FullName "bin\ffprobe.exe") $FfmpegAssets -Force
Copy-Item (Join-Path $FfmpegRoot.FullName "LICENSE") (Join-Path $FfmpegAssets "FFMPEG_LICENSE.txt") -Force
Copy-Item (Join-Path $FfmpegRoot.FullName "README.txt") (Join-Path $FfmpegAssets "FFMPEG_README.txt") -Force

if (-not $SkipTests) {
    $env:NUMBA_DISABLE_JIT = "1"
    & $Python -m pytest -p no:cacheprovider -q --basetemp=.build-cache\pytest-release
    if ($LASTEXITCODE -ne 0) { throw "Focused test suite failed" }
    Remove-Item Env:NUMBA_DISABLE_JIT -ErrorAction SilentlyContinue
}

Remove-Item build, dist, release -Recurse -Force -ErrorAction SilentlyContinue
& $Python -m PyInstaller --noconfirm --clean packaging\ditherzam-standard.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }

Copy-Item packaging\PORTABLE_README.txt dist\ditherzam\README.txt
Copy-Item LICENSE dist\ditherzam\LICENSE
Copy-Item THIRD_PARTY_NOTICES.md dist\ditherzam\THIRD_PARTY_NOTICES.md
New-Item -ItemType Directory -Force release | Out-Null
$Portable = "release\ditherzam-$Version-windows-x64-portable.zip"
Compress-Archive -Path dist\ditherzam\* -DestinationPath $Portable -CompressionLevel Optimal

if (-not $Iscc) {
    $Candidates = @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )
    $Iscc = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $Iscc -or -not (Test-Path $Iscc)) { throw "Inno Setup 6 ISCC.exe not found" }
& $Iscc packaging\ditherzam.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup build failed" }

$Setup = "release\ditherzam-$Version-windows-x64-setup.exe"
$Hashes = @($Portable, $Setup) | ForEach-Object {
    $h = Get-FileHash $_ -Algorithm SHA256
    "$($h.Hash.ToLowerInvariant())  $([IO.Path]::GetFileName($_))"
}
$Hashes | Set-Content release\SHA256SUMS.txt -Encoding ascii
$Hashes
