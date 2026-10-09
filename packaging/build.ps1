# Assemble dist/ASR: Tray exe + embeddable Python + copied sources.
# Run from anywhere: powershell -File packaging/build.ps1
# Requires conda env lingting (for PyInstaller / tray deps).

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Out = Join-Path $Repo "dist\ASR"
$PyVer = "3.10.11"
$EmbedZipName = "python-$PyVer-embed-amd64.zip"
$EmbedUrl = "https://www.python.org/ftp/python/$PyVer/$EmbedZipName"
$GetPipUrl = "https://bootstrap.pypa.io/get-pip.py"
$Cache = Join-Path $Repo "dist\cache"

Write-Host "Repo: $Repo"
Write-Host "Out:  $Out"

New-Item -ItemType Directory -Force (Join-Path $Repo "dist") | Out-Null
New-Item -ItemType Directory -Force $Cache | Out-Null

if (Test-Path $Out) {
    Remove-Item -Recurse -Force $Out
}
New-Item -ItemType Directory -Force $Out | Out-Null

# --- Tray exe ---
Write-Host "Building ASRTray.exe ..."
$LingtingPy = "D:\anaconda3\envs\lingting\python.exe"
if (-not (Test-Path $LingtingPy)) {
    $LingtingPy = (Get-Command python -ErrorAction Stop).Source
}
Push-Location $Repo
try {
    & $LingtingPy -m PyInstaller --noconfirm --clean --distpath (Join-Path $Repo "dist\tray-build") --workpath (Join-Path $Repo "dist\pyi-work") (Join-Path $PSScriptRoot "ASRTray.spec")
    if ($LASTEXITCODE -ne 0) { throw "pyinstaller failed: $LASTEXITCODE" }
} finally {
    Pop-Location
}
$TrayExe = Join-Path $Repo "dist\tray-build\ASRTray.exe"
if (-not (Test-Path $TrayExe)) {
    throw "Tray exe not found: $TrayExe"
}
Copy-Item $TrayExe (Join-Path $Out "ASRTray.exe")

# --- Embeddable CPython ---
$ZipPath = Join-Path $Cache $EmbedZipName
if (-not (Test-Path $ZipPath)) {
    Write-Host "Downloading $EmbedUrl ..."
    Invoke-WebRequest -Uri $EmbedUrl -OutFile $ZipPath
}
$Runtime = Join-Path $Out "runtime"
New-Item -ItemType Directory -Force $Runtime | Out-Null
Write-Host "Unpacking embeddable Python ..."
Expand-Archive -Path $ZipPath -DestinationPath $Runtime -Force

$Pth = Get-ChildItem $Runtime -Filter "python*._pth" | Select-Object -First 1
if (-not $Pth) { throw "python._pth not found in runtime" }
@(
    "python310.zip"
    "."
    "Lib\site-packages"
    "import site"
) | Set-Content -Path $Pth.FullName -Encoding ascii

$GetPip = Join-Path $Cache "get-pip.py"
if (-not (Test-Path $GetPip)) {
    Write-Host "Downloading get-pip.py ..."
    Invoke-WebRequest -Uri $GetPipUrl -OutFile $GetPip
}
$Py = Join-Path $Runtime "python.exe"
Write-Host "Installing pip + requirements into runtime ..."
& $Py $GetPip --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw "get-pip failed" }
& $Py -m pip install --no-warn-script-location -r (Join-Path $Repo "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install requirements failed" }

# --- App sources ---
$App = Join-Path $Out "app"
New-Item -ItemType Directory -Force $App | Out-Null
Copy-Item (Join-Path $Repo "run.py") (Join-Path $App "run.py")
New-Item -ItemType Directory -Force (Join-Path $App "src") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $App "vendor") | Out-Null
robocopy (Join-Path $Repo "src") (Join-Path $App "src") /E /XD __pycache__ .pytest_cache /XF *.pyc /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy src failed: $LASTEXITCODE" }
robocopy (Join-Path $Repo "vendor") (Join-Path $App "vendor") /E /XD __pycache__ bin-b10901-orig logs /XF *.pyc /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy vendor failed: $LASTEXITCODE" }
$Hot = Join-Path $Repo "hotwords.txt"
if (Test-Path $Hot) {
    Copy-Item $Hot (Join-Path $App "hotwords.txt")
    Copy-Item $Hot (Join-Path $Out "hotwords.txt")
}
$EnvSrc = Join-Path $Repo ".env"
if (-not (Test-Path $EnvSrc)) {
    throw "Development .env not found: $EnvSrc"
}
Copy-Item $EnvSrc (Join-Path $Out ".env")

foreach ($d in @("models", "logs", "outputs")) {
    New-Item -ItemType Directory -Force (Join-Path $Out $d) | Out-Null
}

$Readme = @"
Qwen3 ASR (green pack)

1. Copy GGUF + ONNX models into models\
2. Double-click ASRTray.exe
3. Logs: logs\service.log and logs\server.log
"@
Set-Content -Path (Join-Path $Out "README.txt") -Value $Readme -Encoding utf8

Write-Host "Done: $Out"
Write-Host "Zip this folder for others. Do not zip the git repo."
