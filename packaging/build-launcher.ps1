param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Deps = Join-Path $Repo "build\tray-deps"
$PreviousPythonPath = $env:PYTHONPATH
try {
    if (Test-Path $Deps) { $env:PYTHONPATH = "$Deps;$Deps\win32;$Deps\win32\lib;$Deps\pywin32_system32;$PreviousPythonPath" }
    & $Python -m PyInstaller --noconfirm --clean --onefile --windowed --name ASR --paths (Join-Path $Repo "src") --hidden-import pystray._win32 --distpath $Repo --workpath (Join-Path $Repo "build\launcher") --specpath (Join-Path $Repo "build") (Join-Path $Repo "src\tray_app.py")
    if ($LASTEXITCODE -ne 0) { throw "Launcher build failed" }
} finally {
    $env:PYTHONPATH = $PreviousPythonPath
}
