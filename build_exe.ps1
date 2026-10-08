param(
    [string]$PythonExe = "C:\Users\admin\AppData\Local\Programs\Python\Python312\python.exe"
)

$ErrorActionPreference = 'Stop'
$ProjectDir = $PSScriptRoot
$BuildEnv = Join-Path $ProjectDir '.exe-build-venv'
$DistDir = Join-Path $ProjectDir 'dist'
$ReleaseName = 'EISScanGUI-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$GuiDistDir = Join-Path $DistDir $ReleaseName
$TempDistDir = Join-Path ([System.IO.Path]::GetTempPath()) ('EISScanGUI-dist-' + [guid]::NewGuid().ToString('N'))
$WorkDir = Join-Path ([System.IO.Path]::GetTempPath()) ('EISScanGUI-build-' + [guid]::NewGuid().ToString('N'))

if (-not (Test-Path -LiteralPath $PythonExe)) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $PythonCommand) {
        throw 'Python 3.10+ was not found. Install Python or pass -PythonExe <python.exe>.'
    }
    $PythonExe = $PythonCommand.Source
}

Write-Host "Creating build environment with $PythonExe"
if (-not (Test-Path -LiteralPath (Join-Path $BuildEnv 'Scripts\python.exe'))) {
    & $PythonExe -m venv $BuildEnv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the build environment.' }
}

$VenvPython = Join-Path $BuildEnv 'Scripts\python.exe'
& $VenvPython -m pip show pyinstaller *> $null
if ($LASTEXITCODE -ne 0) {
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'Could not upgrade pip.' }
}
& $VenvPython -m pip install -r (Join-Path $ProjectDir 'requirements-exe.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install packaging dependencies.' }

# Icons. NOTE: keep this block ASCII-only. PowerShell 5.1 reads .ps1 as the
# system ANSI codepage, so non-ASCII comments here get mangled and can swallow
# the statements that follow (this actually happened: $AppIcon ended up null).
# .ico goes into the exe via --icon; both files ship via --add-data so the app
# can attach the titlebar/taskbar icon at runtime (see eis_gui._set_window_icon).
$AppIcon = Join-Path $ProjectDir 'assets\app_icon.ico'
$UIIconPng = Join-Path $ProjectDir 'assets\app_icon_ui.png'
$AssetsDir = Join-Path $ProjectDir 'assets'
if (-not (Test-Path -LiteralPath $AppIcon)) {
    throw "Missing app icon: $AppIcon (run: python make_icon.py)"
}
if (-not (Test-Path -LiteralPath $UIIconPng)) {
    throw "Missing UI icon: $UIIconPng (run: python make_icon.py)"
}

Push-Location $ProjectDir
try {
    # PyInstaller writes its INFO logs to stderr. With $ErrorActionPreference='Stop'
    # PowerShell treats those normal logs as terminating errors and aborts the build.
    # Relax it for this call and rely on $LASTEXITCODE instead.
    $ErrorActionPreference = 'Continue'
    & $VenvPython -m PyInstaller `
        --noconfirm `
        --clean `
        --onedir `
        --windowed `
        --name EISScanGUI `
        --icon $AppIcon `
        --add-data "$AssetsDir;assets" `
        --distpath $TempDistDir `
        --workpath $WorkDir `
        --hidden-import eis_main `
        --hidden-import drivers.hp4284a `
        --hidden-import drivers.simulator `
        --hidden-import drivers.relay_board `
        --hidden-import cal.calibration `
        --hidden-import measurements.eis `
        --hidden-import analysis.fitting `
        --hidden-import analysis.auto_fit `
        --hidden-import utils.data_io `
        --hidden-import serial `
        --hidden-import pyvisa_py `
        --collect-all pyvisa `
        --collect-all pyvisa_py `
        --collect-all dearpygui `
        --exclude-module matplotlib.tests `
        --exclude-module pandas.tests `
        --exclude-module pyvisa.tests `
        eis_gui.py 2>&1
    $PyInstallerExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($PyInstallerExit -ne 0) { throw 'PyInstaller failed.' }
}
finally {
    Pop-Location
}

Copy-Item -LiteralPath (Join-Path $TempDistDir 'EISScanGUI') `
    -Destination $GuiDistDir -Recurse
Copy-Item -LiteralPath (Join-Path $ProjectDir 'eis_config.yaml') `
    -Destination (Join-Path $GuiDistDir 'eis_config.yaml') -Force
Compress-Archive -Path $GuiDistDir `
    -DestinationPath (Join-Path $DistDir "$ReleaseName-win64.zip") -Force

$TempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
foreach ($TempPath in @($WorkDir, $TempDistDir)) {
    $ResolvedTempPath = (Resolve-Path -LiteralPath $TempPath).Path
    if (-not $ResolvedTempPath.StartsWith($TempRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Packaging path escaped the temporary directory.'
    }
    Remove-Item -LiteralPath $ResolvedTempPath -Recurse -Force
}

Write-Host "Build complete: $(Join-Path $GuiDistDir 'EISScanGUI.exe')"
Write-Host "ZIP package: $(Join-Path $DistDir "$ReleaseName-win64.zip")"
Write-Host 'The editable eis_config.yaml is beside the executable.'
