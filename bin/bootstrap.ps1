param(
    [switch]$Check,
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

function Resolve-AppDir {
    if ($env:NCAD_APP_DIR) {
        return (Resolve-Path $env:NCAD_APP_DIR).Path
    }
    $scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
    return (Resolve-Path (Join-Path $scriptDir "..")).Path
}

$AppDir = Resolve-AppDir
$CadRoot = if ($env:NCAD_CAD_ROOT) { $env:NCAD_CAD_ROOT } else { Join-Path $HOME "Documents\CAD" }
$EngineVenv = if ($env:NCAD_ENGINE_VENV) { $env:NCAD_ENGINE_VENV } else { Join-Path $CadRoot ".venv" }
$SolverVenv = if ($env:NCAD_SOLVER_VENV) { $env:NCAD_SOLVER_VENV } else { Join-Path $AppDir ".venv" }
$PythonVersion = if ($env:NCAD_PYTHON_VERSION) { $env:NCAD_PYTHON_VERSION } else { "3.12" }
$Uv = if ($env:UV) { $env:UV } else { "uv" }

function Venv-Python($Venv) {
    Join-Path $Venv "Scripts\python.exe"
}

function Venv-Tool($Venv, $Tool) {
    Join-Path $Venv "Scripts\$Tool.exe"
}

function Test-NcadEnvs {
    $enginePython = Venv-Python $EngineVenv
    $solverPython = Venv-Python $SolverVenv
    $agentcad = Venv-Tool $EngineVenv "agentcad"
    if (-not (Test-Path $enginePython)) { return $false }
    if (-not (Test-Path $solverPython)) { return $false }
    if (-not (Test-Path $agentcad)) { return $false }
    & $enginePython -c "import agentcad, build123d" | Out-Null
    & $solverPython -c "import numpy, scipy, skimage, matplotlib" | Out-Null
    return $true
}

if ($Check) {
    if (Test-NcadEnvs) { return }
    throw "nCAD environments are not ready."
}

if (-not (Get-Command $Uv -ErrorAction SilentlyContinue)) {
    throw "nCAD: uv is required to bootstrap Python environments. Install uv first."
}

New-Item -ItemType Directory -Force -Path $CadRoot | Out-Null

if ($Recreate) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $EngineVenv, $SolverVenv
}

function Ensure-Venv($Venv, $Prompt) {
    if (-not (Test-Path (Venv-Python $Venv))) {
        & $Uv venv --python $PythonVersion --prompt $Prompt $Venv
    }
}

function Install-Requirements($Venv, $Requirements) {
    & $Uv pip install --python (Venv-Python $Venv) -r $Requirements
}

Write-Host "nCAD: using app dir $AppDir"
Write-Host "nCAD: using CAD root $CadRoot"

Ensure-Venv $EngineVenv "nCAD-engine"
Install-Requirements $EngineVenv (Join-Path $AppDir "requirements-engine.txt")

Ensure-Venv $SolverVenv "nCAD-solver"
Install-Requirements $SolverVenv (Join-Path $AppDir "requirements-solver.txt")

try {
    Install-Requirements $SolverVenv (Join-Path $AppDir "requirements-solver-optional.txt")
} catch {
    Write-Warning "nCAD: optional scikit-sparse install failed; scipy fallback will be used."
}

if (-not (Test-NcadEnvs)) {
    throw "nCAD: bootstrap completed but environment verification failed."
}

Write-Host "nCAD: bootstrap complete."
