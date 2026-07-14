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
$HostName = if ($env:NCAD_HOST) { $env:NCAD_HOST } else { "127.0.0.1" }
$Port = if ($env:NCAD_PORT) { $env:NCAD_PORT } else { "8765" }
$Url = "http://${HostName}:${Port}"
$Python = Join-Path $EngineVenv "Scripts\python.exe"
$Server = Join-Path $AppDir "server.py"
$Log = if ($env:NCAD_LOG) { $env:NCAD_LOG } else { Join-Path $AppDir "server.log" }
$Bootstrap = Join-Path $AppDir "bin\bootstrap.ps1"

try {
    & $Bootstrap -Check
} catch {
    Write-Host "nCAD: Python environments need setup; running bootstrap..."
    & $Bootstrap
}

function Test-NcadServer {
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "$Url/api/jobs" -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

if (-not (Test-NcadServer)) {
    Write-Host "nCAD: starting server..."
    New-Item -ItemType Directory -Force -Path $CadRoot | Out-Null
    $env:NCAD_APP_DIR = $AppDir
    $env:NCAD_CAD_ROOT = $CadRoot
    $env:NCAD_ENGINE_VENV = $EngineVenv
    $cmd = "/c `"`"$Python`" `"$Server`" >> `"$Log`" 2>&1`""
    Start-Process -FilePath "cmd.exe" -ArgumentList $cmd -WorkingDirectory $AppDir -WindowStyle Hidden
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-NcadServer) { break }
    }
    if (-not (Test-NcadServer)) {
        throw "nCAD: server failed to start; see $Log"
    }
}

Write-Host "nCAD: $Url"
Start-Process $Url
