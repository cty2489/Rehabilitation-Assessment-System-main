param(
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BackendDir = Join-Path $Root "backend"
$FrontendDir = Join-Path $Root "frontend"
$PythonExe = Join-Path $Root ".venv\Scripts\python.exe"
$EnvFile = Join-Path $BackendDir ".env"
$NodeModules = Join-Path $FrontendDir "node_modules"
$StateDir = Join-Path $Root ".cache\local-deploy"
$BackendOut = Join-Path $StateDir "backend.out.log"
$BackendErr = Join-Path $StateDir "backend.err.log"
$FrontendOut = Join-Path $StateDir "frontend.out.log"
$FrontendErr = Join-Path $StateDir "frontend.err.log"
$BackendPid = Join-Path $StateDir "backend.pid"
$FrontendPid = Join-Path $StateDir "frontend.pid"

New-Item -ItemType Directory -Force $StateDir | Out-Null

function Test-PortListening {
  param([int]$Port)
  $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  return $null -ne $conn
}

function Show-PortOwner {
  param([int]$Port)
  $owners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique
  foreach ($ownerPid in $owners) {
    $proc = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
    if ($proc) {
      Write-Host "  port $Port -> PID $ownerPid ($($proc.ProcessName))"
    }
  }
}

function Resolve-NpmCmd {
  $cmd = Get-Command npm.cmd -ErrorAction SilentlyContinue
  if ($cmd) {
    return $cmd.Source
  }

  $candidates = @(
    (Join-Path $env:ProgramFiles "nodejs\npm.cmd"),
    (Join-Path ${env:ProgramFiles(x86)} "nodejs\npm.cmd")
  )
  foreach ($candidate in $candidates) {
    if ($candidate -and (Test-Path $candidate)) {
      return $candidate
    }
  }

  $wingetRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
  if (Test-Path $wingetRoot) {
    $found = Get-ChildItem -Path $wingetRoot -Recurse -Filter npm.cmd -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -notmatch "\\node_modules\\" } |
      Select-Object -First 1
    if ($found) {
      return $found.FullName
    }
  }

  return $null
}

if (-not (Test-Path $PythonExe)) {
  throw "Missing Python venv: $PythonExe. Create it first, then install backend\requirements.txt."
}

if (-not (Test-Path $EnvFile)) {
  throw "Missing backend\.env. Copy backend\.env.example to backend\.env and fill DeepSeek/MySQL settings."
}

if (-not (Test-Path $NodeModules)) {
  throw "Missing frontend\node_modules. Run: cd frontend; npm install"
}

$mysql = Get-Service -Name MySQL* -ErrorAction SilentlyContinue | Select-Object -First 1
if ($mysql -and $mysql.Status -ne "Running") {
  Write-Host "Starting MySQL service: $($mysql.Name)"
  Start-Service -Name $mysql.Name
}

if (Test-PortListening -Port $BackendPort) {
  Write-Host "Backend port $BackendPort is already listening. Existing process:"
  Show-PortOwner -Port $BackendPort
} else {
  Write-Host "Starting backend on http://127.0.0.1:$BackendPort ..."
  $backend = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList @("-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$BackendPort") `
    -WorkingDirectory $BackendDir `
    -RedirectStandardOutput $BackendOut `
    -RedirectStandardError $BackendErr `
    -PassThru `
    -WindowStyle Hidden
  Set-Content -Path $BackendPid -Value $backend.Id -Encoding ASCII
}

if (Test-PortListening -Port $FrontendPort) {
  Write-Host "Frontend port $FrontendPort is already listening. Existing process:"
  Show-PortOwner -Port $FrontendPort
} else {
  $npm = Resolve-NpmCmd
  if (-not $npm) {
    throw "npm.cmd not found. Install Node.js LTS first."
  }
  $nodeDir = Split-Path $npm -Parent
  if ($env:PATH -notlike "*$nodeDir*") {
    $env:PATH = "$nodeDir;$env:PATH"
  }
  Write-Host "Starting frontend on http://127.0.0.1:$FrontendPort ..."
  $frontend = Start-Process `
    -FilePath $npm `
    -ArgumentList @("run", "dev", "--", "--host", "0.0.0.0", "--port", "$FrontendPort") `
    -WorkingDirectory $FrontendDir `
    -RedirectStandardOutput $FrontendOut `
    -RedirectStandardError $FrontendErr `
    -PassThru `
    -WindowStyle Hidden
  Set-Content -Path $FrontendPid -Value $frontend.Id -Encoding ASCII
}

Write-Host ""
Write-Host "Local deployment status:"
Write-Host "  frontend: http://127.0.0.1:$FrontendPort"
Write-Host "  backend : http://127.0.0.1:$BackendPort"
Write-Host "  logs    : $StateDir"
Write-Host ""
Write-Host "Run scripts\check-local.ps1 to verify APIs."
