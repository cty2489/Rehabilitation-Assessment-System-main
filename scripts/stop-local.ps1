param(
  [switch]$StopPortProcesses,
  [int[]]$Ports = @(8000, 5173)
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StateDir = Join-Path $Root ".cache\local-deploy"
$PidFiles = @(
  (Join-Path $StateDir "backend.pid"),
  (Join-Path $StateDir "frontend.pid")
)

foreach ($pidFile in $PidFiles) {
  if (-not (Test-Path $pidFile)) {
    continue
  }
  $pidText = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if (-not $pidText) {
    Remove-Item -LiteralPath $pidFile -Force
    continue
  }
  $proc = Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
  if ($proc) {
    Write-Host "Stopping PID $($proc.Id) ($($proc.ProcessName)) from $pidFile"
    Stop-Process -Id $proc.Id -Force
  }
  Remove-Item -LiteralPath $pidFile -Force
}

if ($StopPortProcesses) {
  foreach ($port in $Ports) {
    $owners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
      Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($owner in $owners) {
      $proc = Get-Process -Id $owner -ErrorAction SilentlyContinue
      if ($proc) {
        Write-Host "Stopping port $port owner PID $owner ($($proc.ProcessName))"
        Stop-Process -Id $owner -Force
      }
    }
  }
}

Write-Host "Done."
