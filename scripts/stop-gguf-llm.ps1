param(
  [switch]$StopPortProcess,
  [int]$Port = 6006
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StateDir = Join-Path $Root ".cache\gguf-llm"
$PidFile = Join-Path $StateDir "gguf.pid"

if (Test-Path $PidFile) {
  $pidText = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($pidText) {
    $proc = Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
    if ($proc) {
      Write-Host "Stopping GGUF LLM PID $($proc.Id) ($($proc.ProcessName))"
      Stop-Process -Id $proc.Id -Force
    }
  }
  Remove-Item -LiteralPath $PidFile -Force
}

if ($StopPortProcess) {
  $owners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique
  foreach ($ownerPid in $owners) {
    $proc = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
    if ($proc) {
      Write-Host "Stopping port $Port owner PID $ownerPid ($($proc.ProcessName))"
      Stop-Process -Id $ownerPid -Force
    }
  }
}

Write-Host "Done."
