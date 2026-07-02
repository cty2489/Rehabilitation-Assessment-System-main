param(
  [int]$Port = 6006,
  [switch]$Generate
)

$ErrorActionPreference = "Continue"

function Show-Check {
  param([string]$Name, [bool]$Ok, [string]$Detail = "")
  $status = if ($Ok) { "OK" } else { "WARN" }
  Write-Host ("[{0}] {1} {2}" -f $status, $Name, $Detail)
}

$conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
Show-Check "port $Port listening" ($null -ne $conn)
if ($conn) {
  $owners = $conn | Select-Object -ExpandProperty OwningProcess -Unique
  foreach ($ownerPid in $owners) {
    $proc = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
    if ($proc) {
      Write-Host "  PID $ownerPid $($proc.ProcessName)"
    }
  }
}

try {
  $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 20
  Show-Check "GGUF /health" ($health.status -eq "ok") ("model={0}; n_ctx={1}; gpu_layers={2}" -f $health.model_path, $health.n_ctx, $health.n_gpu_layers)
} catch {
  Show-Check "GGUF /health" $false $_.Exception.Message
}

if ($Generate) {
  try {
    $body = @{
      messages = @(
        @{ role = "system"; content = "You are a concise assistant." },
        @{ role = "user"; content = "Reply in one short sentence: is the local GGUF model service available?" }
      )
      max_tokens = 48
      temperature = 0
    } | ConvertTo-Json -Depth 8
    $resp = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/generate_messages" -ContentType "application/json; charset=utf-8" -Body $body -TimeoutSec 120
    if ($resp.error) {
      Show-Check "GGUF /generate_messages" $false $resp.error
    } else {
      Show-Check "GGUF /generate_messages" $true $resp.text
    }
  } catch {
    Show-Check "GGUF /generate_messages" $false $_.Exception.Message
  }
}
