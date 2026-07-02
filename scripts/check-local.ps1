param(
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Continue"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EnvFile = Join-Path $Root "backend\.env"

function Show-Check {
  param([string]$Name, [bool]$Ok, [string]$Detail = "")
  $status = if ($Ok) { "OK" } else { "WARN" }
  Write-Host ("[{0}] {1} {2}" -f $status, $Name, $Detail)
}

Show-Check ".venv" (Test-Path (Join-Path $Root ".venv\Scripts\python.exe"))
Show-Check "frontend node_modules" (Test-Path (Join-Path $Root "frontend\node_modules"))
Show-Check "backend .env" (Test-Path $EnvFile)

if (Test-Path $EnvFile) {
  $interesting = @("LLM_PROVIDER", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL", "MYSQL_HOST", "MYSQL_PORT", "MYSQL_USER", "MYSQL_DB")
  Get-Content $EnvFile | ForEach-Object {
    if ($_ -match "^([^#=]+)=(.*)$") {
      $key = $matches[1].Trim()
      $value = $matches[2].Trim()
      if ($interesting -contains $key) {
        Write-Host "  $key=$value"
      } elseif ($key -match "KEY|PASSWORD") {
        Write-Host "  $key=<configured, hidden>"
      }
    }
  }
}

$mysql = Get-Service -Name MySQL* -ErrorAction SilentlyContinue | Select-Object -First 1
Show-Check "MySQL service" ($mysql -and $mysql.Status -eq "Running") $(if ($mysql) { "$($mysql.Name) $($mysql.Status)" } else { "not found" })

foreach ($port in @($BackendPort, $FrontendPort)) {
  $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
  Show-Check "port $port listening" ($null -ne $conn)
  if ($conn) {
    $owners = $conn | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($ownerPid in $owners) {
      $proc = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
      if ($proc) {
        Write-Host "  PID $ownerPid $($proc.ProcessName)"
      }
    }
  }
}

try {
  $health = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/api/health" -TimeoutSec 8
  Show-Check "backend /api/health" $true ($health.status | Out-String).Trim()
} catch {
  Show-Check "backend /api/health" $false $_.Exception.Message
}

try {
  $summary = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/api/stats/summary" -TimeoutSec 8
  Show-Check "backend /api/stats/summary" $true ("patients={0}, assessments={1}" -f $summary.patient_count, $summary.assessment_count)
} catch {
  Show-Check "backend /api/stats/summary" $false $_.Exception.Message
}

try {
  $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$FrontendPort" -TimeoutSec 8
  Show-Check "frontend page" ($response.StatusCode -eq 200) ("HTTP {0}" -f $response.StatusCode)
} catch {
  Show-Check "frontend page" $false $_.Exception.Message
}
