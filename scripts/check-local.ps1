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
  $interesting = @(
    "APP_ADMIN_USER",
    "LLM_PROVIDER",
    "LLM_REMOTE_URL",
    "LLM_REMOTE_TIMEOUT",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_USER",
    "MYSQL_DB"
  )
  Get-Content $EnvFile | ForEach-Object {
    if ($_ -match "^([^#=]+)=(.*)$") {
      $key = $matches[1].Trim()
      $value = $matches[2].Trim()
      if ($interesting -contains $key) {
        Write-Host "  $key=$value"
      } elseif ($key -match "KEY|PASSWORD|TOKEN") {
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

$authHeaders = @{}
try {
  if (-not (Test-Path $EnvFile)) {
    throw "backend .env not found"
  }
  $envMap = @{}
  Get-Content $EnvFile | ForEach-Object {
    if ($_ -match "^([^#=]+)=(.*)$") {
      $envMap[$matches[1].Trim()] = $matches[2].Trim()
    }
  }
  if (-not $envMap["APP_ADMIN_USER"] -or -not $envMap["APP_ADMIN_PASSWORD"]) {
    throw "APP_ADMIN_USER or APP_ADMIN_PASSWORD is missing"
  }
  $body = @{
    username = $envMap["APP_ADMIN_USER"]
    password = $envMap["APP_ADMIN_PASSWORD"]
  } | ConvertTo-Json
  $login = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/api/auth/login" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 8
  if (-not $login.access_token) {
    throw "login response has no access_token"
  }
  $authHeaders = @{ Authorization = "Bearer $($login.access_token)" }
  Show-Check "backend /api/auth/login" $true ("user={0}" -f $login.user)
} catch {
  Show-Check "backend /api/auth/login" $false $_.Exception.Message
}

try {
  $summary = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/api/stats/summary" -Headers $authHeaders -TimeoutSec 8
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
