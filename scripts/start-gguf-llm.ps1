param(
  [string]$ModelPath = "$env:USERPROFILE\Downloads\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
  [int]$Port = 6006,
  [int]$NCtx = 12288,
  [int]$GpuLayers = -1,
  [int]$Threads = 0,
  [int]$MaxTokens = 4096,
  [double]$Temperature = 0.0
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = Join-Path $Root ".venv\Scripts\python.exe"
$StateDir = Join-Path $Root ".cache\gguf-llm"
$OutLog = Join-Path $StateDir "gguf.out.log"
$ErrLog = Join-Path $StateDir "gguf.err.log"
$PidFile = Join-Path $StateDir "gguf.pid"
$TorchLib = Join-Path $Root ".venv\Lib\site-packages\torch\lib"
$LlamaLib = Join-Path $Root ".venv\Lib\site-packages\llama_cpp\lib"

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

if (-not (Test-Path $PythonExe)) {
  throw "Missing Python venv: $PythonExe"
}

if (-not (Test-Path $ModelPath)) {
  throw "Missing GGUF first split: $ModelPath"
}

$SecondSplit = $ModelPath -replace "-00001-of-00002\.gguf$", "-00002-of-00002.gguf"
if ($SecondSplit -ne $ModelPath -and -not (Test-Path $SecondSplit)) {
  throw "Missing GGUF second split: $SecondSplit"
}

if (-not (Test-Path $LlamaLib)) {
  throw "llama-cpp-python is not installed in .venv. See requirements-gguf-server.txt"
}

if (Test-PortListening -Port $Port) {
  Write-Host "GGUF LLM port $Port is already listening. Existing process:"
  Show-PortOwner -Port $Port
  exit 0
}

if ($Threads -le 0) {
  $Threads = [Math]::Max(1, [Environment]::ProcessorCount - 2)
}

$env:PATH = "$TorchLib;$LlamaLib;$env:PATH"
$env:LLM_GGUF_MODEL_PATH = $ModelPath
$env:LLM_GGUF_SERVER_PORT = "$Port"
$env:LLM_GGUF_N_CTX = "$NCtx"
$env:LLM_GGUF_N_GPU_LAYERS = "$GpuLayers"
$env:LLM_GGUF_N_THREADS = "$Threads"
$env:LLM_GGUF_MAX_TOKENS = "$MaxTokens"
$env:LLM_GGUF_TEMPERATURE = "$Temperature"
$env:LLM_GGUF_VERBOSE = "0"

Write-Host "Starting GGUF LLM service on http://127.0.0.1:$Port ..."
Write-Host "  model : $ModelPath"
Write-Host "  n_ctx : $NCtx"
Write-Host "  gpu   : n_gpu_layers=$GpuLayers"
Write-Host "  logs  : $StateDir"

$proc = Start-Process `
  -FilePath $PythonExe `
  -ArgumentList @("llm_gguf_server.py") `
  -WorkingDirectory $Root `
  -RedirectStandardOutput $OutLog `
  -RedirectStandardError $ErrLog `
  -PassThru `
  -WindowStyle Hidden

Set-Content -Path $PidFile -Value $proc.Id -Encoding ASCII
Write-Host "Started PID $($proc.Id). Run scripts\check-gguf-llm.ps1 to verify."
