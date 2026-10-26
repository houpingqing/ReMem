<##
  Dedicated Qwen3-8B text-only vLLM launcher for ReMem annotation.
  It does not modify D:/sentrix-home-web/vllm.ps1.
##>
[CmdletBinding()]
param(
  [ValidateSet('start','stop','restart','status','logs')]
  [string]$Action = 'status',
  [int]$Tail = 60
)

$ErrorActionPreference = 'Stop'
$Distro = 'Ubuntu-22.04-vllm-backup'
$RuntimeRoot = '/home/hpq/vllm-runtime'
$ModelPath = '/mnt/d/ReMem/models/Qwen3-8B'
$ServedModelName = 'qwen3-8b'
$Port = 8000
$GpuMemoryUtilization = '0.90'
$MaxModelLen = '8192'
$MaxNumSeqs = '1'
$LoadFormat = 'auto'
$PidFile = 'D:\vllm-runtime\qwen3-text-vllm.windows.pid'
$LogPath = "\\wsl.localhost\$Distro\home\hpq\vllm-runtime\logs\qwen3-8b-text.log"
$Endpoint = "http://127.0.0.1:$Port/v1"

function Get-Models {
  try { (Invoke-RestMethod -Uri "$Endpoint/models" -TimeoutSec 5).data } catch { $null }
}
function Get-Launcher {
  if (-not (Test-Path -LiteralPath $PidFile)) { return $null }
  $raw = (Get-Content -LiteralPath $PidFile -Raw).Trim()
  if ($raw -notmatch '^\d+$') { return $null }
  Get-Process -Id ([int]$raw) -ErrorAction SilentlyContinue
}
function Stop-Text {
  $p = Get-Launcher
  if ($p) { try { & taskkill.exe /PID $p.Id /T /F 2>&1 | Out-Null } catch {} }
  Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
  try { & wsl.exe --terminate $Distro 2>$null | Out-Null } catch {}
  $deadline = (Get-Date).AddSeconds(20)
  do {
    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $listener) { break }
    Start-Sleep -Milliseconds 500
  } while ((Get-Date) -lt $deadline)
}
function Start-TextLauncher {
  $body = @'
set -eu
runtime_root="/home/hpq/vllm-runtime"
log_file="$runtime_root/logs/qwen3-8b-text.log"
mkdir -p "$runtime_root/logs" "$runtime_root/huggingface"
printf '\n===== QWEN3_TEXT_START %s pid=%s =====\n' "$(date -Is)" "$$" >> "$log_file"
export HF_HOME="$runtime_root/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
unset VLLM_USE_V1 VLLM_HOLD_PT_WEIGHTS VLLM_SKIP_RENDERER_WARMUP
export PYTHONPATH="$runtime_root/launcher-compat"
export CUDA_VISIBLE_DEVICES=0
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$runtime_root/.pycache-runtime"
export PYTHONSAFEPATH=1
rm -rf "$PYTHONPYCACHEPREFIX"
mkdir -p "$PYTHONPYCACHEPREFIX"
cd "$runtime_root"
exec "$runtime_root/.venv-vllm312/bin/python" -m vllm.entrypoints.openai.api_server \
  --model /mnt/d/ReMem/models/Qwen3-8B \
  --served-model-name qwen3-8b \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --load-format auto \
  --trust-remote-code \
  --enforce-eager \
  >> "$log_file" 2>&1
'@
  $si = [Diagnostics.ProcessStartInfo]::new()
  $si.FileName = 'wsl.exe'
  $si.Arguments = "-u hpq -d $Distro --cd $RuntimeRoot --exec /bin/bash -s"
  $si.UseShellExecute = $false
  $si.CreateNoWindow = $true
  $si.RedirectStandardInput = $true
  $p = [Diagnostics.Process]::new(); $p.StartInfo = $si
  if (-not $p.Start()) { throw 'Failed to start the Qwen3-8B WSL launcher.' }
  $p.StandardInput.Write($body); $p.StandardInput.Close()
  $p.Id | Set-Content -LiteralPath $PidFile -NoNewline
  Write-Host "Started Qwen3-8B text vLLM (launcher PID $($p.Id), GPU budget $GpuMemoryUtilization)." -ForegroundColor Cyan
  Write-Host "Endpoint: $Endpoint"
}
function Wait-Ready([int]$Seconds = 480) {
  $deadline = (Get-Date).AddSeconds($Seconds)
  do {
    if (Get-Models) { return $true }
    if (-not (Get-Launcher)) { return $false }
    Start-Sleep -Seconds 5
  } while ((Get-Date) -lt $deadline)
  return $false
}
switch ($Action) {
  'start' {
    $existing = Get-Models
    if ($existing) {
      $ids = ($existing | ForEach-Object id) -join ', '
      if ($ids -eq 'qwen3-8b') { Write-Host 'Qwen3-8B is already ready.'; break }
      Write-Host "Stopping current vLLM model(s): $ids" -ForegroundColor Yellow
      try { & 'D:\sentrix-home-web\vllm.ps1' stop } catch { Stop-Text }
    } else { Stop-Text }
    Start-TextLauncher
    if (-not (Wait-Ready)) {
      Write-Warning 'Qwen3-8B did not become ready; see qwen3-8b-text.log.'
      if (Test-Path -LiteralPath $LogPath) { Get-Content -LiteralPath $LogPath -Tail 80 }
      exit 1
    }
    Write-Host 'Qwen3-8B text vLLM is ready.' -ForegroundColor Green
  }
  'stop' { Stop-Text; Write-Host "vLLM stopped (port $Port)." }
  'restart' { Stop-Text; Start-Sleep 3; & $PSCommandPath start }
  'status' {
    $m = Get-Models
    if ($m) { $m | Select-Object id,root,max_model_len | Format-Table -AutoSize }
    else { Write-Host "vLLM is not ready on $Endpoint" -ForegroundColor Yellow }
    $l = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($l) { Write-Host "TCP $Port`: LISTEN (PID $($l[0].OwningProcess))" -ForegroundColor Green }
    else { Write-Host "TCP $Port`: not listening" -ForegroundColor Yellow }
  }
  'logs' { if (Test-Path -LiteralPath $LogPath) { Get-Content -LiteralPath $LogPath -Tail $Tail } else { Write-Host "Log not found: $LogPath" } }
}
