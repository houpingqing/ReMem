<##
  Qwen3-8B text annotation -> ReMem frame selection -> Qwen3-VL-8B answer.
  This runner waits for the non-quantized text model download, resumes each
  stage by question_id, and restores the Sentrix launcher in finally.
##>
[CmdletBinding()]
param(
  [switch]$ReprepareAnnotations
)
$ErrorActionPreference = 'Stop'
$Root = 'D:\ReMem'
Set-Location $Root
$Python = 'H:\anaconda\envs\remem\python.exe'
$TextModel = Join-Path $Root 'models\Qwen3-8B'
$Expected = @{
  'model-00001-of-00005.safetensors' = 3996250744
  'model-00002-of-00005.safetensors' = 3993160032
  'model-00003-of-00005.safetensors' = 3959604768
  'model-00004-of-00005.safetensors' = 3187841392
  'model-00005-of-00005.safetensors' = 1244659840
  'tokenizer.json' = 11422654
}
$Log = Join-Path $Root 'qwen3_text_vl8_eval.log'
$ParserLabels = Join-Path $Root 'VideoMME\val_qa_full_qwen3_8b.json'
$SelectedLabels = Join-Path $Root 'VideoMME\val_qa_selected_full_32f_qwen3_8b.json'
$Predictions = Join-Path $Root 'VideoMME\vllm_predictions_full_32f_parser_qwen3_8b_answer_vl8b.json'
$Metrics = Join-Path $Root 'VideoMME\vllm_metrics_full_32f_parser_qwen3_8b_answer_vl8b_metrics.json'
$LmmsPredictions = Join-Path $Root 'VideoMME\vllm_predictions_full_32f_parser_qwen3_8b_answer_vl8b_lmms.json'
$LmmsMetrics = Join-Path $Root 'VideoMME\vllm_metrics_full_32f_parser_qwen3_8b_answer_vl8b_lmms.json'

function Log([string]$s) { $line = "$(Get-Date -Format s) $s"; Write-Host $line; Add-Content -LiteralPath $Log -Value $line }
function Run-Python([string[]]$PyArgs) {
  Log ("python " + ($PyArgs -join ' '))
  & $Python @PyArgs
  if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw "Python exited with $LASTEXITCODE" }
}
function Wait-Weights {
  while ($true) {
    $missing = @()
    foreach ($kv in $Expected.GetEnumerator()) {
      $p = Join-Path $TextModel $kv.Key
      if (-not (Test-Path -LiteralPath $p) -or (Get-Item -LiteralPath $p).Length -ne [int64]$kv.Value) { $missing += $kv.Key }
    }
    if ($missing.Count -eq 0) { Log 'Qwen3-8B non-quantized files are complete.'; return }
    $sizes = (Get-ChildItem -LiteralPath $TextModel -Filter 'model-*.safetensors' -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    Log ("Waiting for model download; current weight bytes $sizes / 16381470720; missing $($missing -join ', ')")
    Start-Sleep -Seconds 60
  }
}

$env:OPENAI_API_KEY = 'EMPTY'
$env:REMEM_LLM_BASE_URL = 'http://127.0.0.1:8000/v1'
$changedService = $false
try {
  Wait-Weights

  $parserReady = $false
  $selectedReady = $false
  if (Test-Path -LiteralPath $ParserLabels) {
    try { $parserReady = (@(Get-Content -LiteralPath $ParserLabels -Raw | ConvertFrom-Json).Count -eq 2700) } catch { $parserReady = $false }
  }
  if (Test-Path -LiteralPath $SelectedLabels) {
    try {
      $selectedRows = @(Get-Content -LiteralPath $SelectedLabels -Raw | ConvertFrom-Json)
      $selectedReady = (($selectedRows | ForEach-Object { [string]$_.question_id } | Sort-Object -Unique).Count -eq 2700)
    } catch { $selectedReady = $false }
  }
  if ($ReprepareAnnotations -or -not $parserReady -or -not $selectedReady) {
    Log 'Preparing missing Qwen3-8B annotations/frames.'
    Log 'Starting Qwen3-8B text vLLM for annotation.'
    & (Join-Path $Root 'start_qwen3_8b_text.ps1') start
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw "Qwen3-8B text vLLM start failed ($LASTEXITCODE)" }
    $changedService = $true
    $env:REMEM_LLM_MODEL = 'qwen3-8b'
    Run-Python @('prepare_annotations.py','VideoMME\val_qa_raw_full.json','VideoMME\val_qa_full_qwen3_8b.json')

    Log 'Stopping text vLLM before frame selection.'
    & (Join-Path $Root 'start_qwen3_8b_text.ps1') stop
    Log 'Selecting exactly 32 frames from Qwen3-8B annotations.'
    $env:REMEM_VIDEO_DECODER = 'ffmpeg'
    Run-Python @('frame_select.py','--labels',$ParserLabels,'--video-root','VideoMME\videos','--output',$SelectedLabels,'--num-frames','32','--device','cuda')
  } else {
    Log 'Reusing complete Qwen3-8B annotations and selected frame_idx; no parser/selector rerun.'
  }

  Log 'Starting Qwen3-VL-8B answer vLLM.'
  & (Join-Path $Root 'switch_vllm_profile.ps1') -Profile ReMem8B
  if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw "Qwen3-VL-8B vLLM start failed ($LASTEXITCODE)" }
  $env:REMEM_LLM_MODEL = 'qwen3-vl-8b-instruct'
  Log 'Running lmms-eval-compatible Qwen3-VL message_format adapter with FFmpeg frames.'
  Run-Python @('infer_lmms_qwen3_vl.py','--labels',$SelectedLabels,'--video-root','VideoMME\videos','--output',$LmmsPredictions,'--max-frames','32','--resume')
  Run-Python @('score_lmms_videomme.py','--labels',$ParserLabels,'--predictions',$LmmsPredictions,'--output',$LmmsMetrics)
  Log "lmms-eval-compatible evaluation complete: $LmmsMetrics"
}
finally {
  Log 'Restoring Sentrix vLLM profile.'
  try { & (Join-Path $Root 'restore_sentrix_vllm.ps1') } catch { Log "Sentrix restore failed: $($_.Exception.Message)" }
}
