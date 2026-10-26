param(
    [ValidateSet('all', 'prepare', 'select', 'infer', 'score', 'validate')]
    [string]$Stage = 'all',
    [ValidateSet('auto', 'cuda', 'cpu')]
    [string]$Device = 'auto',
    [ValidateSet('4B', '8B')]
    [string]$AnswerModel = '8B',
    [switch]$UseExistingVllm
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = 'H:\anaconda\envs\remem\python.exe'
$raw = Join-Path $root 'VideoMME\val_qa_raw_full.json'
$prepared = Join-Path $root 'VideoMME\val_qa_full.json'
$selected = Join-Path $root 'VideoMME\val_qa_selected_full_32f.json'
$answerSuffix = $AnswerModel.ToLowerInvariant()
$predictions = Join-Path $root "VideoMME\vllm_predictions_full_32f_${answerSuffix}_lmms_prompt.json"
$metrics = Join-Path $root "VideoMME\vllm_metrics_full_32f_${answerSuffix}_lmms_prompt.json"
$expected = 2700

function Invoke-Python([string[]]$Arguments) {
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python exited with $LASTEXITCODE" }
}

function Get-RecordCount([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    return @((Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json)).Count
}

function Assert-Count([string]$Path, [int]$Count) {
    $actual = Get-RecordCount $Path
    if ($actual -ne $Count) { throw "Expected $Count records in $Path; found $actual" }
}

if (-not (Test-Path -LiteralPath $python)) { throw "Conda Python not found: $python" }
Push-Location $root
try {
    $videoCount = @(Get-ChildItem '.\VideoMME\videos' -Recurse -File -Filter '*.mp4').Count
    if ($videoCount -ne 900) { throw "Expected 900 MP4 videos; found $videoCount" }
    if (-not (Test-Path '.\VideoMME\test-00000-of-00001.parquet')) { throw 'Video-MME parquet annotation is missing' }
    Write-Host "Video-MME: $videoCount videos, expected $expected questions"

    if ($Stage -eq 'validate') {
        Write-Host "Raw: $(Get-RecordCount $raw)/$expected"
        Write-Host "Prepared: $(Get-RecordCount $prepared)/$expected"
        Write-Host "Selected: $(Get-RecordCount $selected)/$expected"
        Write-Host "Predictions: $(Get-RecordCount $predictions)/$expected"
        return
    }

    if ($Stage -in @('all', 'prepare', 'select')) {
        if ((Get-RecordCount $raw) -ne $expected) {
            Write-Host '[1/4] Converting all annotations...'
            Invoke-Python -Arguments @('convert_videomme.py', '.\VideoMME\test-00000-of-00001.parquet', '.\VideoMME\videos', $raw)
        }
        Assert-Count $raw $expected
    }

    if ($Stage -in @('all', 'prepare', 'select')) {
        if ((Get-RecordCount $prepared) -ne $expected) {
            $served = @((Invoke-RestMethod 'http://127.0.0.1:8000/v1/models' -TimeoutSec 5).data.id)
            if ('qwen3-vl-4b-instruct' -notin $served) { throw 'Question parsing requires the Sentrix 4B profile on port 8000' }
            $env:REMEM_LLM_BASE_URL = 'http://127.0.0.1:8000/v1'
            $env:REMEM_LLM_MODEL = 'qwen3-vl-4b-instruct'
            $env:OPENAI_API_KEY = 'EMPTY'
            Write-Host '[2/4] Parsing all questions with the local 4B model (resumable)...'
            Invoke-Python -Arguments @('prepare_annotations.py', $raw, $prepared)
        }
        Assert-Count $prepared $expected
    }
    if ($Stage -eq 'prepare') { return }

    if ($Stage -in @('all', 'select')) {
        if ((Get-RecordCount $selected) -ne $expected) {
            Write-Host '[3/4] Selecting 32 frames per question (resumable)...'
            Invoke-Python -Arguments @('frame_select.py', '--labels', $prepared, '--video-root', '.\VideoMME\videos', '--output', $selected, '--num-frames', '32', '--device', $Device)
        }
        Assert-Count $selected $expected
    }
    if ($Stage -eq 'select') { return }

    if ($Stage -in @('all', 'infer')) {
        Assert-Count $selected $expected
        $changedProfile = $false
        try {
            if ($UseExistingVllm) {
                $expectedModel = if ($AnswerModel -eq '4B') { 'qwen3-vl-4b-instruct' } else { 'qwen3-vl-8b-instruct' }
                $served = (Invoke-RestMethod 'http://127.0.0.1:8000/v1/models' -TimeoutSec 10).data
                if (-not @($served | Where-Object { $_.id -eq $expectedModel -and $_.max_model_len -eq 32768 }).Count) {
                    throw "Port 8000 must serve $expectedModel with context 32768 before inference"
                }
                Write-Host "Using running $expectedModel vLLM server (context 32768)."
            }
            else {
            $changedProfile = $true
            $profile = if ($AnswerModel -eq '4B') { 'ReMem32' } else { 'ReMem8B' }
            for ($attempt = 1; $attempt -le 2; $attempt++) {
                try {
                    & "$root\switch_vllm_profile.ps1" -Profile $profile
                    if ($LASTEXITCODE -ne 0) { throw "Failed to start $profile vLLM profile" }
                    break
                }
                catch {
                    & wsl.exe --terminate Ubuntu-22.04-vllm-backup | Out-Null
                    if ($attempt -eq 2) { throw }
                    Write-Warning "$profile startup failed; retrying from a clean WSL instance"
                    Start-Sleep -Seconds 5
                }
            }
            }
            $env:REMEM_LLM_BASE_URL = 'http://127.0.0.1:8000/v1'
            $env:REMEM_LLM_MODEL = if ($AnswerModel -eq '4B') { 'qwen3-vl-4b-instruct' } else { 'qwen3-vl-8b-instruct' }
            $env:OPENAI_API_KEY = 'EMPTY'
            Write-Host "[4/4] Evaluating with Qwen3-VL-$AnswerModel (resumable)..."
            Invoke-Python -Arguments @('infer_vllm.py', '--labels', $selected, '--video-root', '.\VideoMME\videos', '--output', $predictions, '--max-frames', '32', '--protocol', 'videomme_qwen3_vl', '--resume')
        }
        finally {
            if ($changedProfile) { & "$root\restore_sentrix_vllm.ps1" }
        }
        Assert-Count $predictions $expected
    }

    if ($Stage -in @('all', 'infer', 'score')) {
        Assert-Count $predictions $expected
        Write-Host 'Scoring overall, short, medium, and long accuracy...'
        Invoke-Python -Arguments @('score_predictions.py', '--labels', $prepared, '--predictions', $predictions, '--output', $metrics)
        Write-Host "Metrics: $metrics"
        Write-Host "Predictions: $predictions"
    }
}
finally {
    Pop-Location
}
