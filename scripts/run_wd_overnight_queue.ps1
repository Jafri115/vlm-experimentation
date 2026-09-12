param(
    [string]$LlmPython = ".\.venv-llm-ft\Scripts\python.exe",
    [string]$VlmPython = ".\.venv\Scripts\python.exe",
    [switch]$IncludeVlm,
    [int[]]$Folds = @(1, 2, 3, 4, 5)
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

function Resolve-ProjectPath([string]$PathValue) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    $Relative = $PathValue -replace '^[.][\\/]', ''
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Relative))
}

$LlmPython = Resolve-ProjectPath $LlmPython
$VlmPython = Resolve-ProjectPath $VlmPython
if (-not (Test-Path -LiteralPath $LlmPython)) {
    $FallbackPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $FallbackPython) {
        $LlmPython = $FallbackPython
    }
}
if (-not (Test-Path -LiteralPath $LlmPython)) {
    throw "LLM Python executable not found: $LlmPython"
}
if ($IncludeVlm -and -not (Test-Path -LiteralPath $VlmPython)) {
    throw "VLM Python executable not found: $VlmPython"
}

$RunRoot = Join-Path $ProjectRoot "output\wd_overnight_queue"
$LogRoot = Join-Path $RunRoot "logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$StatusPath = Join-Path $RunRoot "job_status.csv"
$QueueLog = Join-Path $RunRoot "queue.log"
$script:StatusRows = [System.Collections.Generic.List[object]]::new()

function Write-QueueMessage([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $QueueLog -Append
}

function Save-Status {
    $script:StatusRows | Export-Csv -LiteralPath $StatusPath -NoTypeInformation -Encoding utf8
}

function Invoke-QueuedJob {
    param(
        [string]$Name,
        [string]$Python,
        [string[]]$Arguments,
        [string]$CompletionFile
    )
    $CompletionFile = Resolve-ProjectPath $CompletionFile
    $LogPath = Join-Path $LogRoot ($Name + ".log")
    $Started = Get-Date

    if (Test-Path -LiteralPath $CompletionFile) {
        Write-QueueMessage "SKIP $Name (completion file exists)"
        $script:StatusRows.Add([pscustomobject]@{
            job = $Name; status = "SKIPPED_COMPLETE"; exit_code = 0
            started = $Started.ToString("o"); finished = (Get-Date).ToString("o")
            log = $LogPath; completion_file = $CompletionFile
        })
        Save-Status
        return
    }

    Write-QueueMessage "START $Name"
    "COMMAND: $Python $($Arguments -join ' ')" | Out-File -LiteralPath $LogPath -Encoding utf8
    & $Python @Arguments *>> $LogPath
    $ExitCode = $LASTEXITCODE
    $Finished = Get-Date
    $Status = if ($ExitCode -eq 0 -and (Test-Path -LiteralPath $CompletionFile)) {
        "COMPLETED"
    } elseif ($ExitCode -eq 0) {
        "MISSING_COMPLETION_FILE"
    } else {
        "FAILED"
    }
    $script:StatusRows.Add([pscustomobject]@{
        job = $Name; status = $Status; exit_code = $ExitCode
        started = $Started.ToString("o"); finished = $Finished.ToString("o")
        log = $LogPath; completion_file = $CompletionFile
    })
    Save-Status
    Write-QueueMessage "$Status $Name (exit $ExitCode)"
}

function Test-AllFoldFiles([string]$Root, [string]$Filename) {
    foreach ($Fold in $Folds) {
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "$Root\fold_$Fold\$Filename"))) {
            return $false
        }
    }
    return $true
}

Write-QueueMessage "WD overnight queue started"
Write-QueueMessage "Project: $ProjectRoot"
Write-QueueMessage "LLM Python: $LlmPython"
Write-QueueMessage "Include VLM: $IncludeVlm"

# Generative transcript experiments: zero-shot followed by 3+3 few-shot.
foreach ($Fold in $Folds) {
    Invoke-QueuedJob `
        -Name "llm_zero_fold_$Fold" `
        -Python $LlmPython `
        -Arguments @(
            "scripts/run_qwen3_8b_wd_zero_fewshot_aligned.py",
            "--dataset", "output/wd_multimodal_master/fold_$Fold/master_manifest.jsonl",
            "--shot", "zero",
            "--output", "output/llm_wd_zero_cv/fold_$Fold"
        ) `
        -CompletionFile "output/llm_wd_zero_cv/fold_$Fold/summary.json"
}

foreach ($Fold in $Folds) {
    Invoke-QueuedJob `
        -Name "llm_few_3plus3_fold_$Fold" `
        -Python $LlmPython `
        -Arguments @(
            "scripts/run_qwen3_8b_wd_zero_fewshot_aligned.py",
            "--dataset", "output/wd_multimodal_master/fold_$Fold/master_manifest.jsonl",
            "--shot", "few", "--examples-per-class", "3",
            "--output", "output/llm_wd_few_cv/fold_$Fold"
        ) `
        -CompletionFile "output/llm_wd_few_cv/fold_$Fold/summary.json"
}

# QLoRA transcript experiments: regression, consensus binary, then soft labels.
foreach ($Mode in @("regression", "consensus", "soft")) {
    foreach ($Fold in $Folds) {
        Invoke-QueuedJob `
            -Name "llm_${Mode}_fold_$Fold" `
            -Python $LlmPython `
            -Arguments @(
                "scripts/finetune_qwen3_8b_wd_text.py",
                "--dataset", "output/wd_multimodal_master/fold_$Fold/master_manifest.jsonl",
                "--mode", $Mode,
                "--output", "output/llm_wd_${Mode}_cv/fold_$Fold"
            ) `
            -CompletionFile "output/llm_wd_${Mode}_cv/fold_$Fold/final_summary.json"
    }
}

# Optional paired VLM reruns. These use the frozen slide labels and same folds.
if ($IncludeVlm) {
    foreach ($Mode in @("consensus", "soft")) {
        foreach ($Fold in $Folds) {
            Invoke-QueuedJob `
                -Name "vlm_${Mode}_fold_$Fold" `
                -Python $VlmPython `
                -Arguments @(
                    "scripts/finetune_qwen3vl_wd_consensus_binary.py",
                    "--labels-csv", "output/wd_multimodal_master/frozen_rater_labels_long.csv",
                    "--manifest", "output/wd_multimodal_master/fold_$Fold/master_manifest.csv",
                    "--target-mode", $Mode,
                    "--positive-threshold", "2",
                    "--frame-cache", "output/qwen3vl_wd_planning196_thr2/qwen3vl_wd_planning196_thr2/frame_cache_16",
                    "--output-dir", "output/vlm_wd_${Mode}_paired_cv/fold_$Fold"
                ) `
                -CompletionFile "output/vlm_wd_${Mode}_paired_cv/fold_$Fold/final_summary.json"
        }
    }
}

# Out-of-fold aggregation. Prompting scripts call their file predictions.csv;
# fine-tuning scripts call it test_predictions.csv.
foreach ($PromptRoot in @("llm_wd_zero_cv", "llm_wd_few_cv")) {
    if (Test-AllFoldFiles "output/$PromptRoot" "predictions.csv") {
        Invoke-QueuedJob `
            -Name "combine_$PromptRoot" `
            -Python $LlmPython `
            -Arguments @(
                "scripts/combine_wd_cv_predictions.py",
                "--fold-root", "output/$PromptRoot",
                "--filename", "predictions.csv",
                "--output", "output/$PromptRoot/oof_predictions.csv"
            ) `
            -CompletionFile "output/$PromptRoot/oof_predictions.csv"
    }
}

foreach ($ModelRoot in @(
    "llm_wd_regression_cv", "llm_wd_consensus_cv", "llm_wd_soft_cv",
    "vlm_wd_consensus_paired_cv", "vlm_wd_soft_paired_cv"
)) {
    if (Test-AllFoldFiles "output/$ModelRoot" "test_predictions.csv") {
        Invoke-QueuedJob `
            -Name "combine_$ModelRoot" `
            -Python $LlmPython `
            -Arguments @(
                "scripts/combine_wd_cv_predictions.py",
                "--fold-root", "output/$ModelRoot",
                "--output", "output/$ModelRoot/oof_predictions.csv"
            ) `
            -CompletionFile "output/$ModelRoot/oof_predictions.csv"
    }
}

if ($IncludeVlm) {
    foreach ($Mode in @("consensus", "soft")) {
        $VlmPredictions = "output/vlm_wd_${Mode}_paired_cv/oof_predictions.csv"
        $LlmPredictions = "output/llm_wd_${Mode}_cv/oof_predictions.csv"
        if ((Test-Path -LiteralPath (Resolve-ProjectPath $VlmPredictions)) -and
            (Test-Path -LiteralPath (Resolve-ProjectPath $LlmPredictions))) {
            Invoke-QueuedJob `
                -Name "compare_vlm_llm_$Mode" `
                -Python $LlmPython `
                -Arguments @(
                    "scripts/compare_vlm_llm_wd_predictions.py",
                    "--vlm-predictions", $VlmPredictions,
                    "--llm-predictions", $LlmPredictions,
                    "--output", "output/paired_comparison_$Mode"
                ) `
                -CompletionFile "output/paired_comparison_$Mode/comparison_summary.json"
        }
    }
}

Write-QueueMessage "WD overnight queue finished"
