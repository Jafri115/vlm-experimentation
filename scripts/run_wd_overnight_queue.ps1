param(
    [string]$LlmPython = ".\.venv-llm-ft\Scripts\python.exe",
    [string]$VlmPython = ".\.venv\Scripts\python.exe",
    [string]$VlmFrameCache = ".\output\qwen3vl_wd_planning196_thr2\frame_cache_16",
    [string]$MasterRoot = ".\output\wd_multimodal_master_repaired",
    [string]$CohortTag = "repaired",
    [switch]$IncludeVlm,
    [switch]$IncludeFewShot,
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
$VlmFrameCache = Resolve-ProjectPath $VlmFrameCache
$MasterRoot = Resolve-ProjectPath $MasterRoot
if ($CohortTag -notmatch '^[A-Za-z0-9_-]+$') { throw "Invalid CohortTag: $CohortTag" }
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

$RunRoot = Join-Path $ProjectRoot "output\wd_overnight_queue_$CohortTag"
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
    # Windows PowerShell can promote any native stderr output to a terminating
    # NativeCommandError when the queue uses ErrorActionPreference=Stop. Model
    # libraries routinely write progress and warnings to stderr, so temporarily
    # use Continue, capture both streams in the job log, and decide success from
    # the native exit code plus the expected completion artifact.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ExitCode = 1
    try {
        $ErrorActionPreference = "Continue"
        & $Python @Arguments *>> $LogPath
        $ExitCode = $LASTEXITCODE
    } catch {
        $_ | Out-String | Out-File -LiteralPath $LogPath -Append -Encoding utf8
        $ExitCode = 1
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
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
    if ($Status -ne "COMPLETED") {
        throw "Queue stopped after $Name ($Status). Inspect $LogPath"
    }
}

function Test-AllFoldFiles([string]$Root, [string]$Filename) {
    foreach ($Fold in $Folds) {
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "$Root\fold_$Fold\$Filename"))) {
            return $false
        }
    }
    return $true
}

function Test-PythonEnvironment {
    param(
        [string]$Name,
        [string]$Python,
        [string]$Code
    )
    $LogPath = Join-Path $LogRoot ("preflight_" + $Name + ".log")
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Python "-c" $Code *>> $LogPath
        $ExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ExitCode -ne 0) {
        Write-QueueMessage "PREFLIGHT FAILED: $Name Python environment; see $LogPath"
        throw "$Name Python environment preflight failed"
    }
    Write-QueueMessage "Preflight $Name Python environment passed"
}

Write-QueueMessage "WD overnight queue started"
Write-QueueMessage "Project: $ProjectRoot"
Write-QueueMessage "LLM Python: $LlmPython"
Write-QueueMessage "Include VLM: $IncludeVlm"
Write-QueueMessage "Master cohort: $MasterRoot"
Write-QueueMessage "Cohort tag: $CohortTag"

# Fail once with a clear message when the transferred experiment bundle is
# incomplete, instead of producing the same traceback for every fold.
$RequiredCommon = @(
    "scripts/run_qwen3_8b_wd_zero_fewshot_aligned.py",
    "scripts/run_qwen3_8b_wd_3rs_zeroshot.py",
    "scripts/finetune_qwen3_8b_wd_text.py",
    "scripts/plot_wd_training_validation.py",
    "scripts/combine_wd_cv_predictions.py",
    "scripts/compare_vlm_llm_wd_predictions.py",
    (Join-Path $MasterRoot "frozen_rater_labels_long.csv")
)
foreach ($Fold in $Folds) {
    $RequiredCommon += (Join-Path $MasterRoot "fold_$Fold\master_manifest.jsonl")
    $RequiredCommon += (Join-Path $MasterRoot "fold_$Fold\master_manifest.csv")
    $RequiredCommon += (Join-Path $MasterRoot "fold_$Fold\vlm_manifest.csv")
}
$MissingCommon = @(
    $RequiredCommon | Where-Object {
        -not (Test-Path -LiteralPath (Resolve-ProjectPath $_))
    }
)
if ($MissingCommon.Count -gt 0) {
    Write-QueueMessage "PREFLIGHT FAILED: missing required files"
    $MissingCommon | ForEach-Object { Write-QueueMessage "MISSING $_" }
    throw "Overnight experiment bundle is incomplete. See $QueueLog"
}

if ($IncludeVlm) {
    $RequiredVlm = @(
        "scripts/finetune_qwen3vl_wd_consensus_binary.py",
        "scripts/finetune_qwen3vl_rupture_pilot.py",
        $VlmFrameCache
    )
    $MissingVlm = @(
        $RequiredVlm | Where-Object {
            -not (Test-Path -LiteralPath (Resolve-ProjectPath $_))
        }
    )
    if ($MissingVlm.Count -gt 0) {
        Write-QueueMessage "VLM DISABLED: missing VLM files"
        $MissingVlm | ForEach-Object { Write-QueueMessage "MISSING $_" }
        Write-QueueMessage "LLM experiments will continue; restart with -IncludeVlm after supplying the cache"
        $IncludeVlm = $false
    }
}

Write-QueueMessage "Preflight files passed"
Test-PythonEnvironment `
    -Name "llm" `
    -Python $LlmPython `
    -Code "import torch, transformers, peft, bitsandbytes; assert torch.cuda.is_available(), 'CUDA unavailable'; assert torch.cuda.is_bf16_supported(), 'BF16 unsupported'; print(torch.cuda.get_device_name(0)); print(transformers.__version__)"
if ($IncludeVlm) {
    Test-PythonEnvironment `
        -Name "vlm" `
        -Python $VlmPython `
        -Code "import torch, transformers, peft, bitsandbytes; from transformers import Qwen3VLForConditionalGeneration; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); print(transformers.__version__)"
}

# Generative transcript experiment: zero-shot first.
foreach ($Fold in $Folds) {
    Invoke-QueuedJob `
        -Name "llm_zero_fold_$Fold" `
        -Python $LlmPython `
        -Arguments @(
            "scripts/run_qwen3_8b_wd_zero_fewshot_aligned.py",
            "--dataset", (Join-Path $MasterRoot "fold_$Fold\master_manifest.jsonl"),
            "--shot", "zero",
            "--output", "output/llm_wd_zero_${CohortTag}_cv/fold_$Fold"
        ) `
        -CompletionFile "output/llm_wd_zero_${CohortTag}_cv/fold_$Fold/summary.json"
}

if ($IncludeFewShot) {
foreach ($Fold in $Folds) {
    Invoke-QueuedJob `
        -Name "llm_few_3plus3_fold_$Fold" `
        -Python $LlmPython `
        -Arguments @(
            "scripts/run_qwen3_8b_wd_zero_fewshot_aligned.py",
            "--dataset", (Join-Path $MasterRoot "fold_$Fold\master_manifest.jsonl"),
            "--shot", "few", "--examples-per-class", "3",
            "--output", "output/llm_wd_few_${CohortTag}_cv/fold_$Fold"
        ) `
        -CompletionFile "output/llm_wd_few_${CohortTag}_cv/fold_$Fold/summary.json"
}
}

# QLoRA transcript experiments: regression, consensus binary, then soft labels.
foreach ($Mode in @("regression", "consensus", "soft")) {
    foreach ($Fold in $Folds) {
        Invoke-QueuedJob `
            -Name "llm_${Mode}_fold_$Fold" `
            -Python $LlmPython `
            -Arguments @(
                "scripts/finetune_qwen3_8b_wd_text.py",
                "--dataset", (Join-Path $MasterRoot "fold_$Fold\master_manifest.jsonl"),
                "--mode", $Mode,
                # Freeze the original prompt used by this experiment suite.
                # New prompt experiments must use separate output folders.
                "--rubric", "legacy_short_v1",
                "--output", "output/llm_wd_${Mode}_${CohortTag}_cv/fold_$Fold"
            ) `
            -CompletionFile "output/llm_wd_${Mode}_${CohortTag}_cv/fold_$Fold/final_summary.json"
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
                    "--labels-csv", (Join-Path $MasterRoot "frozen_rater_labels_long.csv"),
                    "--manifest", (Join-Path $MasterRoot "fold_$Fold\vlm_manifest.csv"),
                    "--target-mode", $Mode,
                    "--positive-threshold", "2",
                    "--frame-cache", $VlmFrameCache,
                    "--output-dir", "output/vlm_wd_${Mode}_${CohortTag}_paired_cv/fold_$Fold"
                ) `
                -CompletionFile "output/vlm_wd_${Mode}_${CohortTag}_paired_cv/fold_$Fold/final_summary.json"
        }
    }
}

# Out-of-fold aggregation. Prompting scripts call their file predictions.csv;
# fine-tuning scripts call it test_predictions.csv.
foreach ($PromptRoot in @("llm_wd_zero_${CohortTag}_cv", "llm_wd_few_${CohortTag}_cv")) {
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
    "llm_wd_regression_${CohortTag}_cv", "llm_wd_consensus_${CohortTag}_cv", "llm_wd_soft_${CohortTag}_cv",
    "vlm_wd_consensus_${CohortTag}_paired_cv", "vlm_wd_soft_${CohortTag}_paired_cv"
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

# Training curves are generated after all fold jobs finish. Existing model
# scripts write training_history.csv (LLM) and learning_curves.csv (VLM).
$PlotPython = if ($IncludeVlm) { $VlmPython } else { $LlmPython }
Invoke-QueuedJob `
    -Name "plot_wd_training_validation" `
    -Python $PlotPython `
    -Arguments @(
        "scripts/plot_wd_training_validation.py",
        "--root", "output",
        "--output", "output/wd_training_plots_$CohortTag",
        "--suffix", "_$CohortTag"
    ) `
    -CompletionFile "output/wd_training_plots_$CohortTag/llm_consensus_train_validation.png"

if ($IncludeVlm) {
    foreach ($Mode in @("consensus", "soft")) {
        $VlmPredictions = "output/vlm_wd_${Mode}_${CohortTag}_paired_cv/oof_predictions.csv"
        $LlmPredictions = "output/llm_wd_${Mode}_${CohortTag}_cv/oof_predictions.csv"
        if ((Test-Path -LiteralPath (Resolve-ProjectPath $VlmPredictions)) -and
            (Test-Path -LiteralPath (Resolve-ProjectPath $LlmPredictions))) {
            Invoke-QueuedJob `
                -Name "compare_vlm_llm_$Mode" `
                -Python $LlmPython `
                -Arguments @(
                    "scripts/compare_vlm_llm_wd_predictions.py",
                    "--vlm-predictions", $VlmPredictions,
                    "--llm-predictions", $LlmPredictions,
                    "--vlm-threshold-column", "fold_selected_threshold",
                    "--output", "output/paired_comparison_${Mode}_$CohortTag"
                ) `
                -CompletionFile "output/paired_comparison_${Mode}_$CohortTag/comparison_summary.json"
        }
    }
}

Write-QueueMessage "WD overnight queue finished"
