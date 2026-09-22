param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$MasterRoot = ".\output\wd_multimodal_master_expanded",
    [string]$FrameCache = ".\output\wd_multimodal_master_expanded\frame_cache_16",
    [string]$OutputRoot = ".\output\wd_joint_video_transcript_expanded",
    [string]$LlmBaselineRoot = ".\output\llm_wd_soft_expanded_cv",
    [string]$VlmBaselineRoot = ".\output\vlm_wd_soft_expanded_paired_cv",
    [switch]$SmokeOnly,
    [switch]$AllowBusyGpu
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location -LiteralPath $Repo

function Resolve-RepoPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $Repo $Value))
}

$Python = Resolve-RepoPath $Python
$MasterRoot = Resolve-RepoPath $MasterRoot
$FrameCache = Resolve-RepoPath $FrameCache
$OutputRoot = Resolve-RepoPath $OutputRoot
$LlmBaselineRoot = Resolve-RepoPath $LlmBaselineRoot
$VlmBaselineRoot = Resolve-RepoPath $VlmBaselineRoot
$Labels = Join-Path $MasterRoot "frozen_rater_labels_long.csv"
$LogDir = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$QueueLog = Join-Path $OutputRoot "queue.log"

function Log([string]$Message) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    $line | Tee-Object -FilePath $QueueLog -Append
}

function Run-Step([string]$Name, [string[]]$Arguments, [string]$CompletionFile) {
    if (Test-Path -LiteralPath $CompletionFile) {
        Log "SKIP $Name (completion file exists)"
        return
    }
    $log = Join-Path $LogDir "$Name.log"
    "COMMAND: $Python $($Arguments -join ' ')" | Out-File -LiteralPath $log -Encoding utf8
    Log "START $Name"
    & $Python @Arguments *>> $log
    if ($LASTEXITCODE -ne 0) {
        Log "FAILED $Name (exit $LASTEXITCODE)"
        throw "$Name failed; see $log"
    }
    if (-not (Test-Path -LiteralPath $CompletionFile)) {
        throw "$Name exited successfully but did not create $CompletionFile"
    }
    Log "COMPLETED $Name"
}

Log "Joint video-transcript fusion queue started"
if (-not $AllowBusyGpu) {
    $gpuPids = @(& nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>$null | Where-Object { $_.Trim() })
    if ($gpuPids.Count -gt 0) {
        throw "GPU has active compute processes ($($gpuPids -join ', ')); stop them or rerun with -AllowBusyGpu."
    }
}
foreach ($path in @($Python, $MasterRoot, $FrameCache, $Labels)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path missing: $path" }
}

& $Python -c "import torch, transformers, peft, pandas, sklearn; from transformers import Qwen3VLForConditionalGeneration; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" *>> (Join-Path $LogDir "environment_preflight.log")
if ($LASTEXITCODE -ne 0) { throw "Python/CUDA/Qwen3-VL environment preflight failed" }
& $Python -m pip freeze | Out-File -LiteralPath (Join-Path $OutputRoot "dependencies.txt") -Encoding utf8
git rev-parse HEAD | Out-File -LiteralPath (Join-Path $OutputRoot "code_version.txt") -Encoding ascii

Run-Step "dataset_audit" @(
    "scripts/analysis/audit_wd_joint_video_transcript.py",
    "--master-root", $MasterRoot, "--frame-cache", $FrameCache,
    "--output", (Join-Path $OutputRoot "audit")
) (Join-Path $OutputRoot "audit\dataset_summary.json")

Run-Step "multimodal_smoke" @(
    "scripts/vlm/check_wd_joint_multimodal.py",
    "--labels-csv", $Labels,
    "--manifest", (Join-Path $MasterRoot "fold_1\vlm_manifest.csv"),
    "--frame-cache", $FrameCache,
    "--output", (Join-Path $OutputRoot "smoke")
) (Join-Path $OutputRoot "smoke\implementation_checks.json")

if ($SmokeOnly) {
    Log "Smoke-only queue finished"
    exit 0
}

foreach ($fold in 1..5) {
    $foldOutput = Join-Path $OutputRoot "fold_$fold"
    Run-Step "joint_fold_$fold" @(
        "scripts/vlm/finetune_qwen3vl_wd_consensus_binary.py",
        "--labels-csv", $Labels,
        "--manifest", (Join-Path $MasterRoot "fold_$fold\vlm_manifest.csv"),
        "--target-mode", "soft", "--positive-threshold", "2",
        "--input-mode", "joint", "--transcript-column", "transcript_text",
        "--pooling", "last_token", "--max-input-tokens", "32768",
        "--outer-fold", "$fold", "--frame-cache", $FrameCache,
        "--num-frames", "16", "--frame-width", "224", "--epochs", "1",
        "--output-dir", $foldOutput
    ) (Join-Path $foldOutput "final_summary.json")
}

Run-Step "combine_joint_oof" @(
    "scripts/analysis/combine_wd_cv_predictions.py",
    "--fold-root", $OutputRoot,
    "--output", (Join-Path $OutputRoot "oof_predictions.csv")
) (Join-Path $OutputRoot "oof_predictions.csv")

Run-Step "joint_comparison_report" @(
    "scripts/analysis/report_wd_joint_video_transcript.py",
    "--joint-root", $OutputRoot,
    "--llm-root", $LlmBaselineRoot,
    "--vlm-root", $VlmBaselineRoot,
    "--output", (Join-Path $OutputRoot "report"),
    "--bootstrap", "10000", "--seed", "42"
) (Join-Path $OutputRoot "report\report.md")

Log "Joint video-transcript fusion queue finished"
