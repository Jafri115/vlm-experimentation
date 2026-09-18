param(
    [string]$Python = '.\.venv\Scripts\python.exe',
    [string]$MasterRoot = '.\output\wd_multimodal_master_expanded',
    [string]$QueueRoot = '.\output\wd_better_llm_regression_queue'
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Repo
$Python = [IO.Path]::GetFullPath((Join-Path $Repo $Python))
$MasterRoot = [IO.Path]::GetFullPath((Join-Path $Repo $MasterRoot))
$QueueRoot = [IO.Path]::GetFullPath((Join-Path $Repo $QueueRoot))
$LogRoot = Join-Path $QueueRoot 'logs'
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$QueueLog = Join-Path $QueueRoot 'queue.log'
$LockPath = Join-Path $QueueRoot 'queue.lock'
try {
    # The OS releases this handle after a crash; the small lock file may remain.
    $LockStream = [IO.File]::Open($LockPath, [IO.FileMode]::OpenOrCreate,
        [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
} catch {
    throw "Another better-regression queue is already running: $LockPath"
}

function Write-Queue([string]$Message) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    $line | Out-File -LiteralPath $QueueLog -Append -Encoding utf8
    Write-Host $line
}
function Invoke-Checked([string]$Name, [string[]]$Arguments, [string]$CompletionFile) {
    if (Test-Path -LiteralPath $CompletionFile) {
        Write-Queue "SKIP $Name (completion file exists)"
        return
    }
    $log = Join-Path $LogRoot ($Name + '.log')
    Write-Queue "START $Name"
    "COMMAND: $Python $($Arguments -join ' ')" | Out-File -LiteralPath $log -Encoding utf8
    $previousPreference = $ErrorActionPreference
    $exitCode = 1
    try {
        # Native libraries write harmless warnings to stderr. Windows PowerShell
        # must judge the process by its exit code, not by the presence of stderr.
        $ErrorActionPreference = 'Continue'
        & $Python @Arguments *>> $log
        $exitCode = $LASTEXITCODE
    } catch {
        $_ | Out-String | Out-File -LiteralPath $log -Append -Encoding utf8
        $exitCode = 1
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        Write-Queue "FAILED $Name (exit $exitCode); queue stopped"
        throw "$Name failed; see $log"
    }
    if (-not (Test-Path -LiteralPath $CompletionFile)) {
        Write-Queue "FAILED $Name (missing completion file); queue stopped"
        throw "$Name did not create $CompletionFile"
    }
    Write-Queue "COMPLETED $Name"
}

if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
foreach ($fold in 1..5) {
    $manifest = Join-Path $MasterRoot "fold_$fold\master_manifest.jsonl"
    if (-not (Test-Path -LiteralPath $manifest)) { throw "Missing manifest: $manifest" }
}
$preflightLog = Join-Path $LogRoot 'preflight.log'
$previousPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $Python -c "import torch,transformers,peft,bitsandbytes,pandas,sklearn,scipy; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" *>> $preflightLog
$preflightExit = $LASTEXITCODE
$ErrorActionPreference = $previousPreference
if ($preflightExit -ne 0) { throw "Python/CUDA preflight failed; see $preflightLog" }
Write-Queue 'Better LLM regression queue started (14B primary, then 8B objective control)'

$models = @(
    @{ Tag='qwen3_14b'; Id='Qwen/Qwen3-14B'; LearningRate='3e-5'; EvalBatch='1' },
    @{ Tag='qwen3_8b'; Id='Qwen/Qwen3-8B'; LearningRate='3e-5'; EvalBatch='2' }
)
foreach ($model in $models) {
    $root = Join-Path $Repo "output\llm_wd_ordinal_$($model.Tag)_expanded_cv"
    foreach ($fold in 1..5) {
        $out = Join-Path $root "fold_$fold"
        $arguments = @(
            'scripts/finetune_qwen3_8b_wd_text.py',
            '--dataset', (Join-Path $MasterRoot "fold_$fold\master_manifest.jsonl"),
            '--mode', 'ordinal', '--patient-balanced',
            '--model', $model.Id, '--output', $out,
            '--epochs', '3', '--learning-rate', $model.LearningRate,
            '--head-learning-rate', '1e-4', '--batch-size', '1',
            '--eval-batch-size', $model.EvalBatch, '--grad-accum', '8',
            '--max-length', '2048', '--lora-rank', '8', '--lora-alpha', '16',
            '--pooling', 'last_token', '--attention', 'sdpa', '--dtype', 'bfloat16', '--seed', '42'
        )
        Invoke-Checked "ordinal_$($model.Tag)_fold_$fold" $arguments (Join-Path $out 'final_summary.json')
    }
    Invoke-Checked "combine_$($model.Tag)" @(
        'scripts/combine_wd_cv_predictions.py', '--fold-root', $root,
        '--output', (Join-Path $root 'oof_predictions.csv')
    ) (Join-Path $root 'oof_predictions.csv')
}

$reportRoot = Join-Path $Repo 'output\wd_ordinal_regression_expanded_report'
Invoke-Checked 'report_ordinal_regression' @(
    'scripts/report_wd_ordinal_regression.py', '--master-root', $MasterRoot,
    '--output', $reportRoot
) (Join-Path $reportRoot 'ordinal_regression_report.md')
Write-Queue 'Better LLM regression queue finished'
$LockStream.Dispose()
