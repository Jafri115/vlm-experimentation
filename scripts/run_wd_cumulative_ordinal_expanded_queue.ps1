param(
    [string]$Python = '.\.venv\Scripts\python.exe',
    [string]$MasterRoot = '.\output\wd_multimodal_master_expanded',
    [string]$QueueRoot = '.\output\wd_cumulative_ordinal_expanded_queue',
    [string]$Model = 'Qwen/Qwen3-8B'
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Repo

function Resolve-QueuePath([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) { throw 'A required path is empty' }
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $Repo $Value))
}

$Python = Resolve-QueuePath $Python
$MasterRoot = Resolve-QueuePath $MasterRoot
$QueueRoot = Resolve-QueuePath $QueueRoot
$ResultRoot = Join-Path $Repo 'output\llm_wd_cumulative_qwen3_8b_expanded_cv'
$ReportRoot = Join-Path $Repo 'output\wd_cumulative_qwen3_8b_expanded_report'
$LogRoot = Join-Path $QueueRoot 'logs'
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$QueueLog = Join-Path $QueueRoot 'queue.log'
$LockPath = Join-Path $QueueRoot 'queue.lock'
try {
    $LockStream = [IO.File]::Open($LockPath, [IO.FileMode]::OpenOrCreate,
        [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
} catch {
    throw "Another cumulative ordinal queue is already running: $LockPath"
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
    $oldPreference = $ErrorActionPreference
    $exitCode = 1
    try {
        $ErrorActionPreference = 'Continue'
        & $Python @Arguments *>> $log
        $exitCode = $LASTEXITCODE
    } catch {
        $_ | Out-String | Out-File -LiteralPath $log -Append -Encoding utf8
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    if ($exitCode -ne 0 -or -not (Test-Path -LiteralPath $CompletionFile)) {
        Write-Queue "FAILED $Name; queue stopped"
        throw "$Name failed; see $log"
    }
    Write-Queue "COMPLETED $Name"
}

try {
    if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
    foreach ($fold in 1..5) {
        $manifest = Join-Path $MasterRoot "fold_$fold\master_manifest.jsonl"
        if (-not (Test-Path -LiteralPath $manifest)) { throw "Missing manifest: $manifest" }
    }
    $preflight = Join-Path $LogRoot 'preflight.log'
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Python -c "import torch,transformers,peft,bitsandbytes,pandas,sklearn,scipy; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" *>> $preflight
    $preflightExit = $LASTEXITCODE
    $ErrorActionPreference = $oldPreference
    if ($preflightExit -ne 0) { throw "Python/CUDA preflight failed; see $preflight" }

    Write-Queue 'WD cumulative ordinal expanded-cohort queue started'
    foreach ($fold in 1..5) {
        $out = Join-Path $ResultRoot "fold_$fold"
        Invoke-Checked "cumulative_qwen3_8b_fold_$fold" @(
            'scripts/llm/finetune_qwen3_8b_wd_text.py',
            '--dataset', (Join-Path $MasterRoot "fold_$fold\master_manifest.jsonl"),
            '--mode', 'cumulative', '--patient-balanced',
            '--rubric', 'manual_detailed_v3', '--model', $Model, '--output', $out,
            '--epochs', '3', '--learning-rate', '3e-5', '--head-learning-rate', '1e-4',
            '--batch-size', '1', '--eval-batch-size', '2', '--grad-accum', '8',
            '--max-length', '2048', '--lora-rank', '8', '--lora-alpha', '16',
            '--pooling', 'last_token', '--attention', 'sdpa', '--dtype', 'bfloat16',
            '--cumulative-ge2-pos-weight', '1.0', '--cumulative-ge3-pos-weight', '2.5',
            '--seed', '42'
        ) (Join-Path $out 'final_summary.json')
    }
    $oof = Join-Path $ResultRoot 'oof_predictions.csv'
    Invoke-Checked 'combine_cumulative_qwen3_8b' @(
        'scripts/combine_wd_cv_predictions.py', '--fold-root', $ResultRoot, '--output', $oof
    ) $oof
    Invoke-Checked 'report_cumulative_qwen3_8b' @(
        'scripts/report_wd_cumulative_ordinal.py', '--predictions', $oof, '--output', $ReportRoot
    ) (Join-Path $ReportRoot 'cumulative_ordinal_report.md')
    Write-Queue 'WD cumulative ordinal expanded-cohort queue finished'
} finally {
    if ($null -ne $LockStream) { $LockStream.Dispose() }
}
