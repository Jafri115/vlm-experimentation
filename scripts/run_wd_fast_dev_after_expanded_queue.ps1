param(
    [string]$Python = '.\.venv\Scripts\python.exe',
    [string]$ExpandedMasterRoot = '.\output\wd_multimodal_master_expanded',
    [string]$DatasetV2MasterRoot = '.\output\wd_multimodal_master_labels_v2',
    [string]$FastDevRoot = '.\output\wd_fast_dev',
    [string]$QueueRoot = '.\output\wd_fast_dev_after_expanded',
    [int]$ExpandedQueueProcessId = 0,
    [ValidateRange(5,300)][int]$PollSeconds = 30,
    [switch]$SkipDatasetV2
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Repo

function Resolve-RepoPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) {
        return [IO.Path]::GetFullPath($Value)
    }
    return [IO.Path]::GetFullPath((Join-Path $Repo $Value))
}

$Python = Resolve-RepoPath $Python
$ExpandedMasterRoot = Resolve-RepoPath $ExpandedMasterRoot
$DatasetV2MasterRoot = Resolve-RepoPath $DatasetV2MasterRoot
$FastDevRoot = Resolve-RepoPath $FastDevRoot
$QueueRoot = Resolve-RepoPath $QueueRoot
$QueueLog = Join-Path $QueueRoot 'queue.log'
$LockPath = Join-Path $QueueRoot 'scheduler.lock.json'
$CompletionPath = Join-Path $QueueRoot 'completion.json'
$FastDevScript = Join-Path $PSScriptRoot 'run_wd_fast_dev.ps1'

New-Item -ItemType Directory -Path $QueueRoot -Force | Out-Null

function Write-QueueMessage([string]$Message) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    $line | Tee-Object -FilePath $QueueLog -Append
}

function Get-ExpandedQueueProcesses {
    return @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessId -ne $PID -and $_.CommandLine -and (
            $_.CommandLine -like '*run_wd_expanded_full_queue.ps1*' -or
            ($_.CommandLine -like '*run_wd_overnight_queue.ps1*' -and $_.CommandLine -like '*expanded*') -or
            ($_.CommandLine -like '*run_wd_pending_repaired_queue.ps1*' -and $_.CommandLine -like '*expanded*')
        )
    })
}

function Test-ExpandedSuiteComplete {
    $script:MissingExpandedArtifacts = @(
        (Join-Path $Repo 'output\paired_comparison_consensus_expanded\comparison_summary.json'),
        (Join-Path $Repo 'output\paired_comparison_soft_expanded\comparison_summary.json'),
        (Join-Path $Repo 'output\vlm_wd_zero_expanded_cv\oof_predictions.csv'),
        (Join-Path $Repo 'output\vlm_wd_few_expanded_cv\oof_predictions.csv'),
        (Join-Path $Repo 'output\vlm_wd_regression_expanded_cv\oof_predictions.csv')
    ) | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }
    return $script:MissingExpandedArtifacts.Count -eq 0
}

function Wait-ForExpandedSuite {
    if ($ExpandedQueueProcessId -gt 0) {
        $process = Get-Process -Id $ExpandedQueueProcessId -ErrorAction SilentlyContinue
        if ($process) {
            Write-QueueMessage "Waiting for expanded queue process $ExpandedQueueProcessId"
            Wait-Process -Id $ExpandedQueueProcessId
            Write-QueueMessage "Expanded queue process $ExpandedQueueProcessId ended"
        } else {
            Write-QueueMessage "Expanded queue process $ExpandedQueueProcessId is already absent"
        }
    } else {
        $processes = @(Get-ExpandedQueueProcesses)
        if ($processes.Count -gt 0) {
            Write-QueueMessage "Detected expanded queue process(es): $($processes.ProcessId -join ', ')"
            $lastStatus = Get-Date
            while ($true) {
                Start-Sleep -Seconds $PollSeconds
                $processes = @(Get-ExpandedQueueProcesses)
                if ($processes.Count -eq 0) { break }
                if (((Get-Date) - $lastStatus).TotalMinutes -ge 5) {
                    Write-QueueMessage "Still waiting for expanded queue process(es): $($processes.ProcessId -join ', ')"
                    $lastStatus = Get-Date
                }
            }
            Write-QueueMessage 'Expanded queue processes ended'
        } else {
            Write-QueueMessage 'No active expanded queue process detected; checking completion artifacts'
        }
    }

    if (-not (Test-ExpandedSuiteComplete)) {
        foreach ($path in $script:MissingExpandedArtifacts) {
            Write-QueueMessage "MISSING expanded-suite artifact: $path"
        }
        throw 'The expanded suite did not finish successfully; fast experiments were not started.'
    }
    Write-QueueMessage 'Expanded-suite completion artifacts verified'
}

function Invoke-FastExperiment {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string]$MasterRoot,
        [Parameter(Mandatory=$true)][string]$Mode,
        [Parameter(Mandatory=$true)][string]$Rubric,
        [string]$Pooling = 'mean_all',
        [switch]$PatientBalanced
    )

    $summary = Join-Path (Join-Path (Join-Path $FastDevRoot 'runs') $Name) 'final_summary.json'
    if (Test-Path -LiteralPath $summary -PathType Leaf) {
        Write-QueueMessage "SKIP $Name (completion file exists)"
        return
    }

    $manifest = Join-Path $MasterRoot 'fold_1\master_manifest.jsonl'
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
        throw "Manifest missing for ${Name}: $manifest"
    }

    $arguments = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', $FastDevScript,
        '-ExperimentName', $Name,
        '-Python', $Python,
        '-MasterRoot', $MasterRoot,
        '-Mode', $Mode,
        '-Rubric', $Rubric,
        '-Pooling', $Pooling,
        '-Epochs', '1',
        '-Root', $FastDevRoot,
        '-Baseline', 'baseline_regression'
    )
    if ($PatientBalanced) { $arguments += '-PatientBalanced' }

    Write-QueueMessage "START $Name"
    & powershell.exe @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath $summary -PathType Leaf)) {
        throw "$Name ended without its completion file: $summary"
    }
    Write-QueueMessage "COMPLETED $Name"
}

if (Test-Path -LiteralPath $LockPath -PathType Leaf) {
    $oldPid = 0
    try {
        $oldLock = Get-Content -LiteralPath $LockPath -Raw | ConvertFrom-Json
        $oldPid = [int]$oldLock.pid
    } catch {
        Write-QueueMessage 'Ignoring an unreadable or stale scheduler lock'
    }
    if ($oldPid -gt 0) {
        $oldProcess = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
        if ($oldProcess) {
            throw "Another fast-development scheduler is active as PID $oldPid."
        }
    }
}

@{
    pid = $PID
    started_at = (Get-Date).ToString('o')
    repo = $Repo
} | ConvertTo-Json | Set-Content -LiteralPath $LockPath -Encoding utf8

try {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Python executable not found: $Python"
    }
    foreach ($requiredScript in @(
        $FastDevScript,
        (Join-Path $PSScriptRoot 'report_wd_fast_dev.py'),
        (Join-Path $PSScriptRoot 'llm\finetune_qwen3_8b_wd_text.py')
    )) {
        if (-not (Test-Path -LiteralPath $requiredScript -PathType Leaf)) {
            throw "Required fast-development script not found: $requiredScript"
        }
    }

    Write-QueueMessage 'Fast-development follow-on scheduler started'
    Wait-ForExpandedSuite

    Invoke-FastExperiment -Name 'baseline_regression' `
        -MasterRoot $ExpandedMasterRoot -Mode 'regression' -Rubric 'legacy_short_v1'
    Invoke-FastExperiment -Name 'manual_prompt_regression' `
        -MasterRoot $ExpandedMasterRoot -Mode 'regression' -Rubric 'manual_compact_v2'
    Invoke-FastExperiment -Name 'ordinal_manual_prompt' `
        -MasterRoot $ExpandedMasterRoot -Mode 'ordinal' -Rubric 'manual_compact_v2' `
        -PatientBalanced -Pooling 'last_token'
    Invoke-FastExperiment -Name 'cumulative_manual_prompt' `
        -MasterRoot $ExpandedMasterRoot -Mode 'cumulative' -Rubric 'manual_compact_v2' `
        -PatientBalanced -Pooling 'last_token'

    if ($SkipDatasetV2) {
        Write-QueueMessage 'SKIP dataset_v2_regression (-SkipDatasetV2 supplied)'
    } elseif (Test-Path -LiteralPath (Join-Path $DatasetV2MasterRoot 'fold_1\master_manifest.jsonl') -PathType Leaf) {
        Invoke-FastExperiment -Name 'dataset_v2_regression' `
            -MasterRoot $DatasetV2MasterRoot -Mode 'regression' -Rubric 'legacy_short_v1'
    } else {
        Write-QueueMessage "SKIP dataset_v2_regression (dataset is not ready: $DatasetV2MasterRoot)"
    }

    $completedExperiments = @(
        'baseline_regression',
        'manual_prompt_regression',
        'ordinal_manual_prompt',
        'cumulative_manual_prompt'
    )
    if (Test-Path -LiteralPath (Join-Path $FastDevRoot 'runs\dataset_v2_regression\final_summary.json')) {
        $completedExperiments += 'dataset_v2_regression'
    }
    @{
        status = 'complete'
        completed_at = (Get-Date).ToString('o')
        experiments = $completedExperiments
        comparison = (Join-Path $FastDevRoot 'development_comparison.md')
    } | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $CompletionPath -Encoding utf8
    Write-QueueMessage 'All available fast-development experiments completed'
    Write-QueueMessage "Comparison report: $(Join-Path $FastDevRoot 'development_comparison.md')"
} catch {
    Write-QueueMessage "FAILED: $($_.Exception.Message)"
    throw
} finally {
    if (Test-Path -LiteralPath $LockPath) {
        Remove-Item -LiteralPath $LockPath -Force
    }
}
