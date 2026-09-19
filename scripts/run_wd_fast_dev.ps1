param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]*$')][string]$ExperimentName,
    [string]$Python = '.\.venv\Scripts\python.exe',
    [string]$MasterRoot = '.\output\wd_multimodal_master_expanded',
    [ValidateSet('regression','ordinal','cumulative','consensus','soft')][string]$Mode = 'regression',
    [string]$Rubric = 'legacy_short_v1',
    [string]$Model = 'Qwen/Qwen3-8B',
    [ValidateRange(1,5)][int]$Fold = 1,
    [ValidateRange(1,20)][int]$Epochs = 1,
    [string]$Root = '.\output\wd_fast_dev',
    [string]$Baseline = 'baseline_regression',
    [switch]$PatientBalanced,
    [ValidateSet('mean_all','last_token','target_patient')][string]$Pooling = 'mean_all',
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArguments
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Repo

function Resolve-RepoPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $Repo $Value))
}

$Python = Resolve-RepoPath $Python
$MasterRoot = Resolve-RepoPath $MasterRoot
$Root = Resolve-RepoPath $Root
$Manifest = Join-Path $MasterRoot "fold_$Fold\master_manifest.jsonl"
$Output = Join-Path (Join-Path $Root 'runs') $ExperimentName
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $Manifest)) { throw "Manifest not found: $Manifest" }
if (Test-Path -LiteralPath (Join-Path $Output 'final_summary.json')) {
    throw "Completed experiment already exists: $Output. Use a new -ExperimentName so results remain auditable."
}
if (Test-Path -LiteralPath $Output) {
    Write-Host "Resuming/replacing incomplete experiment directory: $Output"
}
New-Item -ItemType Directory -Path $Output -Force | Out-Null

$Arguments = @(
    'scripts/finetune_qwen3_8b_wd_text.py',
    '--dataset', $Manifest, '--mode', $Mode, '--output', $Output,
    '--model', $Model, '--rubric', $Rubric, '--epochs', "$Epochs",
    '--pooling', $Pooling, '--skip-test-evaluation'
)
if ($PatientBalanced) { $Arguments += '--patient-balanced' }
if ($ExtraArguments) { $Arguments += $ExtraArguments }

$CommandPath = Join-Path $Output 'command.txt'
"$Python $($Arguments -join ' ')" | Set-Content -LiteralPath $CommandPath -Encoding utf8
Write-Host "Running fixed-fold development experiment: $ExperimentName"
Write-Host "Fold: $Fold; validation only; held-out test inference disabled"
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Development experiment failed with exit code $LASTEXITCODE" }

& $Python scripts/report_wd_fast_dev.py --root $Root --baseline $Baseline
if ($LASTEXITCODE -ne 0) { throw "Development report failed with exit code $LASTEXITCODE" }
Write-Host "Comparison: $(Join-Path $Root 'development_comparison.md')"
