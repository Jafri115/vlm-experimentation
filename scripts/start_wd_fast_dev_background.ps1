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
if (-not [IO.Path]::IsPathRooted($Root)) { $Root = Join-Path $Repo $Root }
$Root = [IO.Path]::GetFullPath($Root)
$LogRoot = Join-Path $Root 'background_logs'
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$OutLog = Join-Path $LogRoot "${ExperimentName}_${stamp}.out.log"
$ErrLog = Join-Path $LogRoot "${ExperimentName}_${stamp}.err.log"
$Arguments = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass',
    '-File', (Join-Path $PSScriptRoot 'run_wd_fast_dev.ps1'),
    '-ExperimentName', $ExperimentName, '-Python', $Python,
    '-MasterRoot', $MasterRoot, '-Mode', $Mode, '-Rubric', $Rubric,
    '-Model', $Model, '-Fold', "$Fold", '-Epochs', "$Epochs",
    '-Root', $Root, '-Baseline', $Baseline, '-Pooling', $Pooling
)
if ($PatientBalanced) { $Arguments += '-PatientBalanced' }
if ($ExtraArguments) { $Arguments += $ExtraArguments }
$Process = Start-Process -FilePath 'powershell.exe' -ArgumentList $Arguments `
    -WorkingDirectory $Repo -WindowStyle Hidden `
    -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -PassThru
Write-Host "Background PID: $($Process.Id)"
Write-Host "Output log: $OutLog"
Write-Host "Error log: $ErrLog"
Write-Host "Comparison after completion: $(Join-Path $Root 'development_comparison.md')"
