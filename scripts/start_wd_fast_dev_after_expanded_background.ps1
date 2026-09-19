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

$QueueRoot = Resolve-RepoPath $QueueRoot
$Runner = Join-Path $PSScriptRoot 'run_wd_fast_dev_after_expanded_queue.ps1'
New-Item -ItemType Directory -Path $QueueRoot -Force | Out-Null

$existing = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.ProcessId -ne $PID -and $_.CommandLine -and
    $_.CommandLine -like '*run_wd_fast_dev_after_expanded_queue.ps1*'
})
if ($existing.Count -gt 0) {
    throw "A follow-on scheduler is already running as PID(s): $($existing.ProcessId -join ', ')"
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$OutLog = Join-Path $QueueRoot "background_${stamp}.out.log"
$ErrLog = Join-Path $QueueRoot "background_${stamp}.err.log"
$arguments = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass',
    '-File', $Runner,
    '-Python', $Python,
    '-ExpandedMasterRoot', $ExpandedMasterRoot,
    '-DatasetV2MasterRoot', $DatasetV2MasterRoot,
    '-FastDevRoot', $FastDevRoot,
    '-QueueRoot', $QueueRoot,
    '-ExpandedQueueProcessId', "$ExpandedQueueProcessId",
    '-PollSeconds', "$PollSeconds"
)
if ($SkipDatasetV2) { $arguments += '-SkipDatasetV2' }

$process = Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments `
    -WorkingDirectory $Repo -WindowStyle Hidden `
    -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -PassThru

Start-Sleep -Seconds 2
Write-Host "Follow-on scheduler PID: $($process.Id)"
Write-Host "Running: $(-not $process.HasExited)"
Write-Host "Queue log: $(Join-Path $QueueRoot 'queue.log')"
Write-Host "Background output: $OutLog"
Write-Host "Startup/errors: $ErrLog"
if ($process.HasExited -and (Test-Path -LiteralPath $ErrLog)) {
    Write-Host "`n===== STARTUP ERROR ====="
    Get-Content -LiteralPath $ErrLog -Tail 100
}

