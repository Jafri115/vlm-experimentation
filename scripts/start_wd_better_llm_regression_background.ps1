param(
    [string]$Python = '.\.venv\Scripts\python.exe',
    [string]$MasterRoot = '.\output\wd_multimodal_master_expanded',
    [string]$QueueRoot = '.\output\wd_better_llm_regression_queue'
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Repo
if ([IO.Path]::IsPathRooted($QueueRoot)) {
    $QueueRoot = [IO.Path]::GetFullPath($QueueRoot)
} else {
    $QueueRoot = [IO.Path]::GetFullPath((Join-Path $Repo $QueueRoot))
}
New-Item -ItemType Directory -Path $QueueRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$outLog = Join-Path $QueueRoot "background_$stamp.out.log"
$errLog = Join-Path $QueueRoot "background_$stamp.err.log"
$arguments = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass',
    '-File', (Join-Path $PSScriptRoot 'run_wd_better_llm_regression_queue.ps1'),
    '-Python', $Python, '-MasterRoot', $MasterRoot, '-QueueRoot', $QueueRoot
)
$process = Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments `
    -WorkingDirectory $Repo -WindowStyle Hidden `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
Write-Host "Background PID: $($process.Id)"
Write-Host "Queue log: $(Join-Path $QueueRoot 'queue.log')"
Write-Host "Startup errors: $errLog"
