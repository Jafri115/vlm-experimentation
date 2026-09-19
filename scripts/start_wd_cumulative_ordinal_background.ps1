param(
    [string]$Python = '.\.venv\Scripts\python.exe',
    [string]$MasterRoot = '.\output\wd_multimodal_master_expanded',
    [string]$QueueRoot = '.\output\wd_cumulative_ordinal_expanded_queue'
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Repo
if (-not [IO.Path]::IsPathRooted($QueueRoot)) { $QueueRoot = Join-Path $Repo $QueueRoot }
$QueueRoot = [IO.Path]::GetFullPath($QueueRoot)
New-Item -ItemType Directory -Path $QueueRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$stdout = Join-Path $QueueRoot "background_$stamp.out.log"
$stderr = Join-Path $QueueRoot "background_$stamp.err.log"
$queueScript = Join-Path $PSScriptRoot 'run_wd_cumulative_ordinal_expanded_queue.ps1'
$arguments = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $queueScript,
    '-Python', $Python, '-MasterRoot', $MasterRoot, '-QueueRoot', $QueueRoot
)
$process = Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments `
    -WorkingDirectory $Repo -WindowStyle Hidden `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Start-Sleep -Seconds 3
Write-Host "Background PID: $($process.Id)"
Write-Host "Running: $(-not $process.HasExited)"
Write-Host "Queue log: $(Join-Path $QueueRoot 'queue.log')"
Write-Host "Startup errors: $stderr"
if ($process.HasExited -and (Test-Path -LiteralPath $stderr)) {
    Get-Content -LiteralPath $stderr -Tail 100
}
