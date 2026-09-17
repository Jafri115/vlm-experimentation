param(
    [string]$Python = '.\.venv\Scripts\python.exe',
    [switch]$RunSeeds,
    [switch]$IncludeVlmSwap
)
$ErrorActionPreference = 'Stop'
$Project = Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($Python)) {
    $Python = Join-Path $Project $Python
}
$Python = (Resolve-Path -LiteralPath $Python).Path
$QueueScript = Join-Path $PSScriptRoot 'run_wd_three_day_queue.py'
$QueueRoot = Join-Path $Project 'output\wd_presentation_3day'
New-Item -ItemType Directory -Path $QueueRoot -Force | Out-Null
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$OutLog = Join-Path $QueueRoot "background_$Stamp.out.log"
$ErrLog = Join-Path $QueueRoot "background_$Stamp.err.log"
$LaunchArguments = @('-u', ('"{0}"' -f $QueueScript))
if ($RunSeeds) { $LaunchArguments += '--run-seeds' }
if ($IncludeVlmSwap) { $LaunchArguments += '--include-vlm-swap' }
$Launch = @{
    FilePath = $Python
    ArgumentList = $LaunchArguments
    WorkingDirectory = $Project
    WindowStyle = 'Hidden'
    RedirectStandardOutput = $OutLog
    RedirectStandardError = $ErrLog
    PassThru = $true
}
$QueueProcess = Start-Process @Launch
Write-Host "Queue PID: $($QueueProcess.Id)"
Write-Host "Output: $OutLog"
Write-Host "Errors: $ErrLog"
Write-Host "Progress: $QueueRoot\queue.log"
Write-Host 'Startup is not yet verified; inspect the output/error logs. The queue has an OS lock against duplicate runs.'
