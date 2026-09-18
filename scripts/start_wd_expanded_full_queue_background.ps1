param(
    [string]$LlmPython = '.\.venv\Scripts\python.exe',
    [string]$VlmPython = '.\.venv\Scripts\python.exe'
)
$ErrorActionPreference='Stop'
$QueueRepo=Split-Path $PSScriptRoot -Parent
$QueueRoot=Join-Path $QueueRepo 'output\wd_expanded_full_queue'
New-Item -ItemType Directory -Path $QueueRoot -Force | Out-Null
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$Args=@('-NoProfile','-ExecutionPolicy','Bypass','-File',('"{0}"' -f (Join-Path $PSScriptRoot 'run_wd_expanded_full_queue.ps1')),
    '-LlmPython',('"{0}"' -f $LlmPython),'-VlmPython',('"{0}"' -f $VlmPython))
$Launch=@{FilePath='powershell.exe';ArgumentList=$Args;WorkingDirectory=$QueueRepo;WindowStyle='Hidden';
    RedirectStandardOutput=(Join-Path $QueueRoot "background_$Stamp.out.log");
    RedirectStandardError=(Join-Path $QueueRoot "background_$Stamp.err.log");PassThru=$true}
$QueueProcess=Start-Process @Launch
Write-Host "Background PID: $($QueueProcess.Id)"
Write-Host "Main queue log: $QueueRepo\output\wd_overnight_queue_expanded\queue.log"
Write-Host "Launcher error log: $($Launch.RedirectStandardError)"
