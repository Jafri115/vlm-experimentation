param(
    [string]$Python = '.\.venv\Scripts\python.exe',
    [string]$Output = 'output\wd_context_experiment',
    [switch]$PatientPooling
)
$ErrorActionPreference = 'Stop'
$ContextRepo = Split-Path $PSScriptRoot -Parent
if (-not [IO.Path]::IsPathRooted($Python)) { $Python = Join-Path $ContextRepo $Python }
if (-not [IO.Path]::IsPathRooted($Output)) { $Output = Join-Path $ContextRepo $Output }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
New-Item -ItemType Directory -Path $Output -Force | Out-Null
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$LaunchArgs = @('-u', ('"{0}"' -f (Join-Path $PSScriptRoot 'run_wd_context_experiment.py')), '--output', ('"{0}"' -f $Output))
if ($PatientPooling) { $LaunchArgs += '--patient-pooling' }
$Launch = @{
    FilePath = $Python
    ArgumentList = $LaunchArgs
    WorkingDirectory = $ContextRepo
    WindowStyle = 'Hidden'
    RedirectStandardOutput = (Join-Path $Output "background_$Stamp.out.log")
    RedirectStandardError = (Join-Path $Output "background_$Stamp.err.log")
    PassThru = $true
}
$ContextProcess = Start-Process @Launch
Write-Host "Background PID: $($ContextProcess.Id)"
Write-Host "Queue log: $Output\queue.log"
Write-Host "Startup errors: $($Launch.RedirectStandardError)"
