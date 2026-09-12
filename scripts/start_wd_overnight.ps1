param(
    [string]$LlmPython = ".\.venv-llm-ft\Scripts\python.exe",
    [string]$VlmPython = ".\.venv\Scripts\python.exe",
    [switch]$IncludeVlm
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$QueueScript = Join-Path $PSScriptRoot "run_wd_overnight_queue.ps1"
$RunRoot = Join-Path $ProjectRoot "output\wd_overnight_queue"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$Arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $QueueScript,
    "-LlmPython", $LlmPython,
    "-VlmPython", $VlmPython
)
if ($IncludeVlm) {
    $Arguments += "-IncludeVlm"
}

$Stdout = Join-Path $RunRoot "launcher.stdout.log"
$Stderr = Join-Path $RunRoot "launcher.stderr.log"
$Process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $Arguments `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -PassThru

$State = [ordered]@{
    pid = $Process.Id
    started = (Get-Date).ToString("o")
    project_root = $ProjectRoot
    include_vlm = [bool]$IncludeVlm
    queue_log = (Join-Path $RunRoot "queue.log")
    job_status = (Join-Path $RunRoot "job_status.csv")
    stdout = $Stdout
    stderr = $Stderr
}
$State | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunRoot "background_process.json") -Encoding utf8
$State | ConvertTo-Json
