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

function Resolve-LauncherPath([string]$PathValue) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    $Relative = $PathValue -replace '^[.][\\/]', ''
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Relative))
}

$LlmPython = Resolve-LauncherPath $LlmPython
$VlmPython = Resolve-LauncherPath $VlmPython
if (-not (Test-Path -LiteralPath $LlmPython)) {
    $FallbackPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $FallbackPython) {
        $LlmPython = $FallbackPython
    }
}
if (-not (Test-Path -LiteralPath $LlmPython)) {
    throw "LLM Python executable not found. Checked $LlmPython and .venv\Scripts\python.exe"
}
if ($IncludeVlm -and -not (Test-Path -LiteralPath $VlmPython)) {
    throw "VLM Python executable not found: $VlmPython"
}

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
