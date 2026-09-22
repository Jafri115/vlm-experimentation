param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$MasterRoot = ".\output\wd_multimodal_master_expanded",
    [string]$FrameCache = ".\output\wd_multimodal_master_expanded\frame_cache_16",
    [string]$LlmBaselineRoot = ".\output\llm_wd_soft_expanded_cv",
    [string]$VlmBaselineRoot = ".\output\vlm_wd_soft_expanded_paired_cv",
    [switch]$SmokeOnly,
    [switch]$AllowBusyGpu
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Root = Join-Path $Repo "output\wd_joint_video_transcript_expanded"
New-Item -ItemType Directory -Path $Root -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$outLog = Join-Path $Root "background_$stamp.out.log"
$errLog = Join-Path $Root "background_$stamp.err.log"
$queue = Join-Path $Repo "scripts\orchestration\run_wd_joint_video_transcript_queue.ps1"
$arguments = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $queue,
    "-Python", $Python, "-MasterRoot", $MasterRoot, "-FrameCache", $FrameCache,
    "-LlmBaselineRoot", $LlmBaselineRoot, "-VlmBaselineRoot", $VlmBaselineRoot
)
if ($SmokeOnly) { $arguments += "-SmokeOnly" }
if ($AllowBusyGpu) { $arguments += "-AllowBusyGpu" }
$process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments `
    -WorkingDirectory $Repo -WindowStyle Hidden `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
Write-Host "Background PID: $($process.Id)"
Write-Host "Queue log: $Root\queue.log"
Write-Host "Output log: $outLog"
Write-Host "Error log: $errLog"
