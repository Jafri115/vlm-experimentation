param(
    [int]$IdleMinutes = 15,
    [int]$PollSeconds = 60,
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$MasterRoot = ".\output\wd_multimodal_master_expanded",
    [string]$FrameCache = ".\output\wd_multimodal_master_expanded\frame_cache_16",
    [string]$LlmBaselineRoot = ".\output\llm_wd_soft_expanded_cv",
    [string]$VlmBaselineRoot = ".\output\vlm_wd_soft_expanded_paired_cv",
    [switch]$SmokeOnly,
    [switch]$AllowBusyGpu,
    [switch]$Worker
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Root = Join-Path $Repo "output\wd_joint_video_transcript_expanded\scheduler"
New-Item -ItemType Directory -Path $Root -Force | Out-Null
$Log = Join-Path $Root "scheduler.log"

function Log([string]$Message) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" | Tee-Object -FilePath $Log -Append
}

function Get-WorkloadGpuPids {
    # On Windows WDDM, nvidia-smi --query-compute-apps also reports desktop
    # processes (Explorer, Edge, VS Code, etc.). Only treat likely ML/audio
    # workloads as busy; otherwise the scheduler can never become idle.
    $ids = @(& nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>$null |
        ForEach-Object { [int]($_.Trim()) } |
        Where-Object { $_ -gt 0 })
    $busy = @()
    foreach ($id in $ids) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $id" -ErrorAction SilentlyContinue
        if ($null -eq $process) { continue }
        $text = "$($process.Name) $($process.CommandLine)".ToLowerInvariant()
        if ($text -match "python|torch|transcrib|finetune|qwen|asr|transformers|accelerate") {
            $busy += $id
        }
    }
    return @($busy | Sort-Object -Unique)
}

if ($Worker) {
    $start = Join-Path $Repo "scripts\orchestration\start_wd_joint_video_transcript_background.ps1"
    $args = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $start,
        "-Python", $Python, "-MasterRoot", $MasterRoot, "-FrameCache", $FrameCache,
        "-LlmBaselineRoot", $LlmBaselineRoot, "-VlmBaselineRoot", $VlmBaselineRoot
    )
    if ($SmokeOnly) { $args += "-SmokeOnly" }
    if ($AllowBusyGpu) { $args += "-AllowBusyGpu" }
    Log "Starting joint video-transcript background job"
    & powershell.exe @args
    exit $LASTEXITCODE
}

if ($AllowBusyGpu) {
    Log "Starting immediately with -AllowBusyGpu"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath `
        -Python $Python -MasterRoot $MasterRoot -FrameCache $FrameCache `
        -LlmBaselineRoot $LlmBaselineRoot -VlmBaselineRoot $VlmBaselineRoot `
        -SmokeOnly:$SmokeOnly -AllowBusyGpu -Worker
    exit $LASTEXITCODE
}

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi was not found; install/use the NVIDIA driver or pass -AllowBusyGpu."
}

Log "Scheduler started; waiting for $IdleMinutes continuous GPU-idle minutes"
$idleSeconds = 0
while ($true) {
    $pids = @(Get-WorkloadGpuPids)
    if ($pids.Count -eq 0) {
        $idleSeconds += $PollSeconds
        Log "GPU idle for $([math]::Min($idleSeconds, $IdleMinutes * 60))/$($IdleMinutes * 60) seconds"
        if ($idleSeconds -ge ($IdleMinutes * 60)) { break }
    } else {
        if ($idleSeconds -gt 0) { Log "GPU became busy; resetting idle timer" }
        $idleSeconds = 0
        Log "GPU busy with compute PID(s): $($pids -join ', ')"
    }
    Start-Sleep -Seconds $PollSeconds
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath `
    -Python $Python -MasterRoot $MasterRoot -FrameCache $FrameCache `
    -LlmBaselineRoot $LlmBaselineRoot -VlmBaselineRoot $VlmBaselineRoot `
    -SmokeOnly:$SmokeOnly -Worker
exit $LASTEXITCODE
