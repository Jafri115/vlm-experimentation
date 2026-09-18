param(
    [string]$LlmPython = '.\.venv\Scripts\python.exe',
    [string]$VlmPython = '.\.venv\Scripts\python.exe',
    [string]$MasterRoot = '.\output\wd_multimodal_master_expanded',
    [string]$FrameCache = '.\output\wd_multimodal_master_expanded\frame_cache_16'
)
$ErrorActionPreference='Stop'
$QueueRepo=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $QueueRepo

# Main paired suite: transcript zero/few-shot, all transcript fine-tuning,
# visual consensus/soft fine-tuning, curves, aggregation, comparisons.
& (Join-Path $PSScriptRoot 'run_wd_overnight_queue.ps1') `
    -LlmPython $LlmPython -VlmPython $VlmPython `
    -MasterRoot $MasterRoot -VlmFrameCache $FrameCache `
    -CohortTag 'expanded' -IncludeVlm -IncludeFewShot

# Remaining matched visual experiments: zero/few-shot and regression.
# The transcript few-shot stage is safely skipped because the first queue made it.
& (Join-Path $PSScriptRoot 'run_wd_pending_repaired_queue.ps1') `
    -Python $VlmPython -MasterRoot $MasterRoot -FrameCache $FrameCache `
    -QueueRoot '.\output\wd_pending_expanded_queue' `
    -CohortTag 'expanded' -WaitForProcessId 0 -GpuIdleMinutes 0
