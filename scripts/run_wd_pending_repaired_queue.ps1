param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$MasterRoot = ".\output\wd_multimodal_master_repaired",
    [string]$FrameCache = ".\output\qwen3vl_wd_planning196_thr2\frame_cache_16",
    [string]$QueueRoot = ".\output\wd_pending_repaired_queue",
    [int]$WaitForProcessId = 0,
    [int]$GpuIdleMinutes = 10
)

$ErrorActionPreference = "Stop"
$Project = (Get-Location).Path
function Abs([string]$p) { if ([IO.Path]::IsPathRooted($p)) { return $p }; return Join-Path $Project $p }
$Python = Abs $Python; $MasterRoot = Abs $MasterRoot; $FrameCache = Abs $FrameCache; $QueueRoot = Abs $QueueRoot
$Logs = Join-Path $QueueRoot "logs"; New-Item -ItemType Directory -Path $Logs -Force | Out-Null
$QueueLog = Join-Path $QueueRoot "queue.log"

function Note([string]$text) { $line="[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $text"; $line | Tee-Object -FilePath $QueueLog -Append }
function Wait-ForGpuIdle([int]$minutes) {
    if ($minutes -le 0) { return }
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) { Note "nvidia-smi unavailable; skipping GPU-idle check"; return }
    $required = $minutes * 60; $idle = 0
    Note "Waiting for $minutes continuous GPU-idle minutes"
    while ($idle -lt $required) {
        $pids = @(& nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>$null) |
            ForEach-Object { $_.Trim() } | Where-Object { $_ -match '^\d+$' }
        if ($pids.Count -eq 0) { $idle += 30 } else { $idle = 0 }
        if ($idle -lt $required) { Start-Sleep -Seconds 30 }
    }
    Note "GPU has been idle for $minutes minutes"
}
function Run-Step([string]$name,[string[]]$arguments,[string]$complete) {
    if (Test-Path -LiteralPath $complete) { Note "SKIP $name (completion file exists)"; return }
    $log=Join-Path $Logs "$name.log"; "COMMAND: $Python $($arguments -join ' ')" | Set-Content -LiteralPath $log -Encoding utf8
    Note "START $name"
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @arguments *>> $log
    $code=$LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($code -ne 0) { Note "FAILED $name (exit $code)"; return }
    if (-not (Test-Path -LiteralPath $complete)) { Note "FAILED $name (completion file absent)"; return }
    Note "COMPLETED $name"
}

if ($WaitForProcessId -gt 0) {
    $watched = Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue
    if ($watched) {
        Note "Waiting for process $WaitForProcessId to finish"
        Wait-Process -Id $WaitForProcessId
        Note "Process $WaitForProcessId finished"
    } else {
        Note "Process $WaitForProcessId is already finished or absent"
    }
}
Wait-ForGpuIdle $GpuIdleMinutes

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $FrameCache -PathType Container)) { throw "Frame cache not found: $FrameCache" }
1..5 | ForEach-Object { if (-not (Test-Path (Join-Path $MasterRoot "fold_$_\master_manifest.csv"))) { throw "Missing fold $_ master manifest" } }
& $Python -c "import torch,transformers,peft,bitsandbytes,PIL,pandas,numpy,scipy,sklearn"; if ($LASTEXITCODE -ne 0) { throw "Python environment preflight failed" }
Note "Pending repaired-cohort queue started"

# Direct VLM zero-shot.
1..5 | ForEach-Object {
    $fold=$_; $out=Join-Path $Project "output\vlm_wd_zero_repaired_cv\fold_$fold"
    Run-Step "vlm_zero_fold_$fold" @("scripts/run_qwen3vl_wd_zero_fewshot_aligned.py","--manifest",(Join-Path $MasterRoot "fold_$fold\master_manifest.csv"),"--frame-cache",$FrameCache,"--shot","zero","--output",$out) (Join-Path $out "summary.json")
}

# Transcript 3+3 few-shot on the repaired folds.
1..5 | ForEach-Object {
    $fold=$_; $out=Join-Path $Project "output\llm_wd_few_repaired_cv\fold_$fold"
    Run-Step "llm_few_fold_$fold" @("scripts/run_qwen3_8b_wd_zero_fewshot_aligned.py","--dataset",(Join-Path $MasterRoot "fold_$fold\master_manifest.jsonl"),"--shot","few","--examples-per-class","3","--output",$out) (Join-Path $out "summary.json")
}

# Direct visual 3+3 few-shot. Four frames per demonstration and 16 target frames
# keep the multimodal context practical on a 32 GB GPU.
1..5 | ForEach-Object {
    $fold=$_; $out=Join-Path $Project "output\vlm_wd_few_repaired_cv\fold_$fold"
    Run-Step "vlm_few_fold_$fold" @("scripts/run_qwen3vl_wd_zero_fewshot_aligned.py","--manifest",(Join-Path $MasterRoot "fold_$fold\master_manifest.csv"),"--frame-cache",$FrameCache,"--shot","few","--examples-per-class","3","--demo-frames","4","--target-frames","16","--output",$out) (Join-Path $out "summary.json")
}

# Continuous VLM regression using the same frozen fold assignments.
1..5 | ForEach-Object {
    $fold=$_; $out=Join-Path $Project "output\vlm_wd_regression_repaired_cv\fold_$fold"
    Run-Step "vlm_regression_fold_$fold" @("scripts/finetune_qwen3vl_wd_only_large.py","--input-manifest",(Join-Path $MasterRoot "fold_$fold\master_manifest.csv"),"--frame-cache",$FrameCache,"--positive-threshold","2","--epochs","1","--num-frames","16","--frame-width","224","--output-dir",$out) (Join-Path $out "final_summary.json")
}

# Combine successful prompt and regression fold predictions.
foreach ($spec in @(
    @{Name="combine_vlm_zero"; Root="output\vlm_wd_zero_repaired_cv"; File="predictions.csv"},
    @{Name="combine_llm_few"; Root="output\llm_wd_few_repaired_cv"; File="predictions.csv"},
    @{Name="combine_vlm_few"; Root="output\vlm_wd_few_repaired_cv"; File="predictions.csv"},
    @{Name="combine_vlm_regression"; Root="output\vlm_wd_regression_repaired_cv"; File="test_predictions.csv"}
)) {
    $root=Join-Path $Project $spec.Root; $combined=Join-Path $root "oof_predictions.csv"
    Run-Step $spec.Name @("scripts/combine_wd_cv_predictions.py","--fold-root",$root,"--filename",$spec.File,"--output",$combined) $combined
}

Note "Pending repaired-cohort queue finished"
