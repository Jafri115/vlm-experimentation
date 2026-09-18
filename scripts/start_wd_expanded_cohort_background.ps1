param(
    [string]$Python = '.\.venv\Scripts\python.exe',
    [string]$VideoRoot = 'C:\Data\Sequence_model\Memopsy_videos\CONVERTED',
    [string]$RoleCache = 'output\qwen3vl_visual_experiment_v5\patient_role_cache.json',
    [string]$YunetModel = 'models\face_detection_yunet\face_detection_yunet_2026may.onnx',
    [string]$Output = 'output\wd_multimodal_master_expanded'
)
$ErrorActionPreference='Stop'
$CohortRepo=Split-Path $PSScriptRoot -Parent
if (-not [IO.Path]::IsPathRooted($Python)) { $Python=Join-Path $CohortRepo $Python }
if (-not [IO.Path]::IsPathRooted($Output)) { $Output=Join-Path $CohortRepo $Output }
if (-not [IO.Path]::IsPathRooted($RoleCache)) { $RoleCache=Join-Path $CohortRepo $RoleCache }
if (-not [IO.Path]::IsPathRooted($YunetModel)) { $YunetModel=Join-Path $CohortRepo $YunetModel }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
New-Item -ItemType Directory -Path $Output -Force | Out-Null
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$RunArgs=@('-u',('"{0}"' -f (Join-Path $PSScriptRoot 'build_wd_expanded_multimodal_cohort.py')),
    '--video-root',('"{0}"' -f $VideoRoot),'--output',('"{0}"' -f $Output),
    '--role-cache',('"{0}"' -f $RoleCache),'--yunet-model',('"{0}"' -f $YunetModel),
    '--frame-cache',('"{0}"' -f (Join-Path $Output 'frame_cache_16')))
$Launch=@{FilePath=$Python;ArgumentList=$RunArgs;WorkingDirectory=$CohortRepo;WindowStyle='Hidden';
    RedirectStandardOutput=(Join-Path $Output "build_$Stamp.out.log");
    RedirectStandardError=(Join-Path $Output "build_$Stamp.err.log");PassThru=$true}
$CohortProcess=Start-Process @Launch
Write-Host "Background PID: $($CohortProcess.Id)"
Write-Host "Output log: $($Launch.RedirectStandardOutput)"
Write-Host "Error log: $($Launch.RedirectStandardError)"
