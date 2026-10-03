param(
    [string]$Python = '.\.venv\Scripts\python.exe',
    [string]$ReleaseRoot = 'C:\Data\Sequence_model\german-asr-pipeline\artifacts\memopsy_196_dataset_versions_v1',
    [string]$OriginalMaster = '.\output\wd_multimodal_master_expanded',
    [string]$DatasetRoot = '.\output\wd_cohere_available_subset',
    [string]$QueueRoot = '.\output\wd_cohere_available_queue',
    [switch]$WithOriginalControl,
    [switch]$FillUnknownRoles,
    [ValidateSet('plain','timestamped_cues')][string]$TurnStyle = 'plain'
)
$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
function Resolve-RepoPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $Repo $Value))
}
$Python=Resolve-RepoPath $Python
$ReleaseRoot=Resolve-RepoPath $ReleaseRoot
$OriginalMaster=Resolve-RepoPath $OriginalMaster
$DatasetRoot=Resolve-RepoPath $DatasetRoot
$QueueRoot=Resolve-RepoPath $QueueRoot
foreach ($Required in @($Python,$ReleaseRoot,$OriginalMaster)) {
    if (-not (Test-Path -LiteralPath $Required)) { throw "Missing required path: $Required" }
}
New-Item -ItemType Directory -Path $QueueRoot -Force | Out-Null
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$OutLog=Join-Path $QueueRoot "background_$Stamp.out.log"
$ErrLog=Join-Path $QueueRoot "background_$Stamp.err.log"
$QueueArguments=@('-u',(Join-Path $Repo 'scripts\orchestration\run_wd_cohere_subset_suite.py'),
    '--release-root',$ReleaseRoot,'--original-master',$OriginalMaster,
    '--dataset-root',$DatasetRoot,'--queue-root',$QueueRoot,'--turn-style',$TurnStyle)
if ($WithOriginalControl) { $QueueArguments += '--with-original-control' }
if ($FillUnknownRoles) { $QueueArguments += '--fill-unknown-roles' }
$QuotedArguments=($QueueArguments | ForEach-Object { '"'+$_+'"' }) -join ' '
$Process=Start-Process -FilePath $Python -ArgumentList $QuotedArguments -WorkingDirectory $Repo `
    -WindowStyle Hidden -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -PassThru
Start-Sleep -Seconds 3
$Process.Refresh()
if ($Process.HasExited -and $Process.ExitCode -ne 0) {
    Get-Content -LiteralPath $ErrLog -Tail 50
    throw 'Cohere subset queue failed at startup'
}
Write-Host "Background PID: $($Process.Id)"
Write-Host "Queue output: $OutLog"
Write-Host "Startup errors: $ErrLog"
