param(
    [string]$Python = '.\.venv\Scripts\python.exe',
    [string]$MasterRoot = '.\output\wd_multimodal_master_expanded_cohere_ft',
    [string]$QueueRoot = '.\output\wd_cohere_ordinal_replay'
)
$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
function Resolve-RepoPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $Repo $Value))
}
$Python = Resolve-RepoPath $Python
$MasterRoot = Resolve-RepoPath $MasterRoot
$QueueRoot = Resolve-RepoPath $QueueRoot
if (-not (Test-Path -LiteralPath $Python)) { throw "Python missing: $Python" }
if (-not (Test-Path -LiteralPath (Join-Path $MasterRoot 'replacement_audit.json'))) {
    throw "Transcript replacement is not ready: $MasterRoot"
}
New-Item -ItemType Directory -Path $QueueRoot -Force | Out-Null
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$OutLog = Join-Path $QueueRoot "background_$Stamp.out.log"
$ErrLog = Join-Path $QueueRoot "background_$Stamp.err.log"
$QueueArguments = @('-u', (Join-Path $Repo 'scripts\orchestration\run_wd_cohere_ordinal_replay.py'),
    '--master-root', $MasterRoot, '--output', $QueueRoot, '--run')
$QuotedArguments = ($QueueArguments | ForEach-Object { '"' + $_ + '"' }) -join ' '
$Process = Start-Process -FilePath $Python -ArgumentList $QuotedArguments -WorkingDirectory $Repo `
    -WindowStyle Hidden -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -PassThru
Start-Sleep -Seconds 3
$Process.Refresh()
if ($Process.HasExited -and $Process.ExitCode -ne 0) {
    Get-Content -LiteralPath $ErrLog -Tail 40
    throw 'Replay queue failed at startup'
}
Write-Host "Background PID: $($Process.Id)"
Write-Host "Queue output: $OutLog"
Write-Host "Startup errors: $ErrLog"
