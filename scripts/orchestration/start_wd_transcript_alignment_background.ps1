param(
    [string]$Python = '.\.venv\Scripts\python.exe',
    [string]$MediaRoot = '',
    [string]$AudioMap = '',
    [string]$Model = 'large-v3',
    [int]$MaxSessions = 0
)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
function RepoPath([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
    return [IO.Path]::GetFullPath((Join-Path $Repo $Path))
}
$Python = RepoPath $Python
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
$Root = Join-Path $Repo 'output\wd_transcript_variants_aligned_v1'
New-Item -ItemType Directory -Path $Root -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$OutLog = Join-Path $Root "background_$stamp.out.log"
$ErrLog = Join-Path $Root "background_$stamp.err.log"
$QueueArgs = @('-u', (Join-Path $Repo 'scripts\data\align_wd_transcript_variants.py'), '--model', $Model)
if ($MediaRoot) { $QueueArgs += @('--media-root', (RepoPath $MediaRoot)) }
if ($AudioMap) { $QueueArgs += @('--audio-map', (RepoPath $AudioMap)) }
if ($MaxSessions -gt 0) { $QueueArgs += @('--max-sessions', "$MaxSessions") }
# Start-Process joins arguments into one Windows command line; quote paths.
$QuotedArgs = $QueueArgs | ForEach-Object {
    if ($_ -match '"') { throw 'Arguments containing quotes are unsupported' }
    '"' + $_ + '"'
}
$Process = Start-Process -FilePath $Python -ArgumentList $QuotedArgs -WorkingDirectory $Repo `
    -WindowStyle Hidden -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -PassThru
Write-Host "Alignment PID: $($Process.Id)"
Write-Host "Output log: $OutLog"
Write-Host "Error log: $ErrLog"
Start-Sleep -Seconds 3
$Process.Refresh()
if ($Process.HasExited -and $Process.ExitCode -ne 0) {
    Get-Content -LiteralPath $ErrLog -Tail 30
    throw 'Alignment failed during startup'
}
