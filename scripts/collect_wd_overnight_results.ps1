param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$Destination = "wd_overnight_results_for_summary.zip"
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path -LiteralPath $ProjectRoot).Path
$output = Join-Path $project "output"
if (-not (Test-Path -LiteralPath $output -PathType Container)) {
    throw "Output directory not found: $output"
}

$destinationPath = if ([IO.Path]::IsPathRooted($Destination)) {
    $Destination
} else {
    Join-Path $project $Destination
}

$staging = Join-Path ([IO.Path]::GetTempPath()) ("wd-results-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $staging | Out-Null

try {
    $roots = Get-ChildItem -LiteralPath $output -Directory | Where-Object {
        $_.Name -match "^(llm_wd_|vlm_wd_|paired_comparison_|wd_training_plots|wd_overnight_queue)"
    }

    $fileNames = @(
        "comparison_summary.json",
        "cv_summary.json",
        "final_summary.json",
        "summary.json",
        "run_config.json",
        "preparation.json",
        "patient_split.json",
        "training_history.csv",
        "learning_curves.csv",
        "test_metrics.json",
        "metrics.json",
        "regression_metrics.json"
    )

    $collected = New-Object System.Collections.Generic.List[object]
    foreach ($root in $roots) {
        Get-ChildItem -LiteralPath $root.FullName -Recurse -File | Where-Object {
            $fileNames -contains $_.Name
        } | ForEach-Object {
            $relative = $_.FullName.Substring($project.Length).TrimStart([char]'\', [char]'/')
            $target = Join-Path $staging $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $target
            $collected.Add([pscustomobject]@{
                path = $relative
                bytes = $_.Length
                modified = $_.LastWriteTime.ToString("o")
            })
        }
    }

    $manifest = [ordered]@{
        project = $project
        collected_at = (Get-Date).ToString("o")
        file_count = $collected.Count
        files = $collected
    }
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $staging "collection_manifest.json") -Encoding utf8

    if (Test-Path -LiteralPath $destinationPath) {
        Remove-Item -LiteralPath $destinationPath -Force
    }
    Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $destinationPath -CompressionLevel Optimal
    Write-Host "Collected $($collected.Count) result files."
    Write-Host "Archive: $destinationPath"
} finally {
    if (Test-Path -LiteralPath $staging) {
        Remove-Item -LiteralPath $staging -Recurse -Force
    }
}
