param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$Destination = '.\output\wd_expanded_completed_results_for_transfer'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$OutputRoot = Join-Path $ProjectRoot 'output'
if (-not (Test-Path -LiteralPath $OutputRoot -PathType Container)) {
    throw "Output directory not found: $OutputRoot"
}

if (-not [IO.Path]::IsPathRooted($Destination)) {
    $Destination = Join-Path $ProjectRoot $Destination
}
$Destination = [IO.Path]::GetFullPath($Destination)
$resolvedOutput = [IO.Path]::GetFullPath($OutputRoot).TrimEnd('\') + '\'
if (-not ($Destination + '\').StartsWith($resolvedOutput, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Destination must be inside the project output directory: $Destination"
}
if ($Destination.TrimEnd('\').Equals($OutputRoot.TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Destination cannot be the output root itself.'
}

if (Test-Path -LiteralPath $Destination) {
    Remove-Item -LiteralPath $Destination -Recurse -Force
}
New-Item -ItemType Directory -Path $Destination -Force | Out-Null

$specs = @(
    # Main 20-patient expanded experiment suite.
    @{Name='LLM zero-shot'; Folder='llm_wd_zero_expanded_cv'; Marker='oof_predictions.csv'; Group='expanded main'},
    @{Name='LLM 3+3 few-shot'; Folder='llm_wd_few_expanded_cv'; Marker='oof_predictions.csv'; Group='expanded main'},
    @{Name='LLM regression'; Folder='llm_wd_regression_expanded_cv'; Marker='oof_predictions.csv'; Group='expanded main'},
    @{Name='LLM consensus fine-tuning'; Folder='llm_wd_consensus_expanded_cv'; Marker='oof_predictions.csv'; Group='expanded main'},
    @{Name='LLM soft-label fine-tuning'; Folder='llm_wd_soft_expanded_cv'; Marker='oof_predictions.csv'; Group='expanded main'},
    @{Name='VLM consensus fine-tuning'; Folder='vlm_wd_consensus_expanded_paired_cv'; Marker='oof_predictions.csv'; Group='expanded main'},
    @{Name='VLM soft-label fine-tuning'; Folder='vlm_wd_soft_expanded_paired_cv'; Marker='oof_predictions.csv'; Group='expanded main'},
    @{Name='Paired consensus comparison'; Folder='paired_comparison_consensus_expanded'; Marker='comparison_summary.json'; Group='expanded main'},
    @{Name='Paired soft-label comparison'; Folder='paired_comparison_soft_expanded'; Marker='comparison_summary.json'; Group='expanded main'},
    @{Name='Training and validation plots'; Folder='wd_training_plots_expanded'; Marker='llm_consensus_train_validation.png'; Group='expanded main'},

    # Better ordinal-regression experiment, already complete before the main queue.
    @{Name='Qwen3-8B ordinal regression'; Folder='llm_wd_ordinal_qwen3_8b_expanded_cv'; Marker='oof_predictions.csv'; Group='ordinal regression'},
    @{Name='Qwen3-14B ordinal regression'; Folder='llm_wd_ordinal_qwen3_14b_expanded_cv'; Marker='oof_predictions.csv'; Group='ordinal regression'},
    @{Name='Ordinal regression report'; Folder='wd_ordinal_regression_expanded_report'; Marker=''; Group='ordinal regression'},

    # Second phase of the expanded parent queue. These are included only when complete.
    @{Name='VLM zero-shot'; Folder='vlm_wd_zero_expanded_cv'; Marker='oof_predictions.csv'; Group='pending visual phase'},
    @{Name='VLM 3+3 few-shot'; Folder='vlm_wd_few_expanded_cv'; Marker='oof_predictions.csv'; Group='pending visual phase'},
    @{Name='VLM regression'; Folder='vlm_wd_regression_expanded_cv'; Marker='oof_predictions.csv'; Group='pending visual phase'},

    # Cohort summaries and presentation-ready cohort plots.
    @{Name='Expanded cohort overview'; Folder='wd_expanded_cohort_overview'; Marker=''; Group='cohort'},
    @{Name='Expanded label distribution'; Folder='wd_expanded_label_distribution'; Marker=''; Group='cohort'}
)

$presentationExtensions = @('.png', '.jpg', '.jpeg', '.svg', '.md')
$compactFileNames = @(
    'oof_predictions.csv',
    'paired_predictions.csv',
    'final_summary.json',
    'summary.json',
    'comparison_summary.json',
    'cv_summary.json',
    'run_config.json',
    'preparation.json',
    'training_history.csv',
    'learning_curves.csv',
    'test_metrics.json',
    'metrics.json',
    'regression_metrics.json',
    'command.txt',
    'queue.log'
)
$excludedDirectories = @('best_adapter', 'checkpoint', 'checkpoints', 'frame_cache_16', '__pycache__')
$statusRows = New-Object System.Collections.Generic.List[object]
$manifestRows = New-Object System.Collections.Generic.List[object]

function Copy-LightweightResultFolder {
    param([string]$FolderName)

    $source = Join-Path $OutputRoot $FolderName
    $targetRoot = Join-Path $Destination $FolderName
    $reportFolder = $FolderName -match '(report|overview|distribution)$'
    $copied = 0
    Get-ChildItem -LiteralPath $source -Recurse -File | Where-Object {
        $relative = $_.FullName.Substring($source.Length).TrimStart('\', '/')
        $parts = $relative -split '[\\/]'
        $blocked = @($parts | Where-Object { $excludedDirectories -contains $_ }).Count -gt 0
        $extension = $_.Extension.ToLowerInvariant()
        $compactResult = $compactFileNames -contains $_.Name
        $presentationFile = $presentationExtensions -contains $extension
        $reportTable = $reportFolder -and ($extension -in @('.csv', '.json', '.txt'))
        (-not $blocked) -and ($compactResult -or $presentationFile -or $reportTable)
    } | ForEach-Object {
        $relative = $_.FullName.Substring($source.Length).TrimStart('\', '/')
        $target = Join-Path $targetRoot $relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $target -Force
        $copied++
        $manifestRows.Add([pscustomobject]@{
            folder = $FolderName
            relative_path = (Join-Path $FolderName $relative)
            bytes = $_.Length
            modified = $_.LastWriteTime.ToString('o')
        })
    }
    return $copied
}

foreach ($spec in $specs) {
    $source = Join-Path $OutputRoot $spec.Folder
    $folderExists = Test-Path -LiteralPath $source -PathType Container
    $complete = $false
    $markerPath = ''
    if ($folderExists) {
        if ([string]::IsNullOrWhiteSpace($spec.Marker)) {
            $complete = @(Get-ChildItem -LiteralPath $source -File -ErrorAction SilentlyContinue).Count -gt 0
        } else {
            $markerPath = Join-Path $source $spec.Marker
            $complete = Test-Path -LiteralPath $markerPath -PathType Leaf
            if (-not $complete) {
                # Some transferred historical runs contain one extra directory named
                # like the experiment root. Accept the aggregate marker anywhere below
                # the root while still requiring the combined OOF/summary filename.
                $markerLeaf = Split-Path -Leaf $spec.Marker
                $complete = @(Get-ChildItem -LiteralPath $source -Recurse -File -Filter $markerLeaf -ErrorAction SilentlyContinue).Count -gt 0
            }
        }
    }

    $copied = 0
    if ($complete) {
        $copied = Copy-LightweightResultFolder -FolderName $spec.Folder
    }
    $statusRows.Add([pscustomobject]@{
        group = $spec.Group
        experiment = $spec.Name
        source_folder = $spec.Folder
        status = if ($complete) { 'complete_and_collected' } elseif ($folderExists) { 'incomplete' } else { 'not_found' }
        completion_marker = $spec.Marker
        files_copied = $copied
    })
}

# Queue logs are diagnostic evidence and safe to copy while the second phase is running.
foreach ($queueFolder in @('wd_overnight_queue_expanded', 'wd_pending_expanded_queue', 'wd_better_llm_regression_queue')) {
    $source = Join-Path $OutputRoot $queueFolder
    if (Test-Path -LiteralPath $source -PathType Container) {
        $null = Copy-LightweightResultFolder -FolderName $queueFolder
    }
}

# Preserve only compact cohort metadata, not transcript manifests or media.
$cohortSource = Join-Path $OutputRoot 'wd_multimodal_master_expanded'
if (Test-Path -LiteralPath $cohortSource -PathType Container) {
    $cohortTarget = Join-Path $Destination 'wd_multimodal_master_expanded'
    New-Item -ItemType Directory -Path $cohortTarget -Force | Out-Null
    foreach ($name in @('candidate_summary.json', 'cohort_summary.json', 'build_summary.json')) {
        $sourceFile = Join-Path $cohortSource $name
        if (Test-Path -LiteralPath $sourceFile -PathType Leaf) {
            Copy-Item -LiteralPath $sourceFile -Destination (Join-Path $cohortTarget $name) -Force
        }
    }
}

$statusPath = Join-Path $Destination 'collection_status.csv'
$manifestPath = Join-Path $Destination 'collection_manifest.csv'
$statusRows | Export-Csv -LiteralPath $statusPath -NoTypeInformation -Encoding utf8
$manifestRows | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding utf8

$completeRows = @($statusRows | Where-Object status -eq 'complete_and_collected')
$pendingRows = @($statusRows | Where-Object status -ne 'complete_and_collected')
$totalBytes = ($manifestRows | Measure-Object -Property bytes -Sum).Sum
if ($null -eq $totalBytes) { $totalBytes = 0 }

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add('# Expanded WD_P results collected for transfer')
$lines.Add('')
$lines.Add("Collected: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$lines.Add('')
$lines.Add("Completed experiment folders collected: $($completeRows.Count)")
$lines.Add("Files copied: $($manifestRows.Count)")
$lines.Add(('Approximate size: {0:N1} MB' -f ($totalBytes / 1MB)))
$lines.Add('')
$lines.Add('## Completed and copied')
$lines.Add('')
foreach ($row in $completeRows) { $lines.Add("- $($row.experiment): output/$($row.source_folder)") }
$lines.Add('')
$lines.Add('## Not yet complete or unavailable')
$lines.Add('')
if ($pendingRows.Count -eq 0) {
    $lines.Add('- None')
} else {
    foreach ($row in $pendingRows) { $lines.Add("- $($row.experiment): $($row.status)") }
}
$lines.Add('')
$lines.Add('Copy this entire directory to the matching project output directory on the destination machine.')
$lines | Set-Content -LiteralPath (Join-Path $Destination 'README.md') -Encoding utf8

Write-Host "Completed experiment folders collected: $($completeRows.Count)"
Write-Host "Files copied: $($manifestRows.Count)"
Write-Host ('Approximate size: {0:N1} MB' -f ($totalBytes / 1MB))
Write-Host "Transfer folder: $Destination"
if ($pendingRows.Count -gt 0) {
    Write-Host 'Incomplete/not-found experiments:'
    $pendingRows | Format-Table experiment, status, source_folder -AutoSize
}
