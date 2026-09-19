# Fast WD model development

Use this workflow for prompt, target, loss, context, and dataset experiments before committing GPU time to five-fold cross-validation.

## Rules

- Development always uses one fixed patient-disjoint fold (fold 1 by default).
- Model selection uses validation metrics only.
- `--skip-test-evaluation` prevents inference on the held-out test patients.
- Give every run a new, descriptive experiment name. Existing runs are never overwritten.
- Compare only runs with the same validation patients and row count.
- Run folds 2 and 3 only after a candidate passes the development gate.
- Run all five folds only after the configuration is frozen.

For regression, promote a candidate when validation MAE improves by at least 0.02 or Spearman improves by at least 0.05, while the prediction range does not become narrower. These gates are practical screening rules, not significance tests.

## Establish the fast baseline

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_wd_fast_dev.ps1 `
  -ExperimentName baseline_regression `
  -Mode regression `
  -Rubric legacy_short_v1 `
  -Epochs 1
```

## Compare a prompt

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_wd_fast_dev.ps1 `
  -ExperimentName manual_prompt_regression `
  -Mode regression `
  -Rubric manual_compact_v2 `
  -Epochs 1
```

## Compare another objective

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_wd_fast_dev.ps1 `
  -ExperimentName cumulative_v1 `
  -Mode cumulative `
  -Rubric manual_compact_v2 `
  -PatientBalanced `
  -Epochs 1
```

Extra trainer options may be appended after the named PowerShell parameters:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_wd_fast_dev.ps1 `
  -ExperimentName ordinal_short_context `
  -Mode ordinal `
  -Rubric manual_compact_v2 `
  -PatientBalanced `
  --max-length 1024 --learning-rate 3e-5
```

For a changed dataset, pass its master cohort root. It must contain the same `fold_N/master_manifest.jsonl` structure:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_wd_fast_dev.ps1 `
  -ExperimentName expanded_labels_v2 `
  -MasterRoot output\wd_multimodal_master_labels_v2 `
  -Mode regression `
  -Epochs 1
```

Review:

```powershell
Get-Content output\wd_fast_dev\development_comparison.md
```

To launch the same experiment in the background, replace `run_wd_fast_dev.ps1` with `start_wd_fast_dev_background.ps1`. The launcher prints its PID and separate output/error log paths.

If a dataset change alters validation membership, the report marks it non-comparable. Rebuild a shared split or evaluate both models on common validation IDs before claiming improvement.
