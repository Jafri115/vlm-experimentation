# Better transcript regression experiment

This experiment tests whether a stronger transcript model improves continuous
WD_P prediction on the expanded shared cohort without changing the frozen
patient folds.

## Design

- **Primary model:** `Qwen/Qwen3-14B`, NF4 QLoRA.
- **Objective control:** `Qwen/Qwen3-8B` with the identical input, loss, folds,
  epochs, optimizer settings, LoRA rank, seed, and evaluation.
- **Target:** a five-level distribution made directly from the two ratings.
  Ratings 1 and 3 produce `[0.5, 0, 0.5, 0, 0]`; equal ratings produce a one-hot
  target. The continuous prediction is the expected rating from 1 to 5.
- **Patient balance:** each training patient has equal total loss weight, so a
  patient with many segments cannot dominate the fit.
- **Representation:** the last non-padding token, which has attended to the
  complete transcript and instruction.
- **Primary endpoint:** out-of-fold MAE against the mean of the two human WD_P
  ratings. Checkpoints are selected by validation MAE only.
- **Secondary endpoints:** RMSE, Spearman correlation, within-patient centered
  Spearman, prediction/target SD ratio, exact-agreement performance, and
  performance on human-disagreement rows.
- **Uncertainty:** paired patient-cluster bootstrap for the MAE difference.

The 14B model counts as a clear improvement when its MAE is lower than the 8B
ordinal control and the paired interval is mostly or entirely above zero. A
higher Spearman value plus an SD ratio closer to 1 supports that it learned
severity variation instead of predicting the average.

## Run on the 32 GB GPU machine

After pulling the commit, start one hidden background queue from the repository
root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\start_wd_better_llm_regression_background.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -MasterRoot .\output\wd_multimodal_master_expanded
```

Monitor it with:

```powershell
Get-Content .\output\wd_better_llm_regression_queue\queue.log -Tail 30 -Wait
```

If a job fails, inspect its named log under
`output\wd_better_llm_regression_queue\logs`. The queue stops at the first real
failure and safely skips folds that already have `final_summary.json` when it is
started again.

## Read the result

```powershell
Get-Content .\output\wd_ordinal_regression_expanded_report\ordinal_regression_report.md
```

The report also writes:

- `ordinal_regression_metrics.csv`
- `human_reference.csv`
- `qwen14_paired_mae_gains.csv`
- `ordinal_regression_summary.png`
- `ordinal_training_validation.png`

To add the earlier scalar 8B run to the same report after the expanded overnight
suite finishes:

```powershell
python .\scripts\report_wd_ordinal_regression.py `
  --extra "Qwen3-8B scalar=output/llm_wd_regression_expanded_cv/oof_predictions.csv"
```
