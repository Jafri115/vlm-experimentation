# Qwen3-8B zero-shot patient-withdrawal experiment

This experiment predicts only patient withdrawal (`WD_P`) from each timestamped
German transcript. It uses `Qwen/Qwen3-8B` in BF16 on one CUDA GPU. It does not
run inference during setup, import, or `--prepare-only`.

The prompt operationalizes Eubanks and Muran's *Rupture Resolution Rating System
(3RS): Manual Version 2022*, especially the patient "moves away" construct,
withdrawal forms, boundary rules, and 1/3/5 salience anchors. The source is DOI
`10.13140/RG.2.2.29780.17282`.

The model returns a 1-5 `wd_p_score`. The binary prediction is derived as:

```text
score 1   -> NO_WD_P
score 2-5 -> WD_P
```

This matches the primary human-label rule `mean(WD_P) > 1`. Score 2 explicitly
captures plausible or mild movement away that is above score 1 but lacks one
clear, moderately salient marker. The prior four-class prompt effectively
required clear evidence and severely underpredicted withdrawal.

Copy the new script to the GPU machine and inspect the exact cohort and prompt
without loading the model:

```powershell
python scripts/run_qwen3_8b_wd_3rs_zeroshot.py --prepare-only
```

Run inference only when ready:

```powershell
python scripts/run_qwen3_8b_wd_3rs_zeroshot.py
```

The default output is:

```text
output/qwen3_8b_bf16_wd_3rs_zeroshot/inventory_ready
```

It is intentionally different from the earlier four-class output directory, so
the previous checkpoint remains intact. Resume by running the same command.

After inference, evaluate the binary WD predictions:

```powershell
python scripts/compare_qwen3_8b_wd_to_ground_truth.py `
  --predictions output/qwen3_8b_bf16_wd_3rs_zeroshot/inventory_ready/predictions.csv
```

Evaluation outputs include primary mean-rating metrics, a separate WD-only
two-or-more-rater consensus analysis, provider metrics, row-level matches, false
negatives, false positives, and `score_threshold_sweep.csv`. The sweep evaluates
cutoffs from `score >= 2` through `score >= 5` without rerunning inference. Select
a final cutoff on validation data rather than optimizing and reporting it on the
same test cohort. The default comparison directory is:

```text
output/qwen3_8b_wd_3rs_ground_truth_comparison
```

Transcript-only 3RS coding loses paraverbal and nonverbal information. The
prompt therefore forbids inferring tone, affect, pauses, facial behavior, and
posture unless the transcript explicitly supplies them.
