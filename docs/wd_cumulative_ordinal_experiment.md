# WD cumulative ordinal experiment (expanded cohort)

**Status:** implemented, not yet run  
**Primary model:** Qwen3-8B transcript model  
**Frozen cohort:** `output/wd_multimodal_master_expanded`  
**Planned results:** `output/llm_wd_cumulative_qwen3_8b_expanded_cv`  
**Last updated:** 2026-09-19

## Why this experiment exists

The five-level WD_P regression model compresses predictions toward the middle. This is not
mainly a prompt problem. The expanded 20-patient cohort has 4,325 segments, but almost no
reliable examples at the top of the scale:

| Evidence | Segments |
| --- | ---: |
| Human mean at least 3 | 449 |
| Both raters at least 3 | 404 |
| Human mean at least 3.5 | 99 |
| Human mean at least 4 | 15 |
| Both raters at least 4 | 13 |
| Exact 4/4 agreement | 10 |
| Either rater used 5 | 7 |
| Exact 5/5 agreement | 0 |

Exact scores 4 and 5 therefore cannot be learned or evaluated reliably. Oversampling the 13
high-score segments would mostly encourage memorization of six patients. Merging 4 and 5 only
renames the sparse category; it does not create new information.

## Research question

Can a transcript model learn ordered, clinically interpretable WD_P severity without requiring
unsupported exact prediction of ratings 4 and 5?

The experiment asks two cumulative questions:

1. Is WD_P at least **2** (possible/mild withdrawal versus rating 1)?
2. Is WD_P at least **3** (clear or stronger withdrawal versus below 3)?

Together, these produce three operational levels: **1**, **2**, and **3+**.

## Human targets

Both ratings are retained. For threshold `k`, the target is:

`target_ge_k = (I(rater1 >= k) + I(rater2 >= k)) / 2`

| Human pair relative to threshold | Soft target |
| --- | ---: |
| Both below | 0.0 |
| One below, one at/above | 0.5 |
| Both at/above | 1.0 |

For the harder `WD >= 3` threshold, the complete expanded cohort contains 2,933 both-below
segments, 988 disagreements, and 404 both-at/above segments. The disagreement rows remain
training data rather than being discarded.

## Model and loss

- Base: `Qwen/Qwen3-8B`, 4-bit QLoRA.
- Input: the same corrected transcript and patient-disjoint five folds used by the expanded
  LLM/VLM comparison.
- Rubric: `manual_detailed_v3`.
- Pooling: last token.
- Head: two outputs, parameterized so `P(WD>=3) <= P(WD>=2)` for every segment.
- Objective: soft binary cross entropy for both thresholds.
- Patient balancing: each training patient has equal total loss weight.
- Positive weights: 1.0 for `WD>=2`; capped 2.5 for `WD>=3`.
- Epochs: 3; checkpoint selected using the mean validation balanced accuracy across both
  threshold tasks.

The 2.5 weight is deliberately moderate. The raw imbalance ratio for `WD>=3` is larger, but
fully compensating it would risk predicting clear withdrawal too often.

## Frozen evaluation

All reported predictions are out-of-fold and patient-disjoint. The fixed decision threshold is
0.5. No test-set threshold tuning is allowed.

Primary outcomes:

1. Balanced accuracy and AUROC for `WD>=3`, evaluated where both humans agree about that
   threshold.
2. Balanced accuracy and AUROC for `WD>=2`, evaluated where both humans agree about that
   threshold.
3. Three-level balanced accuracy for 1/2/3+.
4. MAE and Spearman for `1 + P(WD>=2) + P(WD>=3)` against the human mean capped at 3.

Diagnostics:

- results by transcript provider and fold;
- probability distributions for human-negative, disagreement, and human-positive groups;
- monotonic violations, which must equal zero;
- prediction distribution, to detect another collapse toward a constant.

## Interpretation and decision rule

This experiment succeeds if `WD>=3` discrimination improves over the existing exact-score
regression while `WD>=2` remains useful and results are not driven by one fold or provider.
Compare AUROC, balanced accuracy, probability spread, and patient-level uncertainty; do not
select the conclusion from MAE alone.

This experiment does **not** claim to predict exact ratings 4 or 5. Its valid conclusion is
whether the transcript supports absent/weak, possible, or clear-and-stronger withdrawal.

If this target still collapses, the next action is additional expert annotation of deliberately
sampled high-salience minutes, especially independent 4/5 examples from more patients. Do not
solve that failure by repeatedly increasing class weights on the same 13 examples.

## Run commands

Prepare-only smoke check, which loads no model:

```powershell
python scripts/finetune_qwen3_8b_wd_text.py `
  --dataset output/wd_multimodal_master_expanded/fold_1/master_manifest.jsonl `
  --mode cumulative `
  --patient-balanced `
  --rubric manual_detailed_v3 `
  --output output/cumulative_prepare_smoke `
  --prepare-only
```

Start all five folds in the background:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts/start_wd_cumulative_ordinal_background.ps1
```

Monitor:

```powershell
Get-Content output/wd_cumulative_ordinal_expanded_queue/queue.log -Tail 30 -Wait
```

Expected final files:

- `output/llm_wd_cumulative_qwen3_8b_expanded_cv/oof_predictions.csv`
- `output/wd_cumulative_qwen3_8b_expanded_report/cumulative_ordinal_report.md`
- `output/wd_cumulative_qwen3_8b_expanded_report/cumulative_ordinal_metrics.csv`
- `output/wd_cumulative_qwen3_8b_expanded_report/cumulative_probability_distributions.png`

## Implementation files

- `scripts/finetune_qwen3_8b_wd_text.py`
- `scripts/run_wd_cumulative_ordinal_expanded_queue.ps1`
- `scripts/start_wd_cumulative_ordinal_background.ps1`
- `scripts/report_wd_cumulative_ordinal.py`
- `tests/test_wd_ordinal_regression.py`
