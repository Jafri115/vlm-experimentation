# Transcript LLM experiments aligned with the WD_P VLM experiments

## Scientific comparison

The primary target is patient withdrawal (`WD_P`) at the 3RS threshold `>=2`.
Keep these factors identical between modalities:

- physical segment IDs;
- raw coder ratings and target construction;
- patient-disjoint train, validation, and test assignments;
- consensus exclusions or soft-label inclusion;
- fixed binary probability threshold `0.5`;
- checkpoint-selection metric;
- test metrics and patient-level uncertainty calculation.

Only the input modality and model change:

- VLM: patient-only video frames and Qwen3-VL;
- LLM: timestamped `T:`/`P:` transcript and Qwen3-8B.

Do not compare headline metrics from different cohorts. Build the LLM data from
the exact VLM manifest for each experiment. Rows without a ready transcript are
reported and excluded. For a fully paired claim, also restrict the saved VLM
predictions to the resulting common segment IDs.

## Environment

Use a separate environment from the GPTQ/VLM environment to avoid its pinned
`transformers` dependency:

```powershell
py -3.11 -m venv .venv-llm-ft
.\.venv-llm-ft\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the CUDA-enabled PyTorch build appropriate for the machine first.
python -m pip install -r requirements_qwen3_text_finetune.txt
```

## 1. Build an exact common-modality dataset

First audit the saved VLM artifacts. This distinguishes completed fold
manifests from preparation and smoke-test copies and prints the target and
split counts:

```powershell
python scripts/audit_vlm_wd_manifests.py --output-root output
```

Use a row marked `usable fixed split` for a single train/validation/test
experiment. For grouped cross-validation, use each `fold_N/cv_manifest.csv`.
The full audit is saved as `output/vlm_wd_manifest_audit.csv`.

For a fixed VLM split, point `--vlm-manifest` to the manifest actually used by
that VLM run. It must contain `patient_id`, `session_id`, `segment_id`, and the
original `split` column.

```powershell
python scripts/build_llm_wd_aligned_dataset.py `
  --vlm-manifest output/YOUR_VLM_RUN/manifest_with_binary_targets.csv `
  --output output/llm_wd_aligned_fixed
```

Inspect:

```powershell
Get-Content output/llm_wd_aligned_fixed/dataset_summary.json
```

Use ready transcripts for the primary experiment. `--allow-review-transcripts`
is an explicit sensitivity analysis because those rows can contain unresolved
roles, timing, or cleaning issues.

For VLM grouped cross-validation, align each saved VLM fold manifest separately:

```powershell
1..5 | ForEach-Object {
  python scripts/build_llm_wd_aligned_dataset.py `
    --vlm-manifest "output/YOUR_VLM_GROUPCV/fold_$_/cv_manifest.csv" `
    --output "output/llm_wd_aligned_cv/fold_$_"
}
```

## 2. Zero-shot and 3+3 few-shot

The zero-shot and few-shot commands use the identical patient-disjoint test
rows. Few-shot demonstrations are deterministically selected from the training
patients: three consensus-negative and three consensus-positive examples.

Preparation checks do not load Qwen:

```powershell
python scripts/run_qwen3_8b_wd_zero_fewshot_aligned.py `
  --dataset output/llm_wd_aligned_fixed/aligned_manifest.jsonl `
  --shot zero `
  --output output/llm_wd_prompt/zero `
  --prepare-only

python scripts/run_qwen3_8b_wd_zero_fewshot_aligned.py `
  --dataset output/llm_wd_aligned_fixed/aligned_manifest.jsonl `
  --shot few `
  --output output/llm_wd_prompt/few_3plus3 `
  --prepare-only
```

Run when ready:

```powershell
python scripts/run_qwen3_8b_wd_zero_fewshot_aligned.py `
  --dataset output/llm_wd_aligned_fixed/aligned_manifest.jsonl `
  --shot zero `
  --output output/llm_wd_prompt/zero

python scripts/run_qwen3_8b_wd_zero_fewshot_aligned.py `
  --dataset output/llm_wd_aligned_fixed/aligned_manifest.jsonl `
  --shot few `
  --examples-per-class 3 `
  --output output/llm_wd_prompt/few_3plus3
```

For a fair VLM/LLM zero-shot comparison, rerun or re-evaluate the VLM on the
same aligned test IDs, with the same 3RS `>=2` boundary. The earlier balanced-94
visual prompt experiment answers a different cohort question unless those IDs
exactly match this manifest.

## 3. Continuous WD_P regression

This mirrors the regression-head experiment with mean coder score as the target,
SmoothL1 loss, mean token pooling, LoRA, and the unchanged patient split:

```powershell
python scripts/finetune_qwen3_8b_wd_text.py `
  --dataset output/llm_wd_aligned_regression/aligned_manifest.jsonl `
  --mode regression `
  --output output/qwen3_8b_wd_text_regression `
  --prepare-only

python scripts/finetune_qwen3_8b_wd_text.py `
  --dataset output/llm_wd_aligned_regression/aligned_manifest.jsonl `
  --mode regression `
  --output output/qwen3_8b_wd_text_regression
```

Use the original VLM regression `pilot_manifest.csv` when building
`llm_wd_aligned_regression`. Compare MAE, Spearman correlation, and predicted
range. Keep this experiment even if regression again collapses; it directly
tests whether lexical information recovers salience variation.

## 4. Consensus-only binary WD_P

Both raters below 2 produce 0; both raters at or above 2 produce 1; disagreements
are excluded from training, validation, and testing. The model outputs one logit,
uses weighted BCE, selects the checkpoint on validation AUPRC, and applies the
predeclared probability threshold 0.5.

```powershell
python scripts/finetune_qwen3_8b_wd_text.py `
  --dataset output/llm_wd_aligned_fixed/aligned_manifest.jsonl `
  --mode consensus `
  --output output/qwen3_8b_wd_text_consensus `
  --prepare-only

python scripts/finetune_qwen3_8b_wd_text.py `
  --dataset output/llm_wd_aligned_fixed/aligned_manifest.jsonl `
  --mode consensus `
  --output output/qwen3_8b_wd_text_consensus
```

## 5. Soft-label WD_P

Training targets retain rater disagreement:

- both negative: `0`;
- disagreement: `0.5`;
- both positive: `1`.

Checkpoint selection and primary testing remain on consensus rows so the result
is directly comparable to consensus-only training.

```powershell
python scripts/finetune_qwen3_8b_wd_text.py `
  --dataset output/llm_wd_aligned_fixed/aligned_manifest.jsonl `
  --mode soft `
  --output output/qwen3_8b_wd_text_soft `
  --prepare-only

python scripts/finetune_qwen3_8b_wd_text.py `
  --dataset output/llm_wd_aligned_fixed/aligned_manifest.jsonl `
  --mode soft `
  --output output/qwen3_8b_wd_text_soft
```

## 6. Patient-grouped cross-validation

Use the exact saved `cv_manifest.csv` from each VLM fold. This preserves outer
test patients and inner validation patients rather than regenerating folds.

```powershell
1..5 | ForEach-Object {
  python scripts/finetune_qwen3_8b_wd_text.py `
    --dataset "output/llm_wd_aligned_cv/fold_$_/aligned_manifest.jsonl" `
    --mode consensus `
    --output "output/qwen3_8b_wd_text_consensus_cv/fold_$_"
}
```

Repeat with `--mode soft` into a different output root. Start with fold 1 and
inspect memory use and outputs before launching every fold.

## 7. Paired VLM-versus-LLM comparison

The fine-tuning scripts save `test_predictions.csv` with `WD_probability`.
Compare predictions only on identical segment IDs:

```powershell
python scripts/compare_vlm_llm_wd_predictions.py `
  --vlm-predictions output/YOUR_VLM_RUN/test_predictions.csv `
  --llm-predictions output/qwen3_8b_wd_text_consensus/test_predictions.csv `
  --output output/comparison_vlm_vs_llm_consensus
```

The comparison reports both metric sets, LLM-minus-VLM differences, an exact
paired McNemar test, and a patient-cluster bootstrap 95% interval for the
balanced-accuracy difference. The latter respects dependence among segments
from the same patient.

For cross-validation, first concatenate the five test prediction files for each
modality into one out-of-fold prediction CSV, then run the same paired command.

## Recommended experiment order

1. Build and audit the common cohort.
2. Run zero-shot and 3+3 few-shot on the fixed test set.
3. Run regression once to test score-range compression.
4. Run consensus binary on the fixed split.
5. Run soft-label training on that identical split.
6. Run the five patient-grouped folds for the strongest fixed-split model.
7. Report paired VLM/LLM differences on identical segments, with patient-level
   uncertainty.

Do not use the test set to select probability thresholds, prompts, epochs, or
hyperparameters. Use validation data for those decisions and retain 0.5 as the
primary fixed threshold when reproducing the reported VLM experiments.
