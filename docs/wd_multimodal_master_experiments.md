# WD_P multimodal master cohort and experiment status

## Canonical cohorts

`output/wd_multimodal_master/visual_master_soft_2512.csv` reproduces the
slide cohort: 743 negative, 735 rater-disagreement, and 1,034 positive rows.
Its consensus subset has 1,777 rows (743 negative and 1,034 positive).

The primary paired benchmark is `paired_master_soft.csv`: 1,151 rows from 14
patients for which both cached visual input and a ready timestamped transcript
exist. Its consensus evaluation subset contains 796 rows (381 negative and 415
positive). Both VLM and LLM must use the manifests in `fold_1` through `fold_5`.

The folds use the same scheme as the VLM group-CV scripts: five outer
patient-grouped test folds, four inner patient-grouped validation folds, and
seed 42. Patients containing only disagreement rows are deterministically
placed into the smallest outer folds; the original consensus-based splitter
left those patients permanently in training. Every paired segment now occurs
in one outer test fold.

The frozen long-form ratings are in `frozen_rater_labels_long.csv`. Use this
file for VLM reruns so later edits to the annotation source cannot change the
slide labels.

## What has already been completed

According to the supplied slides, the earlier VLM work completed:

- a 5-positive/5-negative qualitative review;
- zero-shot and 3+3 few-shot inference on a balanced 94-segment experiment;
- continuous WD_P/CF_P regression on an earlier held-out cohort;
- consensus binary WD_P training/evaluation on the 1,777-row cached cohort;
- soft-label training on 2,512 rows with consensus evaluation on 1,777 rows.

Those results establish the earlier VLM findings, but the balanced-94 and
regression cohorts differ from the new paired benchmark. The local planning
folder contains cohort/cache artifacts but no saved fold predictions or
patient split files, so it cannot support a retrospective paired VLM/LLM test.

On the LLM side, Qwen3-8B WD_P zero-shot inference was completed on 1,538
inventory-ready transcripts. It achieved balanced accuracy 0.606 on the
mean-rating target and 0.612 on 1,039 strict-consensus rows. This is a useful
preliminary result, but it is not paired to the VLM slide cohort.

The common master cohort, target freeze, five fold assignments, and fold-1
preparation checks for zero-shot, 3+3 few-shot, regression, consensus, and soft
training are now complete. No model inference or training was run while
preparing them.

## Experiments still required on the paired benchmark

1. Run transcript zero-shot and 3+3 few-shot over all five outer test folds.
2. Run Qwen3-8B transcript WD_P regression over all five folds.
3. Fine-tune Qwen3-8B consensus binary WD_P over all five folds.
4. Fine-tune Qwen3-8B with soft labels over the same five folds and evaluate
   consensus rows.
5. Rerun the corresponding VLM methods on the same paired fold manifests.
6. Concatenate each method's outer-fold test predictions and perform paired
   VLM-versus-LLM comparisons with patient-cluster bootstrap intervals.

## LLM commands

Run one fold first. Omit `--prepare-only` to train:

```powershell
python scripts/finetune_qwen3_8b_wd_text.py `
  --dataset output/wd_multimodal_master/fold_1/master_manifest.jsonl `
  --mode consensus `
  --output output/llm_wd_consensus_cv/fold_1
```

Run all folds for consensus and then repeat with `soft` and `regression`:

```powershell
1..5 | ForEach-Object {
  python scripts/finetune_qwen3_8b_wd_text.py `
    --dataset "output/wd_multimodal_master/fold_$_/master_manifest.jsonl" `
    --mode consensus `
    --output "output/llm_wd_consensus_cv/fold_$_"
}
```

Zero-shot and few-shot:

```powershell
1..5 | ForEach-Object {
  python scripts/run_qwen3_8b_wd_zero_fewshot_aligned.py `
    --dataset "output/wd_multimodal_master/fold_$_/master_manifest.jsonl" `
    --shot zero `
    --output "output/llm_wd_zero_cv/fold_$_"

  python scripts/run_qwen3_8b_wd_zero_fewshot_aligned.py `
    --dataset "output/wd_multimodal_master/fold_$_/master_manifest.jsonl" `
    --shot few `
    --examples-per-class 3 `
    --output "output/llm_wd_few_cv/fold_$_"
}
```

## VLM commands

Use the VLM environment and its original frame cache. For consensus training:

```powershell
1..5 | ForEach-Object {
  python scripts/finetune_qwen3vl_wd_consensus_binary.py `
    --labels-csv output/wd_multimodal_master/frozen_rater_labels_long.csv `
    --manifest "output/wd_multimodal_master/fold_$_/master_manifest.csv" `
    --target-mode consensus `
    --positive-threshold 2 `
    --frame-cache output/qwen3vl_wd_planning196_thr2/qwen3vl_wd_planning196_thr2/frame_cache_16 `
    --output-dir "output/vlm_wd_consensus_paired_cv/fold_$_"
}
```

Repeat with `--target-mode soft` and a separate output directory for the
soft-label experiment.

## Combine and compare outer-fold predictions

```powershell
python scripts/combine_wd_cv_predictions.py `
  --fold-root output/vlm_wd_consensus_paired_cv `
  --output output/vlm_wd_consensus_paired_cv/oof_predictions.csv

python scripts/combine_wd_cv_predictions.py `
  --fold-root output/llm_wd_consensus_cv `
  --output output/llm_wd_consensus_cv/oof_predictions.csv

python scripts/compare_vlm_llm_wd_predictions.py `
  --vlm-predictions output/vlm_wd_consensus_paired_cv/oof_predictions.csv `
  --llm-predictions output/llm_wd_consensus_cv/oof_predictions.csv `
  --output output/paired_comparison_consensus
```
