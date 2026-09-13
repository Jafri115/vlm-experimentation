# WD_P VLM and LLM experiments: repaired common cohort

## Evaluation design

This report contains only experiments on the repaired common VLM/LLM cohort.
The target is patient withdrawal (`WD_P`) at the 3RS v2022 boundary `>=2` for
each human rater. All folds are patient-disjoint and identical across modalities.

- **VLM:** Qwen3-VL using 16 patient-only video frames per segment.
- **LLM:** Qwen3-8B Instruct using timestamped therapist/patient transcripts.
- **Primary classification threshold:** probability `>=0.5`.
- **Consensus target:** both raters negative or both positive; disagreements excluded.
- **Soft target:** both negative = `0`, disagreement = `0.5`, both positive = `1`.

## Cohort

| Cohort | Rows | Patients | Voxtral | Amberscript |
|---|---:|---:|---:|---:|
| Soft/all training cohort | 2,457 | 16 | 1,508 | 949 |
| Consensus evaluation subset | 1,734 | 16 | — | — |

Among the soft/all rows, 720 are consensus-negative, 723 have rater
disagreement, and 1,014 are consensus-positive. Fifty-five visual rows remain
outside the paired cohort: 14 lack transcript text, 38 have unresolved
utterance roles, one has a timing problem, and two have unresolved session-role
mappings.

Canonical cohort artifacts are under `output/wd_multimodal_master_repaired`.
Transferred overnight summaries are under
`output/wd_overnight_results_for_summary/output`.

## Experiment status

| Experiment | VLM | LLM | Paired comparison |
|---|---|---|---|
| Zero-shot WD_P | Not run on this cohort | Complete: 1,734/1,734 predictions | Pending VLM run |
| 3+3 few-shot WD_P | Not run on this cohort | Not run on this cohort | Not available |
| Continuous WD_P regression | Not run on this cohort | Complete: five folds | Pending VLM run |
| Consensus binary fine-tuning | Complete: five folds | Complete: five folds | **Complete** |
| Soft-label fine-tuning | Complete: five folds | Complete: five folds | **Complete** |
| Training/validation curves | Complete | Complete | Available in transferred results |

## LLM zero-shot

Fold confusion matrices were pooled over all outer test folds. All selected rows
were predicted successfully.

| N | Accuracy | Balanced accuracy | Precision | Recall | Specificity | F1 | Predicted positive rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,734 | 0.653 | 0.621 | 0.668 | 0.811 | 0.432 | 0.732 | 0.710 |

Counts: TP 822, TN 311, FP 409, FN 192.

Zero-shot is essentially equal to consensus fine-tuning in balanced accuracy
(0.6213 versus 0.6207) and slightly higher in F1 (0.7323 versus 0.7283). A
formal paired test requires the row-level zero-shot predictions.

## LLM continuous regression

Pooled MAE and RMSE weight every outer-fold test row equally. Because the
transferred bundle lacks row-level predictions, Spearman is the test-size-weighted
mean of the five fold correlations.

| Model | N | MAE | RMSE | Fold-weighted Spearman | True range | Predicted range | Prediction mean |
|---|---:|---:|---:|---:|---|---|---:|
| Qwen3-8B transcript | 2,457 | 0.568 | 0.710 | 0.225 | 1.0–4.5 | 1.00–2.24 | 1.641 |

The transcript model compresses the human score range substantially. A direct
modality comparison requires VLM regression on these same folds.

## Five-fold classification results

`Selected` means the VLM uses a separate validation-selected threshold in each
fold. Fixed 0.5 rows are the primary modality comparison.

| Training method | Modality | Threshold | N | Accuracy | Balanced accuracy | Precision | Recall | Specificity | F1 | AUPRC | AUROC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Consensus, automatic weight | VLM | Selected | 1,734 | 0.601 | 0.544 | 0.610 | 0.883 | 0.206 | 0.721 | 0.627 | 0.552 |
| Consensus, automatic weight | VLM | 0.5 | 1,734 | 0.483 | 0.504 | 0.589 | 0.384 | 0.624 | 0.465 | 0.627 | 0.552 |
| Consensus, `pos_weight=1` | VLM | 0.5 | 1,734 | 0.621 | 0.574 | 0.630 | **0.853** | 0.294 | 0.725 | 0.628 | 0.569 |
| Consensus | LLM | 0.5 | 1,734 | **0.651** | **0.621** | **0.669** | 0.800 | **0.442** | **0.728** | **0.640** | **0.630** |
| Soft labels | VLM | Selected | 1,734 | 0.617 | 0.549 | 0.611 | **0.953** | 0.144 | **0.744** | 0.646 | 0.596 |
| Soft labels | VLM | 0.5 | 1,734 | 0.601 | 0.585 | 0.653 | 0.680 | 0.490 | 0.666 | 0.646 | 0.596 |
| Soft labels | LLM | 0.5 | 1,734 | **0.636** | **0.615** | **0.672** | **0.736** | **0.494** | **0.702** | **0.654** | **0.640** |

## Primary paired VLM–LLM comparisons

### Consensus fine-tuning

The primary VLM uses one epoch, 16 frames, frame width 224, `mean_all` pooling,
and `pos_weight=1`.

| Metric | VLM | LLM | LLM minus VLM |
|---|---:|---:|---:|
| Accuracy | 0.621 | 0.651 | +0.030 |
| Balanced accuracy | 0.574 | 0.621 | +0.047 |
| Precision | 0.630 | 0.669 | +0.039 |
| Recall | **0.853** | 0.800 | -0.053 |
| Specificity | 0.294 | **0.442** | +0.147 |
| F1 | 0.725 | **0.728** | +0.004 |
| AUPRC | 0.628 | **0.640** | +0.012 |
| AUROC | 0.569 | **0.630** | +0.061 |

McNemar exact `p=0.0418`: 288 rows were correct only for VLM and 340 only for
LLM. The patient-cluster bootstrap 95% interval for the balanced-accuracy
difference is `[-0.031, 0.111]`.

### Soft-label fine-tuning

| Metric | VLM | LLM | LLM minus VLM |
|---|---:|---:|---:|
| Accuracy | 0.601 | 0.636 | +0.034 |
| Balanced accuracy | 0.585 | 0.615 | +0.030 |
| Precision | 0.653 | 0.672 | +0.019 |
| Recall | 0.680 | 0.736 | +0.055 |
| Specificity | 0.490 | 0.494 | +0.004 |
| F1 | 0.666 | 0.702 | +0.036 |
| AUPRC | 0.646 | 0.654 | +0.009 |
| AUROC | 0.596 | 0.640 | +0.045 |

McNemar exact `p=0.0384`: 363 rows were correct only for VLM and 422 only for
LLM. The patient-cluster bootstrap 95% interval for the balanced-accuracy
difference is `[-0.042, 0.096]`.

Both patient-cluster intervals include zero. The point estimates favor the LLM,
but the experiments do not establish a reliable patient-level advantage.

## Positive-weight diagnostic

Both rows below use the repaired cohort.

| VLM consensus setup | Balanced accuracy | Recall | Specificity |
|---|---:|---:|---:|
| Automatic `pos_weight`, threshold 0.5 | 0.504 | 0.384 | **0.624** |
| `pos_weight=1`, threshold 0.5 | **0.574** | **0.853** | 0.294 |

Changing to `pos_weight=1` raised balanced accuracy by 0.070 by greatly
increasing sensitivity. It also caused many more positive predictions and
reduced specificity.

## Findings

1. LLM zero-shot, consensus fine-tuning, and soft-label fine-tuning have similar
   balanced accuracy: 0.621, 0.621, and 0.615.
2. Consensus VLM with `pos_weight=1` has high recall but low specificity and
   predicts 79.2% of rows positive. The LLM has a more balanced error profile.
3. Consensus F1 is nearly identical: 0.725 for VLM and 0.728 for LLM.
4. Soft-label LLM point estimates exceed fixed-threshold VLM estimates for every
   reported metric.
5. Patient-bootstrap intervals include zero for both paired comparisons, so the
   observed LLM advantages are not conclusive at the patient level.
6. LLM regression still shows range compression and weak fold-weighted rank
   correlation, supporting the binary WD_P formulation.

## Remaining work on this cohort

- Run VLM zero-shot for a paired zero-shot modality comparison.
- Run VLM regression for a paired continuous comparison.
- Run 3+3 few-shot for both modalities only if that experiment is still needed.
- Transfer row-level regression predictions for a pooled Spearman correlation
  and patient-cluster confidence interval.

