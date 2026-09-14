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
| Zero-shot WD_P | Complete: 1,734/1,734 | Complete: 1,734/1,734 | Aggregate comparison complete; paired test needs OOF rows |
| 3+3 few-shot WD_P | Complete: 1,734/1,734 | Complete: 1,720/1,734 | Aggregate comparison complete on unequal successful rows |
| Continuous WD_P regression | Complete: five folds | Complete: five folds | Aggregate comparison complete; paired test needs OOF rows |
| Consensus binary fine-tuning | Complete: five folds | Complete: five folds | **Complete** |
| Soft-label fine-tuning | Complete: five folds | Complete: five folds | **Complete** |
| Training/validation curves | Complete | Complete | Available in transferred results |

## Headline comparison

### Continuous regression

| Model | Input | Training N per fold | OOF evaluation N | MAE | RMSE | Fold-weighted Spearman |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-VL | Patient-only video | 1,259–1,656 | 2,457 | 0.583 | **0.695** | 0.185 |
| Qwen3-8B | Transcript | 1,259–1,656 | 2,457 | **0.568** | 0.710 | **0.225** |

### Binary WD_P classification

| Experiment | Model | Input | Training N per fold | OOF evaluation N | Balanced accuracy |
|---|---|---|---:|---:|---:|
| Zero-shot | Qwen3-VL | Patient-only video | 0 | 1,734 | 0.498 |
| Zero-shot | Qwen3-8B | Transcript | 0 | 1,734 | **0.621** |
| 3+3 few-shot | Qwen3-VL | Patient-only video | 6 demonstrations | 1,734 | 0.417 |
| 3+3 few-shot | Qwen3-8B | Transcript | 6 demonstrations | 1,720 | **0.596** |
| Consensus fine-tuning | Qwen3-VL | Patient-only video | 857–1,201 | 1,734 | 0.574 |
| Consensus fine-tuning | Qwen3-8B | Transcript | 857–1,201 | 1,734 | **0.621** |
| Soft-label fine-tuning | Qwen3-VL | Patient-only video | 1,259–1,656 | 1,734 | 0.585 |
| Soft-label fine-tuning | Qwen3-8B | Transcript | 1,259–1,656 | 1,734 | **0.615** |

## Zero-shot and 3+3 few-shot

Fold confusion matrices were pooled over all outer test folds. Both zero-shot
runs and VLM few-shot predicted every selected row. LLM few-shot had 14 errors.

| Prompt | Modality | Successful/selected | Accuracy | Balanced accuracy | Precision | Recall | Specificity | F1 | Predicted positive rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | VLM | 1,734/1,734 | 0.414 | 0.498 | 0.250 | 0.001 | 0.996 | 0.002 | 0.002 |
| Zero-shot | LLM | 1,734/1,734 | **0.653** | **0.621** | **0.668** | **0.811** | 0.432 | **0.732** | 0.710 |
| 3+3 few-shot | VLM | 1,734/1,734 | 0.430 | 0.417 | 0.513 | 0.493 | 0.340 | 0.503 | 0.562 |
| 3+3 few-shot | LLM | 1,720/1,734 | **0.601** | **0.596** | **0.673** | **0.625** | **0.567** | **0.648** | 0.546 |

Zero-shot counts are VLM TP 1, TN 717, FP 3, FN 1,013 and LLM TP
822, TN 311, FP 409, FN 192. The VLM zero-shot prompt almost always predicted
negative. Demonstrations changed its output distribution but reduced balanced
accuracy. LLM few-shot was also worse than LLM zero-shot.

LLM zero-shot is essentially equal to LLM consensus fine-tuning in balanced
accuracy (0.6213 versus 0.6207) and slightly higher in F1 (0.7323 versus
0.7283). Formal paired prompt comparisons require row-level OOF predictions.

## Continuous regression

Pooled MAE and RMSE weight every outer-fold test row equally. Because the
transferred bundle lacks row-level predictions, Spearman is the test-size-weighted
mean of the five fold correlations.

| Modality | N | MAE | RMSE | Fold-weighted Spearman | True range | Predicted range | Prediction mean |
|---|---:|---:|---:|---:|---|---|---:|
| Qwen3-VL video | 2,457 | 0.583 | **0.695** | 0.185 | 1.0–4.5 | 1.24–2.26 | 1.775 |
| Qwen3-8B transcript | 2,457 | **0.568** | 0.710 | **0.225** | 1.0–4.5 | 1.00–2.24 | 1.641 |

Both modalities compress the human score range substantially. LLM has lower
MAE and higher fold-weighted Spearman, while VLM has lower RMSE. Row-level
predictions are needed for paired uncertainty estimates and one pooled
Spearman correlation.

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

## AI treated as a third human rater

Each analysis places human rater 1, human rater 2, and the AI output in a
three-rater matrix. Ordinal AC2 uses quadratic weights on rounded 1–5 scores.
`ICC(A,1)` is a two-way random-effects, absolute-agreement, single-rating ICC.
For binary outputs, each human rating is converted with `WD_P >=2` and the AI
probability is converted at 0.5. Binary AC2 uses the two-category agreement
weights.

### Ordinal 1–5 reliability

Ordinal coefficients are available only when the model produced a 1–5 score.

| Experiment | Model | N | Gwet AC2, quadratic | ICC(A,1), absolute agreement |
|---|---|---:|---:|---:|
| Zero-shot | Qwen3-VL | 1,734 | 0.796 | 0.253 |
| Zero-shot | Qwen3-8B | 1,734 | **0.837** | **0.420** |
| 3+3 few-shot | Qwen3-VL | 1,734 | 0.698 | 0.111 |
| 3+3 few-shot | Qwen3-8B | 1,720 | **0.786** | **0.334** |
| Regression | Qwen3-VL | 2,457 | 0.836 | 0.249 |
| Regression | Qwen3-8B | 2,457 | **0.843** | **0.259** |

### Binary reliability

| Experiment | Model | N | Binary Gwet AC2 | Binary ICC(A,1) |
|---|---|---:|---:|---:|
| Zero-shot | Qwen3-VL | 1,734 | 0.254 | 0.292 |
| Zero-shot | Qwen3-8B | 1,734 | **0.566** | **0.510** |
| 3+3 few-shot | Qwen3-VL | 1,734 | 0.257 | 0.221 |
| 3+3 few-shot | Qwen3-8B | 1,720 | **0.480** | **0.457** |
| Regression | Qwen3-VL | 2,457 | **0.123** | **0.138** |
| Regression | Qwen3-8B | 2,457 | 0.107 | 0.120 |
| Consensus fine-tuning | Qwen3-VL | 2,457 | 0.303 | 0.211 |
| Consensus fine-tuning | Qwen3-8B | 1,734 | **0.561** | **0.508** |
| Soft-label fine-tuning | Qwen3-VL | 2,457 | 0.250 | 0.215 |
| Soft-label fine-tuning | Qwen3-8B | 2,457 | **0.288** | **0.243** |

The reliability populations differ because zero-shot and few-shot inference
were restricted to consensus rows, regression and soft-label prediction cover
the full 2,457 rows, and the saved consensus predictions do not cover the same
population for both modalities. Therefore, compare VLM and LLM coefficients
directly only when their `N` and included segment IDs match.

The ordinal AC2 values are much higher than the ICC values because quadratic
AC2 gives substantial credit to ratings that are only one category apart and
is robust to skewed category prevalence. For example, VLM zero-shot almost
always predicts score 1. That can remain close to many low human ratings and
produce AC2 0.796, while its ICC of 0.253 and balanced accuracy of 0.498 reveal
poor discrimination and weak absolute score agreement. Reliability coefficients
must therefore be read alongside balanced accuracy, score spread, sensitivity,
and specificity.

### Human-rater reference

| Population | N | Human–human ordinal AC2 | Human–human ordinal ICC(A,1) | Human–human binary AC2 | Human–human binary ICC(A,1) | Raw binary agreement |
|---|---:|---:|---:|---:|---:|---:|
| Full repaired cohort | 2,457 | 0.842 | 0.448 | 0.420 | 0.403 | 0.706 |
| Consensus-only subset | 1,734 | 0.931 | 0.785 | 1.000 | 1.000 | 1.000 |

The consensus-only human reliability of 1.0 is guaranteed by construction:
these are precisely the rows where both humans fall on the same side of the
binary threshold. It is not an independent estimate of ordinary human
reliability. The full-cohort row is the meaningful human benchmark. On all
2,457 rows, the humans disagree on the binary label in 723 cases (29.4%), so
even human–human binary AC2 is only 0.420.

### Simple interpretation

Binary conversion makes a one-point difference decisive. Human scores 1 and 2
are close on the five-point scale, but after thresholding they become opposite
labels. This removes the partial credit used by quadratic ordinal AC2 and is
the main reason binary AC2 is lower.

The regression models add a second problem: their predicted scores are tightly
compressed around roughly 1–2. Many rows consequently fall on the same side of
the binary cutoff, producing binary AC2 near 0.1 even though ordinal AC2 remains
high because the predicted scores are numerically close to common human scores.

For direct interpretation:

- Human–human reliability on the full cohort is moderate: binary AC2 0.420 and
  ICC 0.403.
- Soft-label VLM as a third rater is below that benchmark: AC2 0.250 and ICC 0.215.
- Soft-label LLM is also below it but slightly closer: AC2 0.288 and ICC 0.243.
- Regression AI outputs should be judged primarily with MAE, RMSE, Spearman,
  and ordinal ICC. Their binary reliability is distorted by range compression.
- Zero-shot and consensus results use the preselected consensus subset, where
  the two humans already agree perfectly on the binary target. Their AI
  reliability values therefore answer a narrower and easier question and
  should not be compared directly with the full-cohort human benchmark.

## Findings

1. LLM zero-shot is the strongest prompt result (balanced accuracy 0.621); VLM
   zero-shot collapses almost entirely to the negative class (0.498).
2. Adding 3+3 demonstrations does not improve either modality: VLM falls to
   0.417 and LLM reaches 0.596 balanced accuracy.
3. LLM zero-shot, consensus fine-tuning, and soft-label fine-tuning have similar
   balanced accuracy: 0.621, 0.621, and 0.615.
4. Consensus VLM with `pos_weight=1` has high recall but low specificity and
   predicts 79.2% of rows positive. The LLM has a more balanced error profile.
5. Consensus F1 is nearly identical: 0.725 for VLM and 0.728 for LLM.
6. Soft-label LLM point estimates exceed fixed-threshold VLM estimates for every
   reported metric.
7. Patient-bootstrap intervals include zero for both fine-tuning comparisons, so the
   observed LLM advantages are not conclusive at the patient level.
8. Both regression models compress the score range. LLM has slightly lower MAE
   and higher fold-weighted Spearman; VLM has slightly lower RMSE.
9. Treating AI as a third rater favors the LLM for zero-shot and few-shot on
   both ordinal and binary reliability. Regression ordinal reliability is very
   similar and low by ICC despite high quadratic AC2.

## Remaining work on this cohort

- Transfer the zero-shot, few-shot, and regression OOF prediction CSVs for
  exact paired tests, pooled regression correlation, and patient-cluster
  confidence intervals.
- Investigate the 14 failed LLM few-shot rows before treating the prompt
  comparison as fully paired.
