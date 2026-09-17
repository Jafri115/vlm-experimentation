# Is the WD model learning useful information?

This evaluation asks whether saved out-of-fold predictions contain useful held-out information beyond rating frequency and average severity. It does not retrain models. Beating a baseline supports predictive value; it does not by itself prove which cues the model used, that fine-tuning caused the gain, or that the model can replace a human.

## Run on the experiment machine

```powershell
python scripts/evaluate_wd_learning.py --bootstrap 2000
```

Dependencies: numpy, pandas, scikit-learn, matplotlib. No GPU required. The default input paths are in `scripts/wd_pairwise_reliability_specs.json`; the master is `output/wd_multimodal_master_repaired`. Do not use old-cohort predictions. Missing OOF prediction files cause a clear error. Use `--no-plots` if plotting is not needed.

Outputs go to `output/wd_learning_evaluation`:

| File | Question answered |
|---|---|
| report.md | Does AI reduce error compared with training-only baselines? |
| human_pair_prediction_distributions.csv | For every human pair (1,1), (2,2), (1,3), etc., how many AI predictions fall in each category? What are their mean, SD and quantiles? |
| numbered PNG plots | Do predictions follow different human scores, or collapse near one score? |
| metrics.csv | How do accuracy, score errors, ranking, spread and agreement change by fold, patient and human-agreement group? |
| baseline_gains.csv | How large is error reduction over each baseline, with a paired patient-bootstrap interval? |
| training_baselines.csv | Which training counts and constants produced each baseline? |
| calibration.csv | Do predicted probabilities correspond to the human positive fraction? |
| row_level_audit.csv | Each segment's two human ratings, AI prediction, fold and baseline; no transcript or audio text is exported. |
| coverage.csv | Which predictions were usable and which were excluded? |

## 1. Inspect exactly what the AI predicts

Start with the heatmap for each experiment. Each row is one shared human score, and each column is an AI category. Every row shows its N. A strong exact-score model places most mass on the diagonal. A vertical stripe means it predicts the same category across different human scores. Missing human categories are N/A, not zero performance.

The ordinal table uses half-up rounding for AI category assignment. Scatterplots, ICC, MAE, RMSE and score spread retain continuous predictions. Classifiers producing only probabilities cannot be interpreted as 1-5 severity raters. Their heatmaps have two columns: no withdrawal and withdrawal. A prompted model's explicit score and probability are evaluated separately; they need not be internally consistent.

Inspect all exact-agreement scores, not only 3. The available cohort has 720 pairs (1,1), 356 pairs (2,2), 178 pairs (3,3), six pairs (4,4), and no pairs (5,5). Thus always predicting 1 matches 720/1260 = 57.1% of exact-agreement rows, without using video or transcript. The six score-4 cases cannot support a stable score-4 conclusion. No score-5 conclusion is possible.

## 2. Compare with predictors that see no input

For every outer fold, constants are estimated from its training split only. Validation and test labels never set baselines or thresholds. Manifests are checked for patient disjointness and exact agreement with the frozen master labels. Baseline training counts are exported.

- Training mean: predicts the average training human-mean score for every held-out segment.
- Training median: a constant baseline suited to absolute error.
- Training mode: predicts the most common pooled individual human rating; ties use the lower score.
- Always 1 and always 2: transparent diagnostic constants, reported without choosing between them on test data.
- Random training rating: analytic expected MAE/MSE from independently drawing a 1-5 rating from the pooled training-human frequency distribution. This is an expectation, not a single lucky random run.
- Binary training prevalence: predicts the training positive fraction for every segment; the training-majority baseline outputs the corresponding fixed label.

Consensus fine-tuning and prompting use consensus training rows for these comparator baselines; regression and soft-label experiments use all training rows. For prompting, this is a reference baseline using the available training cohort, not a claim that the base model was trained on that cohort. For fine-tuning, verify the actual run used the full supplied manifest without extra filtering before asserting identical training samples.

Error reduction = baseline error minus AI error. Positive is better for AI. An interval entirely above zero supports improvement over that specific baseline on this cohort; crossing zero is inconclusive. Beating random ratings but failing to beat the mean is weak evidence of useful severity prediction. Training-mean improvement is different from standard R-squared, whose denominator uses held-out target variation.

## 3. Does the AI follow severity changes?

Read MAE/RMSE, Spearman and prediction SD together. Predictions narrowly clustered around 1.7 may have tolerable error if many human scores are low, but cannot track the range. `sd_ratio` compares AI spread with the human-mean spread; it is a diagnostic, not a metric to maximize. Excessive spread is also undesirable. The within-patient-centered Spearman removes each patient's mean before ranking; use it with individual-patient results to distinguish within-patient variation from differences between patients. Constants have undefined correlations, reported as NaN.

On exact-agreement rows, also inspect `exact_agreement_macro_score_recall`: it averages the match rate across observed shared scores, preventing the many score-1 rows from dominating. Its score-4 component is based on only six cases, so always show support counts.

## 4. Keep agreement and disagreement analyses separate

Report all rows, exactly equal human scores, and different human scores. For disagreement there is no single agreed answer. Error to the human mean is a declared modeling target, not indisputable truth. Individual-human agreement is also reported. Matching either of two ratings is an easier criterion than matching one shared rating. Do not compare those percentages as if the success rule were identical.

Binary AUROC, average precision, balanced accuracy, sensitivity and specificity use binary-consensus rows only. Brier error on all rows uses the fraction of the two humans calling withdrawal positive. That fraction describes these raters, not a known true probability. If humans disagree on binary labels, matching either human is guaranteed and provides no evidence of learning.

## 5. AC2 and ICC are supporting evidence

AC2 uses partial credit for nearby categories. It may be high for an input-independent predictor on a concentrated rating scale. That does not make its formula wrong; it makes an interpretation of high AC2 as proof of learning wrong. The new audit computes the same coefficient for constant baselines, exposing this directly.

Absolute-agreement, single-measure ICC asks a different question and depends on the variation in the evaluation sample. Low ICC is relevant evidence against interchangeability, but is not a universal verdict on whether the model has any signal. Do not switch to average-measure or consistency ICC just to increase the result. Pairwise AI-human coefficients and human-human coefficients use identical rows; ICC to a two-human average is not substituted for single-human agreement.

## 6. Statistical limits and subsequent experiments

Confidence intervals resample whole patients and compare model and baseline errors on the same sampled rows. They retain within-patient dependence, but condition on the fitted models and do not include repeated-training or training-set uncertainty. These exploratory intervals are not adjusted for testing many experiments. Model comparisons use common successful IDs within an experiment; cohorts still differ across experiments. Per-patient and per-fold tables reveal whether a pooled gain is driven by a few patients.

To attribute gains specifically to fine-tuning, compare the base model and fine-tuned model on identical held-out IDs, target definitions and inference procedures, then repeat training across seeds. To investigate use of the actual content, a separately planned input-ablation experiment (blank input or mismatched transcript/video, preserving the rest of the inference procedure) is useful. These require new inference/training and are not run by this audit. Do not claim a formal random-chance test or causal proof of learning from these summary metrics alone.

## Supervisor explanation

“We checked the full distribution of AI ratings for each human-rating pair. We then tested whether held-out predictions beat an average-rating and frequency-only predictor fitted on training data, and whether the AI tracked severity differences across segments and patients. Agreement coefficients were reported alongside these diagnostics, not used as standalone evidence of learning.”

## References

- [Scikit-learn dummy baselines](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.dummy).
- [Koo and Li: choosing and reporting ICC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4913118/).
- [Gwet reference AC1/AC2 implementation](https://github.com/cran/irrCAC/blob/master/R/agree.coeff3.raw.r).
