# WD pairwise reliability

Run from the repository root on the machine containing the repaired-cohort predictions:

```powershell
python scripts/evaluate_wd_pairwise_reliability.py --bootstrap 2000
```

Requires numpy and pandas, no GPU or model loading. Input paths for all ten experiments are in `scripts/wd_pairwise_reliability_specs.json`. Missing files cause an error; edit that list explicitly if evaluating fewer experiments.

Outputs under `output/wd_pairwise_reliability`: `report.md`, `binary_reliability.csv`, `ordinal_reliability.csv`, `pairwise_reliability.csv`, and `coverage.csv`.

Each model has human1-human2, AI-human1, AI-human2, and mean AI agreement minus human agreement. Binary metrics are exact agreement and unweighted Gwet AC1. Ordinal metrics are quadratic AC2, absolute-agreement single-measure ICC(A,1), MAE and within-one-point agreement. Probability-only classifiers do not receive invented ordinal scores.

`available` uses each model's valid predictions. `common_models` uses identical successful segments across models within each experiment, separately for each scale. Start with common_models when comparing modalities. N is the evaluation count, not a training count. Confidence intervals resample whole patients 2,000 times, retaining segments within each sampled patient. They describe variability conditional on the saved fitted models, not repeated model training.

Consensus-only human binary agreement equals 1 by selection. It is not representative human reliability on all segments. Model coverage and the number of disagreement segments are reported. An interval for the AI-minus-human difference containing zero does not establish equivalence. Higher MAE is worse; higher agreement coefficients are better.

Human binary labels use rating >=2; classifiers use saved probability >=0.5; regression uses score >=2. Ordinal AC2 uses the fixed 1-5 scale and rounds predictions half upward. ICC and error metrics retain continuous regression values. No thresholds are optimized on test labels.

The ordinal AC2 expected-agreement formula follows [Gwet's reference irrCAC implementation](https://github.com/cran/irrCAC/blob/master/R/agree.coeff3.raw.r): sum(weights) times sum(p*(1-p)), divided by q*(q-1). Earlier ordinal AC2 outputs from the old third-rater script should be recalculated; that script used a different expected-agreement expression. Pairwise and three-rater coefficients also measure different comparisons.
