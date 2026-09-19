"""Create a self-contained comparison of expanded-cohort Qwen3 ordinal WD models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_squared_error,
    roc_auc_score,
)


def resolve_oof(path: Path) -> Path:
    if path.is_file():
        return path
    direct = path / "oof_predictions.csv"
    if direct.exists():
        return direct
    matches = list(path.glob("*/oof_predictions.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one oof_predictions.csv under {path}; found {len(matches)}")
    return matches[0]


def md(frame: pd.DataFrame, digits: int = 3) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---:" if pd.api.types.is_numeric_dtype(frame[c]) else "---"
                                  for c in columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("NA")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.{digits}f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def icc_a1(a: np.ndarray, b: np.ndarray) -> float:
    values = np.column_stack([a, b]).astype(float)
    n, k = values.shape
    grand = values.mean()
    row_mean = values.mean(axis=1)
    col_mean = values.mean(axis=0)
    ms_row = k * np.square(row_mean - grand).sum() / (n - 1)
    ms_col = n * np.square(col_mean - grand).sum() / (k - 1)
    residual = values - row_mean[:, None] - col_mean[None, :] + grand
    ms_error = np.square(residual).sum() / ((n - 1) * (k - 1))
    return float((ms_row - ms_error) /
                 (ms_row + (k - 1) * ms_error + k * (ms_col - ms_error) / n))


def validate(a: pd.DataFrame, b: pd.DataFrame) -> str:
    key = "sample_id" if "sample_id" in a else "segment_uid"
    required = {key, "patient_id", "outer_fold", "WD_P_mean", "WD_P_rater1",
                "WD_P_rater2", "WD_prediction", "WD_score_probability_1"}
    for name, frame in [("8B", a), ("14B", b)]:
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"{name} contains missing or duplicate identifiers")
    if set(a[key].astype(str)) != set(b[key].astype(str)):
        raise ValueError("The two OOF files do not contain identical samples")
    return key


def binary_metrics(frame: pd.DataFrame) -> dict:
    consensus = (frame.WD_P_rater1.ge(2) == frame.WD_P_rater2.ge(2))
    subset = frame.loc[consensus]
    y = subset.WD_P_rater1.ge(2).astype(int).to_numpy()
    probability = 1 - subset.WD_score_probability_1.to_numpy(float)
    prediction = probability >= .5
    return {
        "N": len(subset),
        "Balanced accuracy": balanced_accuracy_score(y, prediction),
        "AUROC": roc_auc_score(y, probability),
        "AUPRC": average_precision_score(y, probability),
        "F1": f1_score(y, prediction),
    }


def model_metrics(frame: pd.DataFrame) -> dict:
    y = frame.WD_P_mean.to_numpy(float)
    p = frame.WD_prediction.to_numpy(float)
    patient = frame.patient_id.astype(str)
    yc = pd.Series(y) - pd.Series(y).groupby(patient).transform("mean")
    pc = pd.Series(p) - pd.Series(p).groupby(patient).transform("mean")
    rounded = np.clip(np.floor(p + .5), 1, 5).astype(int)
    qwk = np.mean([
        cohen_kappa_score(frame.WD_P_rater1, rounded, labels=[1, 2, 3, 4, 5], weights="quadratic"),
        cohen_kappa_score(frame.WD_P_rater2, rounded, labels=[1, 2, 3, 4, 5], weights="quadratic"),
    ])
    icc = np.mean([
        icc_a1(frame.WD_P_rater1.to_numpy(), p),
        icc_a1(frame.WD_P_rater2.to_numpy(), p),
    ])
    return {
        "N": len(frame), "MAE": np.mean(np.abs(p-y)),
        "RMSE": np.sqrt(mean_squared_error(y, p)),
        "Spearman": spearmanr(y, p).statistic,
        "Within-patient Spearman": spearmanr(yc, pc).statistic,
        "Prediction mean": p.mean(), "Prediction SD": p.std(ddof=0),
        "Target SD": y.std(ddof=0), "SD ratio": p.std(ddof=0)/y.std(ddof=0),
        "Within 0.5": np.mean(np.abs(p-y) <= .5),
        "Within 1.0": np.mean(np.abs(p-y) <= 1),
        "AI-human pairwise MAE": np.mean((np.abs(p-frame.WD_P_rater1) +
                                           np.abs(p-frame.WD_P_rater2))/2),
        "AI-human mean QWK": qwk, "AI-human mean ICC(A,1)": icc,
        "Prediction minimum": p.min(), "Prediction maximum": p.max(),
    }


def patient_bootstrap(a: pd.DataFrame, b: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    patients = a.patient_id.astype(str).unique()
    groups = {patient: np.flatnonzero(a.patient_id.astype(str).to_numpy() == patient)
              for patient in patients}
    y = a.WD_P_mean.to_numpy(float)
    pa = a.WD_prediction.to_numpy(float)
    pb = b.WD_prediction.to_numpy(float)
    rng = np.random.default_rng(seed)
    mae, mse = [], []
    for _ in range(draws):
        indices = np.concatenate([groups[x] for x in rng.choice(patients, len(patients), replace=True)])
        mae.append(np.mean(np.abs(pa[indices]-y[indices]) - np.abs(pb[indices]-y[indices])))
        mse.append(np.mean(np.square(pa[indices]-y[indices]) - np.square(pb[indices]-y[indices])))
    rows = []
    for metric, values in [("MAE reduction", mae), ("MSE reduction", mse)]:
        values = np.asarray(values)
        rows.append({"Metric": metric, "14B improvement": values.mean(),
                     "95% CI lower": np.quantile(values, .025),
                     "95% CI upper": np.quantile(values, .975),
                     "P(improvement <= 0)": np.mean(values <= 0)})
    return pd.DataFrame(rows)


def old_regression_summary(root: Path) -> dict:
    rows = []
    for path in sorted(root.glob("fold_*/final_summary.json")):
        metrics = json.loads(path.read_text(encoding="utf-8-sig"))["test_metrics"]
        rows.append({"N": int(metrics["N"]), "MAE": float(metrics["mae"]),
                     "RMSE": float(metrics["rmse"]), "Spearman": float(metrics["spearman"]),
                     "Prediction mean": float(metrics["prediction_mean"]),
                     "Prediction minimum": float(metrics["prediction_min"]),
                     "Prediction maximum": float(metrics["prediction_max"])})
    if len(rows) != 5:
        raise FileNotFoundError(f"Expected five old regression summaries under {root}; found {len(rows)}")
    frame = pd.DataFrame(rows)
    weights = frame.N.to_numpy()
    return {"N": int(weights.sum()), "MAE": np.average(frame.MAE, weights=weights),
            "RMSE": np.sqrt(np.average(np.square(frame.RMSE), weights=weights)),
            "Fold-weighted Spearman": np.average(frame.Spearman, weights=weights),
            "Prediction mean": np.average(frame["Prediction mean"], weights=weights),
            "Prediction minimum": frame["Prediction minimum"].min(),
            "Prediction maximum": frame["Prediction maximum"].max()}


def subset_metrics(frame: pd.DataFrame, mask: np.ndarray, label: str) -> dict:
    subset = frame.loc[mask]
    y = subset.WD_P_mean.to_numpy(float)
    prediction = subset.WD_prediction.to_numpy(float)
    return {"Subset": label, "N": len(subset),
            "Patients": subset.patient_id.astype(str).nunique(),
            "MAE": np.mean(np.abs(prediction-y)),
            "RMSE": np.sqrt(np.mean(np.square(prediction-y))),
            "Spearman": spearmanr(y, prediction).statistic,
            "Target mean": y.mean(), "Target SD": y.std(ddof=0),
            "Prediction mean": prediction.mean(), "Prediction SD": prediction.std(ddof=0)}


def main(args) -> None:
    path8, path14 = resolve_oof(args.qwen8), resolve_oof(args.qwen14)
    q8 = pd.read_csv(path8, encoding="utf-8-sig", low_memory=False)
    q14 = pd.read_csv(path14, encoding="utf-8-sig", low_memory=False)
    key = validate(q8, q14)
    q8 = q8.sort_values(key).reset_index(drop=True)
    q14 = q14.sort_values(key).reset_index(drop=True)
    for column in ["outer_fold", "WD_P_mean", "WD_P_rater1", "WD_P_rater2"]:
        if not np.allclose(q8[column], q14[column], equal_nan=True):
            raise ValueError(f"The models differ on {column}")

    metrics = pd.DataFrame([
        {"Model": "Qwen3-8B", **model_metrics(q8)},
        {"Model": "Qwen3-14B", **model_metrics(q14)},
    ])
    binary = pd.DataFrame([
        {"Model": "Qwen3-8B", **binary_metrics(q8)},
        {"Model": "Qwen3-14B", **binary_metrics(q14)},
    ])
    folds = []
    for fold in sorted(q8.outer_fold.unique()):
        for name, frame in [("Qwen3-8B", q8), ("Qwen3-14B", q14)]:
            subset = frame[frame.outer_fold == fold]
            result = model_metrics(subset)
            folds.append({"Fold": int(fold), "Model": name, "N": len(subset),
                          "MAE": result["MAE"], "RMSE": result["RMSE"],
                          "Spearman": result["Spearman"]})
    folds = pd.DataFrame(folds)

    severity = []
    for score in sorted(q8.WD_P_mean.unique()):
        mask = q8.WD_P_mean.eq(score)
        severity.append({"Human mean score": score, "N": int(mask.sum()),
                         "8B mean prediction": q8.loc[mask, "WD_prediction"].mean(),
                         "14B mean prediction": q14.loc[mask, "WD_prediction"].mean()})
    severity = pd.DataFrame(severity)

    exact = q8.WD_P_rater1.eq(q8.WD_P_rater2)
    exact_rows = []
    for score in sorted(q8.loc[exact, "WD_P_rater1"].unique()):
        mask = exact & q8.WD_P_rater1.eq(score)
        exact_rows.append({"Human rating": score, "N": int(mask.sum()),
                           "8B rounded exact": np.mean(np.floor(q8.loc[mask, "WD_prediction"]+.5)==score),
                           "14B rounded exact": np.mean(np.floor(q14.loc[mask, "WD_prediction"]+.5)==score),
                           "8B mean prediction": q8.loc[mask, "WD_prediction"].mean(),
                           "14B mean prediction": q14.loc[mask, "WD_prediction"].mean()})
    exact_table = pd.DataFrame(exact_rows)
    human = pd.DataFrame([{
        "N": len(q8), "Human-human MAE": np.mean(np.abs(q8.WD_P_rater1-q8.WD_P_rater2)),
        "Exact agreement": np.mean(exact),
        "Within one point": np.mean(np.abs(q8.WD_P_rater1-q8.WD_P_rater2) <= 1),
        "Quadratic weighted kappa": cohen_kappa_score(
            q8.WD_P_rater1, q8.WD_P_rater2, labels=[1, 2, 3, 4, 5], weights="quadratic"),
        "ICC(A,1)": icc_a1(q8.WD_P_rater1.to_numpy(), q8.WD_P_rater2.to_numpy()),
    }])
    uncertainty = patient_bootstrap(q8, q14, args.bootstrap, args.seed)

    old_comparison = overlap = cohort = None
    if args.old_regression_root.exists() and args.old_cohort.exists():
        old = pd.read_csv(args.old_cohort, encoding="utf-8-sig", low_memory=False)
        old_metrics = old_regression_summary(args.old_regression_root)
        new_fold_metrics = []
        for _, subset in q8.groupby("outer_fold"):
            new_fold_metrics.append((len(subset), spearmanr(
                subset.WD_P_mean, subset.WD_prediction).statistic))
        old_comparison = pd.DataFrame([
            {"Cohort/run": "Previous repaired cohort",
             "Patients": old.patient_id.astype(str).nunique(),
             "Objective": "Direct regression", **old_metrics},
            {"Cohort/run": "Expanded cohort",
             "Patients": q8.patient_id.astype(str).nunique(),
             "Objective": "Soft ordinal cross-entropy", "N": len(q8),
             "MAE": metrics.loc[metrics.Model.eq("Qwen3-8B"), "MAE"].iloc[0],
             "RMSE": metrics.loc[metrics.Model.eq("Qwen3-8B"), "RMSE"].iloc[0],
             "Fold-weighted Spearman": np.average(
                 [x[1] for x in new_fold_metrics], weights=[x[0] for x in new_fold_metrics]),
             "Prediction mean": q8.WD_prediction.mean(),
             "Prediction minimum": q8.WD_prediction.min(),
             "Prediction maximum": q8.WD_prediction.max()},
        ])
        old_ids = set(old.sample_id.astype(str))
        in_old = q8.sample_id.astype(str).isin(old_ids).to_numpy()
        overlap = pd.DataFrame([
            subset_metrics(q8, in_old, "Segments overlapping previous cohort"),
            subset_metrics(q8, ~in_old, "Newly added segments"),
            subset_metrics(q8, np.ones(len(q8), dtype=bool), "All expanded segments"),
        ])
        cohort = pd.DataFrame([
            {"Cohort": "Previous repaired", "Rows": len(old),
             "Patients": old.patient_id.astype(str).nunique(),
             "Amberscript": int(old.transcript_provider.eq("amberscript").sum()),
             "Voxtral": int(old.transcript_provider.eq("voxtral").sum()),
             "Target mean": old.WD_P_mean.mean(), "Target SD": old.WD_P_mean.std(ddof=0),
             "Human exact agreement": np.mean(old.WD_P_rater1.eq(old.WD_P_rater2))},
            {"Cohort": "Expanded", "Rows": len(q8),
             "Patients": q8.patient_id.astype(str).nunique(),
             "Amberscript": int(q8.transcript_provider.eq("amberscript").sum()),
             "Voxtral": int(q8.transcript_provider.eq("voxtral").sum()),
             "Target mean": q8.WD_P_mean.mean(), "Target SD": q8.WD_P_mean.std(ddof=0),
             "Human exact agreement": np.mean(q8.WD_P_rater1.eq(q8.WD_P_rater2))},
        ])

    args.output.mkdir(parents=True, exist_ok=True)
    for filename, frame in [
        ("overall_metrics.csv", metrics), ("fold_metrics.csv", folds),
        ("binary_consensus_metrics.csv", binary), ("severity_calibration.csv", severity),
        ("human_exact_agreement.csv", exact_table), ("human_reference.csv", human),
        ("qwen14_bootstrap_improvement.csv", uncertainty),
    ]:
        frame.to_csv(args.output / filename, index=False, encoding="utf-8-sig")
    if old_comparison is not None:
        old_comparison.to_csv(args.output / "qwen8_cohort_expansion_comparison.csv", index=False,
                              encoding="utf-8-sig")
        overlap.to_csv(args.output / "expanded_old_overlap_diagnostic.csv", index=False,
                       encoding="utf-8-sig")
        cohort.to_csv(args.output / "cohort_composition_comparison.csv", index=False,
                      encoding="utf-8-sig")

    lines = [
        "# Expanded-cohort Qwen3 ordinal WD_P comparison", "",
        "This report compares Qwen3-8B and Qwen3-14B on the same 4,325 out-of-fold segments, "
        "20 patients, and five patient-disjoint folds. Both models predict a probability distribution "
        "over ratings 1-5; the expected score is evaluated against the mean human rating.", "",
        "## Main regression results", "",
        md(metrics[["Model", "N", "MAE", "RMSE", "Spearman", "Within-patient Spearman",
                    "Prediction SD", "Target SD", "SD ratio", "Within 0.5", "Within 1.0"]]), "",
        "Qwen3-14B has the best overall result. Its main gains are better ranking and fewer large errors. "
        "The MAE difference is small, and both models explain little of the full severity variation.", "",
        "## Fold stability", "", md(folds), "",
        "The 14B model has lower MAE and higher Spearman correlation in four of five folds.", "",
        "## Statistical uncertainty for the 14B improvement", "", md(uncertainty, 4), "",
        "Positive values favor Qwen3-14B. Intervals resample the 20 patients as clusters. The MAE "
        "interval crosses zero, while the MSE interval indicates fewer large errors for 14B.", "",
        "## Severity-range calibration", "", md(severity), "",
        "Human scores span 1-4.5, but both model means remain close to 1.6-1.9. This range compression "
        "is the main limitation: the models detect some ordering signal but do not reproduce severe ratings.", "",
        "## Comparison with human raters", "", md(human), "",
        md(metrics[["Model", "AI-human pairwise MAE", "AI-human mean QWK", "AI-human mean ICC(A,1)"]]), "",
        "AI-human reliability remains substantially below human-human reliability. Comparing model MAE "
        "against the mean of two humans alone is optimistic because the averaged target is less noisy.", "",
        "### Where both humans gave exactly the same rating", "", md(exact_table), "",
        "The models frequently reproduce rating 2, but neither produces an expected score high enough to "
        "round to 3 or 4. This is evidence of prediction toward the average rather than full severity learning.", "",
        "## Derived binary withdrawal detection", "", md(binary), "",
        "For consensus score-1 versus score-at-least-2 detection, Qwen3-14B improves balanced accuracy "
        "from 0.588 to 0.609. Binary detection is currently more defensible than exact severity estimation.", "",
        "## Presentation conclusion", "",
        "> Qwen3-14B consistently improved ranking, RMSE, binary withdrawal detection, and agreement with "
        "human ratings relative to Qwen3-8B. However, both models compressed predictions toward the "
        "average and failed to represent high-severity withdrawal. Transcripts contain useful signal for "
        "detecting withdrawal, while reliable 1-5 severity estimation remains unresolved.", "",
        "## Recommended next experiment", "",
        "Use Qwen3-14B with severity-balanced sampling or loss weighting, report macro error by rating, and "
        "select the checkpoint using a validation metric that rewards both rank correlation and prediction "
        "spread. Keep the present patient-disjoint folds unchanged for a fair comparison.", "",
        f"Sources: `{path8}` and `{path14}`.",
    ]
    if old_comparison is not None:
        expansion_lines = [
            "## Did cohort expansion improve Qwen3-8B?", "", md(cohort), "",
            md(old_comparison), "",
            "Observed error decreased: MAE changed from 0.568 to 0.557 and RMSE from 0.710 to 0.690. "
            "However, fold-weighted Spearman changed from 0.225 to 0.204, so ranking did not improve.", "",
            "This is not a controlled cohort-size experiment. In addition to expanding from 2,457 to "
            "4,325 segments and from 16 to 20 patients, the new run changed direct regression to soft "
            "ordinal cross-entropy, used three rather than two epochs, reduced the learning rate, enabled "
            "patient-balanced training, and used a revised manual-based rubric. The patient folds also changed.", "",
            "### Expanded-model performance on old-overlap and newly added segments", "", md(overlap), "",
            "The newly added segments are easier for the expanded model than the old-overlap segments. "
            "On the 2,426 overlapping segments, expanded-model MAE is 0.569 and RMSE is 0.709, essentially "
            "the same error as the previous run's aggregate result. The lower overall error is therefore "
            "partly a cohort-composition effect and does not demonstrate that additional training data improved "
            "the model. A causal test requires rerunning the old direct-regression configuration on the "
            "expanded folds, or rerunning the new ordinal configuration on the previous folds.", "",
            "### Presentation wording", "",
            "> Expanding the cohort increased coverage from 16 to 20 patients and reduced aggregate MAE "
            "slightly, but ranking performance did not improve. Because the objective, prompt, training "
            "schedule, and folds also changed, the current results do not isolate a benefit from cohort "
            "expansion. On segments overlapping the old cohort, prediction error remained essentially unchanged.", "",
        ]
        insert_at = lines.index("## Presentation conclusion")
        lines[insert_at:insert_at] = expansion_lines
    (args.output / "ordinal_model_comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(2); width = .36
    axes[0].bar(x-width/2, metrics.MAE, width, label="MAE")
    axes[0].bar(x+width/2, metrics.RMSE, width, label="RMSE")
    axes[0].set_xticks(x, metrics.Model); axes[0].set_ylim(0, .8)
    axes[0].set_title("Ordinal regression error"); axes[0].legend(); axes[0].grid(axis="y", alpha=.2)
    axes[1].plot(severity["Human mean score"], severity["8B mean prediction"], "o-", label="Qwen3-8B")
    axes[1].plot(severity["Human mean score"], severity["14B mean prediction"], "o-", label="Qwen3-14B")
    axes[1].plot([1, 5], [1, 5], "--", color="gray", label="Ideal")
    axes[1].set(xlabel="Mean human rating", ylabel="Mean model prediction", xlim=(.8, 5.1), ylim=(.8, 5.1),
                title="Prediction-range compression")
    axes[1].legend(); axes[1].grid(alpha=.2)
    fig.suptitle("Expanded 20-patient cohort: Qwen3 ordinal WD_P")
    fig.tight_layout(); fig.savefig(args.output / "ordinal_model_comparison.png", dpi=180); plt.close(fig)
    print(f"Report: {(args.output/'ordinal_model_comparison.md').resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen8", type=Path,
                        default=Path("output/llm_wd_ordinal_qwen3_8b_expanded_cv"))
    parser.add_argument("--qwen14", type=Path,
                        default=Path("output/llm_wd_ordinal_qwen3_14b_expanded_cv"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/wd_ordinal_regression_expanded_report"))
    parser.add_argument("--old-regression-root", type=Path,
                        default=Path("output/llm_wd_regression_repaired_cv"))
    parser.add_argument("--old-cohort", type=Path,
                        default=Path("output/wd_multimodal_master_repaired/paired_master_soft.csv"))
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
