"""Compile completed expanded-cohort WD_P results from a transfer folder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def markdown(frame: pd.DataFrame, digits: int = 3) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---:" if pd.api.types.is_numeric_dtype(frame[c]) else "---" for c in columns) + " |",
    ]
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


def first_recursive(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found below {root}")
    return matches[0]


def binary_metrics(frame: pd.DataFrame, probability: str = "WD_probability") -> dict:
    data = frame.copy()
    data["target"] = pd.to_numeric(data["WD_consensus"], errors="coerce")
    data["probability"] = pd.to_numeric(data[probability], errors="coerce")
    eligible = data["target"].notna()
    eligible_n = int(eligible.sum())
    data = data[eligible & np.isfinite(data["probability"])].copy()
    y = data["target"].astype(int).to_numpy()
    p = data["probability"].to_numpy(float)
    prediction = (p >= 0.5).astype(int)
    return {
        "N_eligible": eligible_n,
        "N_evaluated": len(data),
        "missing_predictions": eligible_n - len(data),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "accuracy": accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "specificity": recall_score(y, prediction, pos_label=0, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "predicted_positive_rate": prediction.mean(),
    }


def regression_metrics(path: Path) -> dict:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    y = pd.to_numeric(frame["WD_P_mean"], errors="raise").to_numpy(float)
    p = pd.to_numeric(frame["WD_prediction"], errors="raise").to_numpy(float)
    rounded = np.clip(np.floor(p + 0.5), 1, 5).astype(int)
    kappas = []
    for column in ("WD_P_rater1", "WD_P_rater2"):
        human = pd.to_numeric(frame[column], errors="raise").astype(int)
        kappas.append(cohen_kappa_score(human, rounded, labels=[1, 2, 3, 4, 5], weights="quadratic"))
    pairwise_mae = np.mean(
        (np.abs(p - frame["WD_P_rater1"].to_numpy(float)) +
         np.abs(p - frame["WD_P_rater2"].to_numpy(float))) / 2
    )
    return {
        "N_evaluated": len(frame),
        "MAE": mean_absolute_error(y, p),
        "RMSE": mean_squared_error(y, p) ** 0.5,
        "Spearman": spearmanr(y, p).statistic,
        "prediction_min": p.min(),
        "prediction_max": p.max(),
        "prediction_SD": p.std(),
        "target_min": y.min(),
        "target_max": y.max(),
        "target_SD": y.std(),
        "AI_human_pairwise_MAE": pairwise_mae,
        "AI_human_mean_quadratic_kappa": np.nanmean(kappas),
    }


def vlm_regression_metrics(path: Path, label_reference_path: Path) -> dict:
    """Normalize the VLM regression schema and attach frozen human ratings."""
    prediction = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    reference = pd.read_csv(label_reference_path, encoding="utf-8-sig", low_memory=False)
    pred_id = "segment_uid" if "segment_uid" in prediction else "sample_id"
    ref_id = "segment_uid" if "segment_uid" in reference else "sample_id"
    reference = reference[[ref_id, "WD_P_rater1", "WD_P_rater2"]].copy()
    if ref_id != pred_id:
        reference = reference.rename(columns={ref_id: pred_id})
    frame = prediction.merge(reference, on=pred_id, how="left", validate="one_to_one")
    if frame[["WD_P_rater1", "WD_P_rater2"]].isna().any().any():
        raise ValueError("VLM regression rows could not all be matched to human ratings")
    frame["WD_P_mean"] = pd.to_numeric(frame["WD_P_true"], errors="raise")
    frame["WD_prediction"] = pd.to_numeric(frame["WD_P_pred"], errors="raise")
    normalized = path.parent / "oof_predictions_with_human_ratings.csv"
    frame.to_csv(normalized, index=False, encoding="utf-8-sig")
    return regression_metrics(normalized)


def main(root: Path) -> None:
    root = root.resolve()
    binary_rows: list[dict] = []
    for label, folder, training in (
        ("Qwen3-8B zero-shot", "llm_wd_zero_expanded_cv", "0"),
        ("Qwen3-8B 3+3 few-shot", "llm_wd_few_expanded_cv", "6 demonstrations/fold"),
        ("Qwen3-VL zero-shot", "vlm_wd_zero_expanded_cv", "0"),
        ("Qwen3-VL 3+3 few-shot", "vlm_wd_few_expanded_cv", "6 demonstrations/fold"),
        ("Qwen3-8B consensus fine-tuning", "llm_wd_consensus_expanded_cv", "consensus rows"),
        ("Qwen3-8B soft-label fine-tuning", "llm_wd_soft_expanded_cv", "all rows"),
    ):
        path = root / folder / "oof_predictions.csv"
        metrics = binary_metrics(pd.read_csv(path, encoding="utf-8-sig", low_memory=False))
        binary_rows.append({"model": label, "training": training, "threshold": "0.5", **metrics})

    for mode, label in (("consensus", "Qwen3-VL consensus fine-tuning"),
                        ("soft", "Qwen3-VL soft-label fine-tuning")):
        path = root / f"paired_comparison_{mode}_expanded" / "comparison_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8-sig"))
        metrics = summary["vlm_metrics"]
        binary_rows.append({
            "model": label,
            "training": "consensus rows" if mode == "consensus" else "all rows",
            "threshold": "validation-selected/fold",
            "N_eligible": summary["shared_consensus_rows"],
            "N_evaluated": metrics["N"],
            "missing_predictions": 0,
            "balanced_accuracy": metrics["balanced_accuracy"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "AUROC": metrics["auroc"],
            "AUPRC": metrics["auprc"],
            "predicted_positive_rate": metrics["predicted_positive_rate"],
        })

    binary = pd.DataFrame(binary_rows)
    binary.to_csv(root / "compiled_binary_results.csv", index=False, encoding="utf-8-sig")

    regression_rows: list[dict] = []
    regression_sources = (
        ("Qwen3-8B standard regression", "llm_wd_regression_expanded_cv"),
        ("Qwen3-8B ordinal", "llm_wd_ordinal_qwen3_8b_expanded_cv"),
        ("Qwen3-14B ordinal", "llm_wd_ordinal_qwen3_14b_expanded_cv"),
        ("Mistral Small 3.2 24B ordinal", "llm_wd_ordinal_mistral_small_24b_expanded_cv"),
    )
    for label, folder in regression_sources:
        path = first_recursive(root / folder, "oof_predictions.csv")
        regression_rows.append({"model": label, **regression_metrics(path)})

    vlm_path = first_recursive(root / "vlm_wd_regression_expanded_cv", "oof_predictions.csv")
    label_reference = first_recursive(root / "llm_wd_ordinal_qwen3_14b_expanded_cv", "oof_predictions.csv")
    regression_rows.append({
        "model": "Qwen3-VL standard regression",
        **vlm_regression_metrics(vlm_path, label_reference),
    })

    report_root = root / "wd_ordinal_regression_expanded_report"
    existing = pd.read_csv(report_root / "ordinal_regression_metrics.csv", encoding="utf-8-sig")
    baseline = existing.loc[existing["model"] == "Training mean"].iloc[0]
    regression_rows.insert(0, {
        "model": "Fold training-mean baseline",
        "N_evaluated": int(baseline["N_eval"]),
        "MAE": baseline["MAE"],
        "RMSE": baseline["RMSE"],
        "Spearman": baseline["Spearman"],
        "prediction_min": np.nan,
        "prediction_max": np.nan,
        "prediction_SD": baseline["prediction_SD"],
        "target_min": np.nan,
        "target_max": np.nan,
        "target_SD": baseline["target_SD"],
        "AI_human_pairwise_MAE": baseline["AI_human_pairwise_MAE"],
        "AI_human_mean_quadratic_kappa": baseline["AI_human_mean_quadratic_kappa"],
    })
    regression = pd.DataFrame(regression_rows)
    regression.to_csv(root / "compiled_regression_results.csv", index=False, encoding="utf-8-sig")

    human = pd.read_csv(report_root / "human_reference.csv", encoding="utf-8-sig").iloc[0]
    consensus_comparison = json.loads(
        (root / "paired_comparison_consensus_expanded" / "comparison_summary.json").read_text(encoding="utf-8-sig")
    )
    soft_comparison = json.loads(
        (root / "paired_comparison_soft_expanded" / "comparison_summary.json").read_text(encoding="utf-8-sig")
    )

    binary_display = binary[[
        "model", "N_evaluated", "balanced_accuracy", "AUROC", "AUPRC",
        "precision", "recall", "specificity", "f1", "predicted_positive_rate",
    ]].copy()
    regression_display = regression[[
        "model", "N_evaluated", "MAE", "RMSE", "Spearman", "prediction_min",
        "prediction_max", "prediction_SD", "AI_human_pairwise_MAE",
        "AI_human_mean_quadratic_kappa",
    ]].copy()

    lines = [
        "# Completed expanded-cohort WD_P results",
        "",
        "All model rows are out-of-fold predictions from the frozen 20-patient cohort. "
        "Binary evaluation uses the 3,026 rows where the two humans agree on WD_P absence "
        "versus presence. Regression evaluation uses all 4,325 rows and the mean of the two "
        "human 1-5 ratings.",
        "",
        "## Binary WD_P classification",
        "",
        markdown(binary_display),
        "",
        "The transcript models cluster around 0.61 balanced accuracy. Fine-tuning did not "
        "clearly improve over zero-shot on this cohort. The completed VLM prompt-based and "
        "fine-tuned results can now be compared on the same frozen folds.",
        "",
        "The paired LLM-minus-VLM balanced-accuracy difference was "
        f"{consensus_comparison['llm_minus_vlm']['balanced_accuracy']:.3f} for consensus training "
        f"(patient-bootstrap 95% CI {consensus_comparison['patient_cluster_bootstrap_balanced_accuracy_difference_95ci']}) "
        f"and {soft_comparison['llm_minus_vlm']['balanced_accuracy']:.3f} for soft-label training "
        f"(95% CI {soft_comparison['patient_cluster_bootstrap_balanced_accuracy_difference_95ci']}). "
        "Both intervals include zero, so this cohort does not establish a reliable modality difference.",
        "",
        "Few-shot inference has 29 missing predictions; all other completed binary rows have full coverage.",
        "",
        "## Continuous/ordinal WD_P severity",
        "",
        markdown(regression_display),
        "",
        "All learned models are evaluated on the same 4,325 out-of-fold segments. The standard "
        "8B regression has the lowest transcript MAE; the 14B ordinal model has the best transcript "
        "RMSE. The VLM row completes the direct continuous-severity comparison.",
        "",
        "Severity collapse remains: human mean ratings span approximately 1-4.5 in the OOF data, "
        "while learned predictions remain concentrated near the lower end. Increasing from 8B to "
        "14B therefore did not restore the human score range.",
        "",
        "## Comparison with human reliability",
        "",
        f"The two humans have exact 1-5 agreement {human['human_human_exact']:.3f}, within-one "
        f"agreement {human['human_human_within_one']:.3f}, MAE {human['human_human_MAE']:.3f}, "
        f"and quadratic kappa {human['human_human_quadratic_kappa']:.3f}.",
        "",
        "The best completed AI-human mean quadratic kappa is only about 0.14. This means the "
        "models carry some WD_P signal, but they are not close to reproducing human ordinal severity "
        "judgments. Balanced accuracy around 0.61 supports weak binary discrimination and should not "
        "be described as human-level agreement.",
        "",
        "## Completion status",
        "",
        "The expanded-cohort zero-shot, 3+3 few-shot, continuous regression, consensus fine-tuning, "
        "and soft-label fine-tuning results are now complete for both modalities.",
    ]
    (root / "compiled_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {root / 'compiled_results.md'}")
    print(f"Wrote {root / 'compiled_binary_results.csv'}")
    print(f"Wrote {root / 'compiled_regression_results.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
