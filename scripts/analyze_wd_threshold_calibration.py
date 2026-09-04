#!/usr/bin/env python
"""
Post-hoc threshold / calibration analysis for Qwen3-VL WD_P patient-grouped CV.

For each fold, this script:
  1. Reads fold_N/val_predictions.csv and fold_N/test_predictions.csv.
  2. Restricts threshold selection to consensus-only validation rows.
  3. Selects thresholds using validation data ONLY:
       - max F1
       - max balanced accuracy
       - max Youden J
       - fixed 0.50
  4. Freezes each threshold and applies it to consensus-only TEST rows.
  5. Aggregates fold and pooled test metrics.

No model inference or retraining is performed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
)


def safe_auc(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def safe_auprc(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))


def confusion_metrics(y, p, threshold: float) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )

    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tn = int(np.sum((y == 0) & (pred == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    balanced_accuracy = (
        (sensitivity + specificity) / 2.0
        if np.isfinite(sensitivity) and np.isfinite(specificity)
        else np.nan
    )
    youden_j = (
        sensitivity + specificity - 1.0
        if np.isfinite(sensitivity) and np.isfinite(specificity)
        else np.nan
    )

    auprc = safe_auprc(y, p)
    prevalence = float(np.mean(y)) if len(y) else np.nan

    return {
        "n": int(len(y)),
        "positive_n": int(np.sum(y)),
        "negative_n": int(len(y) - np.sum(y)),
        "prevalence": prevalence,
        "threshold": float(threshold),
        "AUROC": safe_auc(y, p),
        "AUPRC": auprc,
        "AUPRC_minus_prevalence": auprc - prevalence if np.isfinite(auprc) else np.nan,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "youden_j": float(youden_j),
        "f1": float(f1),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def candidate_thresholds(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=float)
    uniq = np.unique(np.clip(prob[np.isfinite(prob)], 0.0, 1.0))
    if len(uniq) == 0:
        return np.array([0.5], dtype=float)

    mids = (uniq[:-1] + uniq[1:]) / 2.0 if len(uniq) > 1 else np.array([])
    values = np.concatenate([
        np.array([0.0, 0.5, 1.0], dtype=float),
        uniq,
        mids,
    ])
    return np.unique(np.clip(values, 0.0, 1.0))


def choose_best_threshold(y: np.ndarray, p: np.ndarray, objective: str) -> Tuple[float, pd.DataFrame]:
    rows = []
    for t in candidate_thresholds(p):
        m = confusion_metrics(y, p, float(t))
        rows.append({
            "threshold": float(t),
            "f1": m["f1"],
            "balanced_accuracy": m["balanced_accuracy"],
            "youden_j": m["youden_j"],
            "precision": m["precision"],
            "recall": m["recall"],
            "specificity": m["specificity"],
        })

    table = pd.DataFrame(rows)
    best_value = table[objective].max()
    candidates = table[np.isclose(table[objective], best_value, equal_nan=False)].copy()

    candidates["distance_to_0_5"] = np.abs(candidates["threshold"] - 0.5)
    candidates = candidates.sort_values(
        ["specificity", "recall", "distance_to_0_5"],
        ascending=[False, False, True],
    )
    return float(candidates.iloc[0]["threshold"]), table


def load_consensus(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"WD_consensus", "WD_probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    out = df[df["WD_consensus"].notna()].copy()
    out["WD_consensus"] = out["WD_consensus"].astype(int)
    out["WD_probability"] = out["WD_probability"].astype(float)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--folds", default="1,2,3,4,5")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "threshold_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
    strategies = ["f1", "balanced_accuracy", "youden_j", "fixed_0.5"]

    threshold_rows: List[dict] = []
    fold_metric_rows: List[dict] = []
    pooled_prediction_rows: List[pd.DataFrame] = []

    print("WD_P VALIDATION-DERIVED THRESHOLD ANALYSIS")
    print("=" * 72)
    print("Run:", run_dir)
    print("Threshold selection: consensus-only validation rows")
    print("Evaluation: consensus-only test rows")
    print()

    for fold in folds:
        fold_dir = run_dir / f"fold_{fold}"
        val = load_consensus(fold_dir / "val_predictions.csv")
        test = load_consensus(fold_dir / "test_predictions.csv")

        y_val = val["WD_consensus"].to_numpy(dtype=int)
        p_val = val["WD_probability"].to_numpy(dtype=float)
        y_test = test["WD_consensus"].to_numpy(dtype=int)
        p_test = test["WD_probability"].to_numpy(dtype=float)

        thresholds = {}
        for strategy in ["f1", "balanced_accuracy", "youden_j"]:
            t, search_table = choose_best_threshold(y_val, p_val, strategy)
            thresholds[strategy] = t
            search_table.to_csv(
                output_dir / f"fold_{fold}_{strategy}_validation_search.csv",
                index=False,
            )
        thresholds["fixed_0.5"] = 0.5

        print(
            f"Fold {fold}: val n={len(val)} | test n={len(test)} | "
            f"test prevalence={y_test.mean():.3f}"
        )
        print("  " + " | ".join(f"{s}={thresholds[s]:.3f}" for s in strategies))

        fold_test_predictions = test.copy()
        fold_test_predictions["outer_fold"] = fold

        for strategy in strategies:
            t = float(thresholds[strategy])
            val_metrics = confusion_metrics(y_val, p_val, t)
            test_metrics = confusion_metrics(y_test, p_test, t)

            threshold_rows.append({
                "outer_fold": fold,
                "strategy": strategy,
                "selected_threshold": t,
                "val_n": len(val),
                "val_prevalence": float(y_val.mean()),
                "val_f1": val_metrics["f1"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_recall": val_metrics["recall"],
                "val_specificity": val_metrics["specificity"],
            })

            fold_metric_rows.append({
                "outer_fold": fold,
                "strategy": strategy,
                **test_metrics,
            })

            fold_test_predictions[f"threshold_{strategy}"] = t
            fold_test_predictions[f"pred_{strategy}"] = (
                fold_test_predictions["WD_probability"] >= t
            ).astype(int)

        pooled_prediction_rows.append(fold_test_predictions)

    thresholds_df = pd.DataFrame(threshold_rows)
    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    pooled_df = pd.concat(pooled_prediction_rows, ignore_index=True)

    thresholds_df.to_csv(output_dir / "thresholds_by_fold.csv", index=False)
    fold_metrics_df.to_csv(output_dir / "test_metrics_by_fold_strategy.csv", index=False)
    pooled_df.to_csv(output_dir / "pooled_predictions_with_thresholds.csv", index=False)

    y = pooled_df["WD_consensus"].to_numpy(dtype=int)
    p = pooled_df["WD_probability"].to_numpy(dtype=float)

    pooled_summary_rows = []
    for strategy in strategies:
        pred = pooled_df[f"pred_{strategy}"].to_numpy(dtype=int)

        tp = int(np.sum((y == 1) & (pred == 1)))
        fp = int(np.sum((y == 0) & (pred == 1)))
        fn = int(np.sum((y == 1) & (pred == 0)))
        tn = int(np.sum((y == 0) & (pred == 0)))

        precision, recall, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        balanced_accuracy = (recall + specificity) / 2.0
        fold_sub = fold_metrics_df[fold_metrics_df["strategy"] == strategy]
        auprc = safe_auprc(y, p)
        prevalence = float(y.mean())

        pooled_summary_rows.append({
            "strategy": strategy,
            "n": len(y),
            "prevalence": prevalence,
            "AUROC": safe_auc(y, p),
            "AUPRC": auprc,
            "AUPRC_minus_prevalence": auprc - prevalence,
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "balanced_accuracy": float(balanced_accuracy),
            "youden_j": float(recall + specificity - 1.0),
            "f1": float(f1),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "mean_fold_balanced_accuracy": float(fold_sub["balanced_accuracy"].mean()),
            "sd_fold_balanced_accuracy": float(fold_sub["balanced_accuracy"].std(ddof=0)),
            "mean_fold_specificity": float(fold_sub["specificity"].mean()),
            "mean_fold_recall": float(fold_sub["recall"].mean()),
        })

    pooled_summary = pd.DataFrame(pooled_summary_rows)
    pooled_summary.to_csv(output_dir / "pooled_strategy_summary.csv", index=False)

    summary_lines = [
        "# WD_P Threshold / Calibration Analysis",
        "",
        "Thresholds were selected using consensus-only validation rows inside each fold and then frozen before evaluation on that fold's consensus-only test patients.",
        "",
        "Note: maximizing balanced accuracy and maximizing Youden's J are mathematically equivalent because balanced accuracy=(J+1)/2, so they should normally select the same operating point.",
        "",
        "## Pooled held-out results",
        "",
        pooled_summary[[
            "strategy", "precision", "recall", "specificity",
            "balanced_accuracy", "f1", "TP", "FP", "FN", "TN"
        ]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "AUROC and AUPRC are ranking metrics and therefore do not change when only the decision threshold changes.",
        "",
        "## Fold-specific thresholds",
        "",
        thresholds_df[[
            "outer_fold", "strategy", "selected_threshold", "val_f1",
            "val_balanced_accuracy", "val_recall", "val_specificity"
        ]].to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (output_dir / "threshold_calibration_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    print("\nPOOLED HELD-OUT RESULTS")
    print("=" * 72)
    print(
        pooled_summary[[
            "strategy", "precision", "recall", "specificity",
            "balanced_accuracy", "f1", "TP", "FP", "FN", "TN"
        ]].to_string(index=False)
    )
    print("\nSaved:", output_dir)


if __name__ == "__main__":
    main()