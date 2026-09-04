
#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CONDITIONS = ("direct", "windowed_direct", "describe_judge")
DEFAULT_SEGMENTS = "4,10,16,63,65"


def parse_indices(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def patient_type(row):
    wd = float(row["WD_P_mean"]) > 1.0
    cf = float(row["CF_P_mean"]) > 1.0
    if wd and cf:
        return "MIXED_P"
    if wd:
        return "WD_P"
    if cf:
        return "CF_P"
    return "NO_RUPTURE"


def binary_metrics(y_true, y_pred):
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity
        else 0.0
    )
    return {
        "N": len(y_true),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": round((tp + tn) / len(y_true), 4) if y_true else 0.0,
        "balanced_accuracy": round((sensitivity + specificity) / 2, 4),
        "precision": round(precision, 4),
        "recall_sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
    }


def prepare_predictions(path, model_name, indices):
    frame = pd.read_csv(path)
    required = {
        "segment_idx",
        "condition",
        "status",
        "primary_label",
        "rupture_present",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    frame["segment_idx"] = pd.to_numeric(frame["segment_idx"], errors="raise").astype(int)
    frame = frame[
        frame["segment_idx"].isin(indices)
        & frame["condition"].isin(CONDITIONS)
        & frame["status"].eq("ok")
    ].copy()

    frame = frame.sort_values(["segment_idx", "condition"]).drop_duplicates(
        ["segment_idx", "condition"],
        keep="last",
    )
    return frame.rename(
        columns={
            "primary_label": f"{model_name}_label",
            "rupture_present": f"{model_name}_rupture",
            "strength": f"{model_name}_strength",
            "reason": f"{model_name}_reason",
        }
    )


def prepare_labels(path, indices):
    labels = pd.read_csv(path)
    required = {
        "eval_id",
        "human_binary",
        "WD_P_mean",
        "WD_T_mean",
        "CF_P_mean",
        "CF_T_mean",
    }
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    labels = labels.rename(columns={"eval_id": "segment_idx"})
    labels["segment_idx"] = pd.to_numeric(labels["segment_idx"], errors="raise").astype(int)
    labels = labels[labels["segment_idx"].isin(indices)].copy()
    labels["legacy_any_3rs_marker"] = pd.to_numeric(
        labels["human_binary"], errors="raise"
    ).astype(int)
    labels["patient_only_rupture"] = (
        (pd.to_numeric(labels["WD_P_mean"], errors="raise") > 1.0)
        | (pd.to_numeric(labels["CF_P_mean"], errors="raise") > 1.0)
    ).astype(int)
    labels["patient_type_label"] = labels.apply(patient_type, axis=1)
    return labels


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = parse_indices(args.segment_indices)

    qwen = prepare_predictions(args.qwen_predictions, "qwen", indices)
    internvl = prepare_predictions(args.internvl_predictions, "internvl", indices)
    labels = prepare_labels(args.labels, indices)

    prediction_cols = [
        "segment_idx",
        "condition",
        "qwen_label",
        "qwen_rupture",
        "qwen_strength",
        "qwen_reason",
    ]
    qwen = qwen.reindex(columns=prediction_cols)

    prediction_cols = [
        "segment_idx",
        "condition",
        "internvl_label",
        "internvl_rupture",
        "internvl_strength",
        "internvl_reason",
    ]
    internvl = internvl.reindex(columns=prediction_cols)

    comparison = qwen.merge(
        internvl,
        on=["segment_idx", "condition"],
        how="outer",
        validate="one_to_one",
    ).merge(
        labels[
            [
                "segment_idx",
                "legacy_any_3rs_marker",
                "patient_only_rupture",
                "patient_type_label",
                "WD_P_mean",
                "WD_T_mean",
                "CF_P_mean",
                "CF_T_mean",
            ]
        ],
        on="segment_idx",
        how="left",
        validate="many_to_one",
    )

    comparison["models_agree"] = (
        comparison["qwen_label"] == comparison["internvl_label"]
    )
    comparison["qwen_type_correct"] = (
        comparison["qwen_label"] == comparison["patient_type_label"]
    )
    comparison["internvl_type_correct"] = (
        comparison["internvl_label"] == comparison["patient_type_label"]
    )
    comparison = comparison.sort_values(["segment_idx", "condition"])

    comparison_path = output_dir / "five_segment_side_by_side.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    metric_rows = []
    for model in ("qwen", "internvl"):
        pred_col = f"{model}_rupture"
        for condition in CONDITIONS:
            condition_rows = comparison[comparison["condition"] == condition]
            for target in ("legacy_any_3rs_marker", "patient_only_rupture"):
                subset = condition_rows.dropna(subset=[pred_col, target])
                metrics = binary_metrics(
                    subset[target].astype(int).tolist(),
                    subset[pred_col].astype(int).tolist(),
                )
                metric_rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "target": target,
                        **metrics,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "five_segment_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    disagreements = comparison[~comparison["models_agree"].fillna(False)]
    disagreements.to_csv(
        output_dir / "five_segment_disagreements.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missing_pairs = comparison[
        comparison["qwen_label"].isna() | comparison["internvl_label"].isna()
    ][["segment_idx", "condition", "qwen_label", "internvl_label"]]

    print("\nSIDE-BY-SIDE PREDICTIONS")
    print(
        comparison[
            [
                "segment_idx",
                "condition",
                "patient_type_label",
                "qwen_label",
                "internvl_label",
                "models_agree",
            ]
        ].to_string(index=False)
    )
    print("\nBINARY METRICS")
    print(metrics.to_string(index=False))
    if not missing_pairs.empty:
        print("\nWARNING: missing model predictions for these pairs:")
        print(missing_pairs.to_string(index=False))
    print(f"\nSaved comparison outputs to: {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare Qwen and InternVL predictions against human 3RS labels."
    )
    parser.add_argument("--qwen-predictions", required=True)
    parser.add_argument("--internvl-predictions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--segment-indices", default=DEFAULT_SEGMENTS)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())