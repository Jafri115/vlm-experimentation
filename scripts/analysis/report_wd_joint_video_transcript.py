#!/usr/bin/env python
"""Compare joint Qwen3-VL with matched LLM, VLM, and equal late fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)


def id_col(frame: pd.DataFrame) -> str:
    return "segment_uid" if "segment_uid" in frame and frame["segment_uid"].notna().all() else "sample_id"


def load(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    key = id_col(frame)
    probability = next((c for c in ("WD_probability", "probability") if c in frame), None)
    if probability is None:
        raise ValueError(f"No probability column in {path}")
    keep = [key, "patient_id", probability]
    for column in ("WD_consensus", "y_true"):
        if column in frame:
            keep.append(column)
    out = frame[keep].copy().rename(columns={key: "sample_id", probability: f"p_{prefix}"})
    label = "WD_consensus" if "WD_consensus" in out else "y_true"
    out = out[out[label].notna()].copy()
    out["y_true"] = out[label].astype(int)
    return out[["sample_id", "patient_id", "y_true", f"p_{prefix}"]]


def threshold_balanced_accuracy(frame: pd.DataFrame, probability: str) -> float:
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (frame[probability] >= threshold).astype(int)
        rows.append((balanced_accuracy_score(frame.y_true, pred), recall_score(frame.y_true, pred), abs(threshold - 0.5), threshold))
    # Match the nested late-fusion convention: maximize validation BA, then
    # prefer higher recall and the threshold closest to 0.5.
    return float(sorted(rows, key=lambda x: (-x[0], -x[1], x[2]))[0][3])


def metrics(y, p, threshold):
    y, p = np.asarray(y, int), np.asarray(p, float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auroc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    average_precision = average_precision_score(y, p) if np.any(y == 1) else np.nan
    return {
        "N": len(y), "threshold": float(threshold), "TN": int(tn), "FP": int(fp),
        "FN": int(fn), "TP": int(tp), "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred), "specificity": tn / max(tn + fp, 1),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0), "accuracy": accuracy_score(y, pred),
        "auroc_pooled": auroc,
        "average_precision_noninterpolated": average_precision,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value)
    headers = [str(c) for c in frame.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    rows.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def bootstrap_difference(frame, candidate, baseline, thresholds, n, seed):
    rng = np.random.default_rng(seed)
    patients = frame.patient_id.astype(str).unique()
    observed = metrics(frame.y_true, frame[candidate], thresholds[candidate])["balanced_accuracy"] - metrics(frame.y_true, frame[baseline], thresholds[baseline])["balanced_accuracy"]
    draws = []
    groups = {p: frame[frame.patient_id.astype(str) == p] for p in patients}
    for _ in range(n):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        boot = pd.concat([groups[p] for p in sampled], ignore_index=True)
        if boot.y_true.nunique() < 2:
            continue
        a = metrics(boot.y_true, boot[candidate], thresholds[candidate])["balanced_accuracy"]
        b = metrics(boot.y_true, boot[baseline], thresholds[baseline])["balanced_accuracy"]
        draws.append(a - b)
    return {
        "observed_difference": observed, "bootstrap_mean": float(np.mean(draws)),
        "ci_95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "resamples_requested": n, "resamples_used": len(draws), "seed": seed,
    }


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    fold_frames, thresholds_by_fold = [], []
    for fold in range(1, 6):
        test_parts = {
            "joint": load(args.joint_root / f"fold_{fold}" / "test_predictions.csv", "joint"),
            "llm": load(args.llm_root / f"fold_{fold}" / "test_predictions.csv", "llm"),
            "vlm": load(args.vlm_root / f"fold_{fold}" / "test_predictions.csv", "vlm"),
        }
        val_parts = {
            "joint": load(args.joint_root / f"fold_{fold}" / "val_predictions.csv", "joint"),
            "llm": load(args.llm_root / f"fold_{fold}" / "val_predictions.csv", "llm"),
            "vlm": load(args.vlm_root / f"fold_{fold}" / "val_predictions.csv", "vlm"),
        }
        test = test_parts["joint"]
        val = val_parts["joint"]
        for name in ("llm", "vlm"):
            test = test.merge(test_parts[name].drop(columns=["patient_id", "y_true"]), on="sample_id", validate="one_to_one")
            val = val.merge(val_parts[name].drop(columns=["patient_id", "y_true"]), on="sample_id", validate="one_to_one")
        test["p_equal_late_fusion"] = (test.p_llm + test.p_vlm) / 2
        val["p_equal_late_fusion"] = (val.p_llm + val.p_vlm) / 2
        selected = {column: threshold_balanced_accuracy(val, column) for column in ("p_joint", "p_llm", "p_vlm", "p_equal_late_fusion")}
        test["outer_fold"] = fold
        for column, threshold in selected.items():
            test[f"threshold_{column}"] = threshold
        fold_frames.append(test)
        thresholds_by_fold.append({"outer_fold": fold, **selected})

    pooled = pd.concat(fold_frames, ignore_index=True)
    if pooled.sample_id.duplicated().any():
        raise ValueError("A segment occurs in more than one outer test fold")
    pooled.to_csv(args.output / "matched_outer_test_predictions.csv", index=False)
    pd.DataFrame(thresholds_by_fold).to_csv(args.output / "validation_selected_thresholds.csv", index=False)

    results = []
    fold_results = []
    patient_results = []
    model_columns = {"Joint video-transcript": "p_joint", "Transcript LLM": "p_llm", "Video VLM": "p_vlm", "Equal late fusion": "p_equal_late_fusion"}
    for label, column in model_columns.items():
        fixed = metrics(pooled.y_true, pooled[column], 0.5)
        pred = np.zeros(len(pooled), dtype=int)
        for fold in range(1, 6):
            mask = pooled.outer_fold == fold
            threshold = float(pooled.loc[mask, f"threshold_{column}"].iloc[0])
            pred[mask] = (pooled.loc[mask, column] >= threshold).astype(int)
        # Metrics from fold-specific frozen thresholds; ranking metrics remain pooled.
        selected = metrics(pooled.y_true, pred.astype(float), 0.5)
        selected["auroc_pooled"] = fixed["auroc_pooled"]
        selected["average_precision_noninterpolated"] = fixed["average_precision_noninterpolated"]
        selected["threshold"] = "fold-specific validation-selected"
        results.extend([{"model": label, "setting": "fixed_0.5", **fixed}, {"model": label, "setting": "validation_selected", **selected}])
        for fold in range(1, 6):
            group = pooled[pooled.outer_fold == fold]
            threshold = float(group[f"threshold_{column}"].iloc[0])
            fold_results.append({"model": label, "outer_fold": fold, **metrics(group.y_true, group[column], threshold)})
        for patient_id, group in pooled.groupby("patient_id"):
            fold = int(group.outer_fold.iloc[0])
            threshold = float(group[f"threshold_{column}"].iloc[0])
            patient_results.append({"model": label, "patient_id": str(patient_id), "outer_fold": fold, **metrics(group.y_true, group[column], threshold)})
    pd.DataFrame(results).to_csv(args.output / "pooled_metrics.csv", index=False)
    pd.DataFrame(fold_results).to_csv(args.output / "per_fold_metrics.csv", index=False)
    pd.DataFrame(patient_results).to_csv(args.output / "per_patient_metrics.csv", index=False)

    # For patient bootstrap, use each row's frozen fold-specific decisions as 0/1 scores.
    decision_thresholds = {}
    for column in model_columns.values():
        decision = f"d_{column}"
        pooled[decision] = 0.0
        for fold in range(1, 6):
            mask = pooled.outer_fold == fold
            t = float(pooled.loc[mask, f"threshold_{column}"].iloc[0])
            pooled.loc[mask, decision] = (pooled.loc[mask, column] >= t).astype(float)
        decision_thresholds[decision] = 0.5
    comparisons = {
        baseline: bootstrap_difference(pooled, "d_p_joint", f"d_{column}", decision_thresholds, args.bootstrap, args.seed)
        for baseline, column in (("Transcript LLM", "p_llm"), ("Video VLM", "p_vlm"), ("Equal late fusion", "p_equal_late_fusion"))
    }
    payload = {"experiment": "joint video-transcript fusion", "bootstrap_note": "Paired patient resampling of saved outer-test predictions; retraining variability is not included.", "comparisons": comparisons}
    (args.output / "paired_bootstrap.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    table = pd.DataFrame(results)
    lines = ["# Joint video-transcript fusion", "", "Exploratory model-development comparison on frozen patient-disjoint outer folds.", "", "## Pooled outer-test metrics", "", markdown_table(table), "", "## Paired patient bootstrap", ""]
    for baseline, result in comparisons.items():
        lo, hi = result["ci_95"]
        lines.append(f"- Joint minus {baseline}: observed {result['observed_difference']:+.4f}; bootstrap mean {result['bootstrap_mean']:+.4f}; 95% CI [{lo:+.4f}, {hi:+.4f}].")
    lines.extend(["", "The intervals condition on saved predictions and do not include model retraining variability."])
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint-root", type=Path, required=True)
    p.add_argument("--llm-root", type=Path, required=True)
    p.add_argument("--vlm-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
