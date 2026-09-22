#!/usr/bin/env python
"""
Qwen3-VL WD_P consensus-only patient-grouped cross-validation — 3-epoch experiment
=============================================================

Scientific question
-------------------
Does the consensus-only WD_P >= threshold classifier generalize across
patients, rather than only on one fixed two-patient test split?

This script reuses the tested implementation in:
    finetune_qwen3vl_wd_consensus_binary.py

Key design
----------
- Same Qwen3-VL 8B 4-bit QLoRA architecture
- 3 training epochs with best validation-AUPRC checkpoint restoration
- Same 3RS-informed visual-only prompt
- Same 16 patient frames / 224 px / mean_all default
- Same consensus target:
      both raters below threshold -> 0
      both raters at/above threshold -> 1
      binary rater disagreement -> excluded from TRAINING
- OUTER patient-grouped cross-validation for held-out testing
- INNER patient-grouped split for validation / checkpoint selection
- No patient appears in train, validation, and test within a fold
- Original manifest train/val/test labels are ignored for CV assignment
- Each outer patient is tested exactly once when all folds are run

Important interpretation
------------------------
This is exploratory cross-validation over the available role-cache dataset.
It is NOT a new untouched external test set.

Primary outputs
---------------
- cv_patient_assignments.csv
- cv_fold_summary.csv
- oof_predictions.csv
- oof_consensus_predictions.csv
- oof_patient_metrics.csv
- cv_summary.json
- fold_metrics.png

The most informative diagnostics are:
- macro outer-fold AUROC
- macro outer-fold AUPRC versus each fold's prevalence baseline
- macro within-patient AUROC
- patient-centered pooled AUROC
- probability offsets by patient

Place this file in:
    VLM_experiments/scripts/

alongside:
    finetune_qwen3vl_wd_consensus_binary.py
    finetune_qwen3vl_rupture_pilot.py
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vlm import finetune_qwen3vl_wd_consensus_binary as exp


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            json_clean(obj),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def safe_auroc(y: Sequence[int], p: Sequence[float]) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_auprc(y: Sequence[int], p: Sequence[float]) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) == 0 or y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, p))


def mean_sd(values: Iterable[float]) -> Tuple[float, float, int]:
    arr = np.asarray(
        [x for x in values if x is not None and np.isfinite(x)],
        dtype=float,
    )
    if len(arr) == 0:
        return float("nan"), float("nan"), 0
    return float(arr.mean()), float(arr.std(ddof=0)), int(len(arr))


def parse_fold_indices(text: str, n_folds: int) -> List[int]:
    text = str(text or "all").strip().lower()
    if text in {"all", "*"}:
        return list(range(1, n_folds + 1))

    out = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1 or value > n_folds:
            raise ValueError(
                f"Fold {value} is outside 1..{n_folds}."
            )
        out.append(value)

    out = sorted(set(out))
    if not out:
        raise ValueError("No CV folds selected.")
    return out


# ---------------------------------------------------------------------------
# Patient-grouped fold construction
# ---------------------------------------------------------------------------

def _splitter(
    n_splits: int,
    seed: int,
    y: np.ndarray,
    groups: np.ndarray,
):
    """
    Prefer StratifiedGroupKFold to balance binary target prevalence while
    keeping patients intact. Fall back to GroupKFold if unavailable.
    """
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return splitter.split(
            X=np.zeros((len(y), 1), dtype=np.float32),
            y=y,
            groups=groups,
        )

    splitter = GroupKFold(n_splits=n_splits)
    return splitter.split(
        X=np.zeros((len(y), 1), dtype=np.float32),
        y=y,
        groups=groups,
    )


def build_outer_patient_folds(
    manifest: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> Dict[str, int]:
    consensus = manifest[
        manifest["WD_consensus"].notna()
    ].copy()

    if consensus.empty:
        raise RuntimeError("No consensus rows available for CV.")

    patients = sorted(
        consensus["patient_id"].astype(str).unique().tolist()
    )

    if len(patients) < n_folds:
        raise RuntimeError(
            f"Only {len(patients)} patients have consensus rows, "
            f"but --cv-folds={n_folds}."
        )

    y = consensus["WD_consensus"].astype(int).to_numpy()
    groups = consensus["patient_id"].astype(str).to_numpy()

    patient_to_fold: Dict[str, int] = {}

    for fold_idx, (_, test_idx) in enumerate(
        _splitter(
            n_splits=n_folds,
            seed=seed,
            y=y,
            groups=groups,
        ),
        start=1,
    ):
        fold_patients = sorted(
            set(groups[test_idx].tolist())
        )

        for patient_id in fold_patients:
            if patient_id in patient_to_fold:
                raise RuntimeError(
                    f"Patient {patient_id} assigned to more than one "
                    "outer fold."
                )
            patient_to_fold[patient_id] = fold_idx

    missing = sorted(set(patients) - set(patient_to_fold))
    if missing:
        raise RuntimeError(
            "Patients missing outer-fold assignment: "
            + ", ".join(missing)
        )

    return patient_to_fold


def choose_inner_validation_patients(
    manifest: pd.DataFrame,
    outer_test_patients: Sequence[str],
    n_inner_folds: int,
    seed: int,
    outer_fold: int,
) -> List[str]:
    outer_test_patients = {str(x) for x in outer_test_patients}

    pool = manifest[
        ~manifest["patient_id"].astype(str).isin(outer_test_patients)
        & manifest["WD_consensus"].notna()
    ].copy()

    pool_patients = sorted(
        pool["patient_id"].astype(str).unique().tolist()
    )

    if len(pool_patients) < 2:
        raise RuntimeError(
            "Need at least two non-test patients for inner validation."
        )

    n_splits = min(
        int(n_inner_folds),
        len(pool_patients),
    )

    if n_splits < 2:
        raise RuntimeError("Inner CV needs at least 2 folds.")

    y = pool["WD_consensus"].astype(int).to_numpy()
    groups = pool["patient_id"].astype(str).to_numpy()

    splits = list(
        _splitter(
            n_splits=n_splits,
            seed=seed + 1000 + outer_fold,
            y=y,
            groups=groups,
        )
    )

    # Rotate which inner fold supplies validation so all outer folds do not
    # systematically choose the first partition.
    chosen = (outer_fold - 1) % len(splits)
    _, val_idx = splits[chosen]

    return sorted(set(groups[val_idx].tolist()))


def assign_fold_splits(
    manifest: pd.DataFrame,
    patient_to_outer_fold: Dict[str, int],
    outer_fold: int,
    inner_folds: int,
    seed: int,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    test_patients = sorted(
        [
            patient
            for patient, fold in patient_to_outer_fold.items()
            if fold == outer_fold
        ]
    )

    val_patients = choose_inner_validation_patients(
        manifest=manifest,
        outer_test_patients=test_patients,
        n_inner_folds=inner_folds,
        seed=seed,
        outer_fold=outer_fold,
    )

    all_patients = sorted(
        manifest["patient_id"].astype(str).unique().tolist()
    )

    train_patients = sorted(
        set(all_patients)
        - set(test_patients)
        - set(val_patients)
    )

    if (
        set(train_patients) & set(val_patients)
        or set(train_patients) & set(test_patients)
        or set(val_patients) & set(test_patients)
    ):
        raise RuntimeError("Patient leakage detected in CV split.")

    fold_manifest = manifest.copy()
    fold_manifest["split"] = "train"

    patient_str = fold_manifest["patient_id"].astype(str)

    fold_manifest.loc[
        patient_str.isin(val_patients),
        "split",
    ] = "val"

    fold_manifest.loc[
        patient_str.isin(test_patients),
        "split",
    ] = "test"

    return (
        fold_manifest,
        train_patients,
        val_patients,
        test_patients,
    )


def split_counts(
    fold_manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for split in ["train", "val", "test"]:
        sub = fold_manifest[
            fold_manifest["split"] == split
        ].copy()

        cons = sub[
            sub["WD_consensus"].notna()
        ].copy()

        rows.append(
            {
                "split": split,
                "all_rows": int(len(sub)),
                "patients": int(
                    sub["patient_id"].astype(str).nunique()
                ),
                "consensus_rows": int(len(cons)),
                "consensus_negative": int(
                    (cons["WD_consensus"] == 0).sum()
                ),
                "consensus_positive": int(
                    (cons["WD_consensus"] == 1).sum()
                ),
                "consensus_prevalence": (
                    float(cons["WD_consensus"].mean())
                    if len(cons)
                    else float("nan")
                ),
                "disagreement_rows": int(
                    sub["WD_consensus"].isna().sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CV aggregation
# ---------------------------------------------------------------------------

def load_fold_results(
    root: Path,
    fold_indices: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    oof_frames = []

    for fold in fold_indices:
        fold_dir = root / f"fold_{fold}"
        summary_path = fold_dir / "final_summary.json"
        pred_path = fold_dir / "test_predictions.csv"

        if not summary_path.exists() or not pred_path.exists():
            continue

        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )

        preds = pd.read_csv(pred_path)
        preds["outer_fold"] = fold
        preds["fold_selected_threshold"] = float(
            summary["selected_probability_threshold"]
        )

        if "WD_probability" not in preds.columns:
            raise RuntimeError(
                f"{pred_path} has no WD_probability column."
            )

        oof_frames.append(preds)

        cons_metrics = (
            summary.get("test_metrics", {})
            .get("consensus_only", {})
        )

        n = int(cons_metrics.get("n", 0) or 0)
        pos = int(cons_metrics.get("positive_n", 0) or 0)
        prevalence = float(pos / n) if n else float("nan")
        auprc = cons_metrics.get("AUPRC", float("nan"))
        auroc = cons_metrics.get("AUROC", float("nan"))

        fold_rows.append(
            {
                "outer_fold": fold,
                "consensus_n": n,
                "consensus_positive_n": pos,
                "prevalence_baseline_AUPRC": prevalence,
                "AUPRC": auprc,
                "AUPRC_minus_prevalence": (
                    float(auprc - prevalence)
                    if auprc is not None
                    and np.isfinite(auprc)
                    and np.isfinite(prevalence)
                    else float("nan")
                ),
                "AUROC": auroc,
                "AUROC_minus_0.5": (
                    float(auroc - 0.5)
                    if auroc is not None and np.isfinite(auroc)
                    else float("nan")
                ),
                "precision": cons_metrics.get(
                    "precision", float("nan")
                ),
                "recall": cons_metrics.get(
                    "recall", float("nan")
                ),
                "f1": cons_metrics.get(
                    "f1", float("nan")
                ),
                "TP": cons_metrics.get("TP", 0),
                "FP": cons_metrics.get("FP", 0),
                "FN": cons_metrics.get("FN", 0),
                "TN": cons_metrics.get("TN", 0),
                "selected_threshold": float(
                    summary["selected_probability_threshold"]
                ),
                "best_val_AUPRC": summary.get(
                    "best_val_AUPRC", float("nan")
                ),
            }
        )

    fold_df = pd.DataFrame(fold_rows)

    if oof_frames:
        oof = pd.concat(oof_frames, ignore_index=True)
    else:
        oof = pd.DataFrame()

    return fold_df, oof


def build_patient_metrics(
    oof_consensus: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for patient_id, group in oof_consensus.groupby("patient_id"):
        y = group["WD_consensus"].astype(int).to_numpy()
        p = group["WD_probability"].astype(float).to_numpy()

        neg_probs = p[y == 0]
        pos_probs = p[y == 1]

        rows.append(
            {
                "patient_id": str(patient_id),
                "outer_fold": int(group["outer_fold"].iloc[0]),
                "n": int(len(group)),
                "negative_n": int((y == 0).sum()),
                "positive_n": int((y == 1).sum()),
                "prevalence": float(y.mean()),
                "AUROC": safe_auroc(y, p),
                "AUPRC": safe_auprc(y, p),
                "prob_mean": float(p.mean()),
                "prob_std": float(p.std(ddof=0)),
                "prob_mean_negative": (
                    float(neg_probs.mean())
                    if len(neg_probs)
                    else float("nan")
                ),
                "prob_mean_positive": (
                    float(pos_probs.mean())
                    if len(pos_probs)
                    else float("nan")
                ),
                "within_patient_prob_delta": (
                    float(pos_probs.mean() - neg_probs.mean())
                    if len(pos_probs) and len(neg_probs)
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


def aggregate_cv(
    output_root: Path,
    fold_indices: Sequence[int],
) -> None:
    fold_df, oof = load_fold_results(
        root=output_root,
        fold_indices=fold_indices,
    )

    if fold_df.empty or oof.empty:
        print(
            "\nNo completed fold results available yet for aggregation."
        )
        return

    fold_df.to_csv(
        output_root / "cv_fold_summary.csv",
        index=False,
    )

    oof.to_csv(
        output_root / "oof_predictions.csv",
        index=False,
    )

    oof_consensus = oof[
        oof["WD_consensus"].notna()
    ].copy()

    oof_consensus["WD_consensus"] = (
        oof_consensus["WD_consensus"].astype(int)
    )

    oof_consensus["fold_predicted_class"] = (
        oof_consensus["WD_probability"]
        >= oof_consensus["fold_selected_threshold"]
    ).astype(int)

    oof_consensus.to_csv(
        output_root / "oof_consensus_predictions.csv",
        index=False,
    )

    patient_df = build_patient_metrics(
        oof_consensus
    )

    patient_df.to_csv(
        output_root / "oof_patient_metrics.csv",
        index=False,
    )

    y = oof_consensus["WD_consensus"].astype(int).to_numpy()
    p = oof_consensus["WD_probability"].astype(float).to_numpy()
    pred = oof_consensus["fold_predicted_class"].astype(int).to_numpy()

    pooled_auroc = safe_auroc(y, p)
    pooled_auprc = safe_auprc(y, p)
    pooled_prevalence = float(y.mean())

    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        pred,
        average="binary",
        zero_division=0,
    )

    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())

    # Remove each patient's mean probability. This specifically tests whether
    # minute-to-minute ranking survives after eliminating patient-level offsets.
    centered = oof_consensus.copy()
    centered["patient_centered_probability"] = (
        centered["WD_probability"]
        - centered.groupby("patient_id")["WD_probability"].transform("mean")
    )

    centered_auroc = safe_auroc(
        centered["WD_consensus"].astype(int).to_numpy(),
        centered["patient_centered_probability"].astype(float).to_numpy(),
    )

    centered_auprc = safe_auprc(
        centered["WD_consensus"].astype(int).to_numpy(),
        centered["patient_centered_probability"].astype(float).to_numpy(),
    )

    fold_auroc_mean, fold_auroc_sd, fold_auroc_n = mean_sd(
        fold_df["AUROC"].tolist()
    )
    fold_auprc_mean, fold_auprc_sd, fold_auprc_n = mean_sd(
        fold_df["AUPRC"].tolist()
    )
    fold_lift_mean, fold_lift_sd, fold_lift_n = mean_sd(
        fold_df["AUPRC_minus_prevalence"].tolist()
    )

    patient_auroc_mean, patient_auroc_sd, patient_auroc_n = mean_sd(
        patient_df["AUROC"].tolist()
    )
    patient_delta_mean, patient_delta_sd, patient_delta_n = mean_sd(
        patient_df["within_patient_prob_delta"].tolist()
    )

    patient_prob_prevalence_spearman = float("nan")
    if len(patient_df) >= 3:
        patient_prob_prevalence_spearman = float(
            patient_df[
                ["prob_mean", "prevalence"]
            ].corr(method="spearman").iloc[0, 1]
        )

    summary = {
        "completed_folds": fold_df["outer_fold"].astype(int).tolist(),
        "n_completed_folds": int(len(fold_df)),
        "n_oof_consensus_segments": int(len(oof_consensus)),
        "n_oof_patients": int(
            oof_consensus["patient_id"].astype(str).nunique()
        ),
        "macro_outer_fold": {
            "AUROC_mean": fold_auroc_mean,
            "AUROC_sd": fold_auroc_sd,
            "AUROC_n_folds": fold_auroc_n,
            "AUPRC_mean": fold_auprc_mean,
            "AUPRC_sd": fold_auprc_sd,
            "AUPRC_n_folds": fold_auprc_n,
            "AUPRC_minus_prevalence_mean": fold_lift_mean,
            "AUPRC_minus_prevalence_sd": fold_lift_sd,
            "AUPRC_minus_prevalence_n_folds": fold_lift_n,
        },
        "pooled_oof_consensus": {
            "prevalence_AUPRC_baseline": pooled_prevalence,
            "AUPRC": pooled_auprc,
            "AUPRC_minus_prevalence": (
                pooled_auprc - pooled_prevalence
                if np.isfinite(pooled_auprc)
                else float("nan")
            ),
            "AUROC": pooled_auroc,
            "precision_using_fold_val_thresholds": float(precision),
            "recall_using_fold_val_thresholds": float(recall),
            "f1_using_fold_val_thresholds": float(f1),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        },
        "within_patient": {
            "macro_patient_AUROC_mean": patient_auroc_mean,
            "macro_patient_AUROC_sd": patient_auroc_sd,
            "patients_with_both_classes": patient_auroc_n,
            "patient_centered_pooled_AUROC": centered_auroc,
            "patient_centered_pooled_AUPRC": centered_auprc,
            "mean_positive_minus_negative_probability": (
                patient_delta_mean
            ),
            "sd_positive_minus_negative_probability": (
                patient_delta_sd
            ),
            "patients_with_delta": patient_delta_n,
        },
        "patient_offset_diagnostic": {
            "spearman_patient_mean_probability_vs_patient_prevalence": (
                patient_prob_prevalence_spearman
            )
        },
    }

    write_json(
        output_root / "cv_summary.json",
        summary,
    )

    # Simple fold diagnostic plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(fold_df))
    width = 0.26

    ax.bar(
        x - width,
        fold_df["AUROC"].to_numpy(dtype=float),
        width,
        label="AUROC",
    )
    ax.bar(
        x,
        fold_df["AUPRC"].to_numpy(dtype=float),
        width,
        label="AUPRC",
    )
    ax.bar(
        x + width,
        fold_df["prevalence_baseline_AUPRC"].to_numpy(dtype=float),
        width,
        label="AUPRC prevalence baseline",
    )
    ax.axhline(0.5, linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Fold {int(f)}" for f in fold_df["outer_fold"]]
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(
        "Patient-grouped CV: consensus WD classification"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_root / "fold_metrics.png",
        dpi=160,
    )
    plt.close(fig)

    print("\nCROSS-VALIDATION SUMMARY")
    print("=" * 72)
    print(
        json.dumps(
            json_clean(summary),
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# Main CV driver
# ---------------------------------------------------------------------------

def make_parser():
    p = exp.make_parser()

    p.description = (
        "Qwen3-VL consensus-only WD_P patient-grouped cross-validation."
    )

    p.set_defaults(
        target_mode="consensus",
        positive_threshold=2.0,
        pos_weight="1",
        epochs=3,
        num_frames=16,
        frame_width=224,
        pooling="mean_all",
        max_train_steps=0,
        max_val_examples=0,
        max_test_examples=0,
        eval_every_steps=20,
        curve_val_examples=40,
    )

    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of outer patient-grouped test folds.",
    )

    p.add_argument(
        "--inner-folds",
        type=int,
        default=4,
        help=(
            "Patient-grouped partitions within the outer training pool. "
            "One inner partition becomes validation."
        ),
    )

    p.add_argument(
        "--fold-indices",
        default="all",
        help=(
            "Which outer folds to run, e.g. '1', '1,2,3', or 'all'. "
            "Useful for running folds separately."
        ),
    )

    p.add_argument(
        "--overwrite-folds",
        action="store_true",
        help=(
            "Rerun a fold even when fold_N/final_summary.json already exists."
        ),
    )

    return p


def main():
    args = make_parser().parse_args()

    if args.target_mode != "consensus":
        raise ValueError(
            "This CV experiment is intentionally consensus-only. "
            "Use --target-mode consensus."
        )

    if int(args.cv_folds) < 2:
        raise ValueError("--cv-folds must be >=2.")

    if int(args.inner_folds) < 2:
        raise ValueError("--inner-folds must be >=2.")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        "QWEN3-VL WD_P CONSENSUS PATIENT-GROUPED CROSS-VALIDATION — 3 EPOCHS"
    )
    print("=" * 72)
    print("Outer folds:", args.cv_folds)
    print("Inner folds:", args.inner_folds)
    print("Positive threshold:", args.positive_threshold)
    print("Prompt: 3RS-informed visual-only")
    print("Frames:", args.num_frames)
    print("Frame width:", args.frame_width)
    print("Pooling:", args.pooling)
    print("Epochs:", args.epochs)
    print("LoRA learning rate:", args.learning_rate)
    print("Head learning rate:", args.head_learning_rate)
    print(
        "Experiment intent: isolate the effect of more epochs; "
        "keep all other settings identical to the 1-epoch larger-data baseline."
    )
    print(
        "NOTE: original manifest train/val/test labels are ignored "
        "for cross-validation assignment."
    )

    # Build two-rater targets once. The helper also preserves all manifest
    # metadata required by FrameCache and Qwen training.
    manifest = exp.merge_targets_with_manifest(
        manifest_path=Path(args.manifest),
        labels_csv=Path(args.labels_csv),
        threshold=float(args.positive_threshold),
        output_dir=output_root,
    )

    manifest["patient_id"] = manifest["patient_id"].astype(str)

    patient_to_outer_fold = build_outer_patient_folds(
        manifest=manifest,
        n_folds=int(args.cv_folds),
        seed=int(args.seed),
    )

    assignment_rows = []
    for patient_id in sorted(patient_to_outer_fold):
        sub = manifest[
            manifest["patient_id"].astype(str) == str(patient_id)
        ]
        cons = sub[sub["WD_consensus"].notna()]

        assignment_rows.append(
            {
                "patient_id": str(patient_id),
                "outer_fold": int(patient_to_outer_fold[patient_id]),
                "all_rows": int(len(sub)),
                "consensus_rows": int(len(cons)),
                "consensus_negative": int(
                    (cons["WD_consensus"] == 0).sum()
                ),
                "consensus_positive": int(
                    (cons["WD_consensus"] == 1).sum()
                ),
                "consensus_prevalence": (
                    float(cons["WD_consensus"].mean())
                    if len(cons)
                    else float("nan")
                ),
            }
        )

    assignment_df = pd.DataFrame(assignment_rows)
    assignment_df.to_csv(
        output_root / "cv_patient_assignments.csv",
        index=False,
    )

    selected_folds = parse_fold_indices(
        args.fold_indices,
        int(args.cv_folds),
    )

    print("\nOUTER PATIENT ASSIGNMENTS")
    print("=" * 72)
    print(
        assignment_df.to_string(index=False)
    )

    # Precompute and save the exact split definition for every fold,
    # including the nested patient-disjoint validation set.
    fold_plan_rows = []
    fold_manifests = {}

    for fold in range(1, int(args.cv_folds) + 1):
        (
            fold_manifest,
            train_patients,
            val_patients,
            test_patients,
        ) = assign_fold_splits(
            manifest=manifest,
            patient_to_outer_fold=patient_to_outer_fold,
            outer_fold=fold,
            inner_folds=int(args.inner_folds),
            seed=int(args.seed),
        )

        fold_manifests[fold] = fold_manifest

        counts = split_counts(fold_manifest)
        counts["outer_fold"] = fold
        fold_plan_rows.append(counts)

        fold_dir = output_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_manifest.to_csv(
            fold_dir / "cv_manifest.csv",
            index=False,
        )

        write_json(
            fold_dir / "patient_split.json",
            {
                "outer_fold": fold,
                "train_patients": train_patients,
                "val_patients": val_patients,
                "test_patients": test_patients,
            },
        )

        print(f"\nFOLD {fold}")
        print("-" * 72)
        print("Train patients:", ", ".join(train_patients))
        print("Val patients:  ", ", ".join(val_patients))
        print("Test patients: ", ", ".join(test_patients))
        print(counts.to_string(index=False))

    pd.concat(
        fold_plan_rows,
        ignore_index=True,
    ).to_csv(
        output_root / "cv_fold_plan.csv",
        index=False,
    )

    if args.prepare_only:
        print(
            "\nPREPARE-ONLY complete. "
            "No Qwen model was loaded."
        )
        return

    # exp.train expects None rather than 0 for unlimited evaluation.
    if args.max_val_examples is not None and args.max_val_examples <= 0:
        args.max_val_examples = None

    if args.max_test_examples is not None and args.max_test_examples <= 0:
        args.max_test_examples = None

    for fold in selected_folds:
        fold_dir = output_root / f"fold_{fold}"
        final_summary = fold_dir / "final_summary.json"

        if final_summary.exists() and not args.overwrite_folds:
            print(
                f"\nFOLD {fold}: already complete; skipping. "
                "Use --overwrite-folds to rerun."
            )
            continue

        print("\n" + "#" * 72)
        print(f"OUTER FOLD {fold}/{args.cv_folds}")
        print("#" * 72)

        fold_args = copy.deepcopy(args)
        fold_args.output_dir = str(fold_dir)

        # Keep identical training initialization across folds so fold
        # differences primarily reflect held-out patients, not arbitrary seeds.
        fold_args.seed = int(args.seed)

        exp.train(
            fold_args,
            fold_manifests[fold],
        )

        # Ensure GPU memory is released before loading the next fold.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Re-aggregate after every completed fold so a long CV run remains
        # useful even if interrupted later.
        aggregate_cv(
            output_root=output_root,
            fold_indices=list(
                range(1, int(args.cv_folds) + 1)
            ),
        )

    aggregate_cv(
        output_root=output_root,
        fold_indices=list(
            range(1, int(args.cv_folds) + 1)
        ),
    )

    print("\nFINISHED PATIENT-GROUPED CROSS-VALIDATION")
    print("Output:", output_root)


if __name__ == "__main__":
    main()