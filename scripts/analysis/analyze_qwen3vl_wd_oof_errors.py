#!/usr/bin/env python
"""
Systematic error analysis for patient-disjoint Qwen3-VL WD_P predictions.

Designed for the soft-label 5-fold output:
    output/qwen3vl_wd_soft_groupcv_thr2_5fold_1epoch/
        oof_consensus_predictions.csv

The script:
1. Reconstructs TP/TN/FP/FN using each fold's validation-derived threshold.
2. Ranks examples by confidence margin from that fold-specific threshold.
3. Creates a diverse manual-review sample, limiting examples per patient.
4. Creates per-patient and per-fold error summaries.
5. Adds empty columns for structured manual visual review.
6. Writes a short Markdown summary for the next meeting.

No video is decoded and no model inference is run. This is analysis only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "sample_id",
    "patient_id",
    "video",
    "segment_id",
    "WD_P_rater1",
    "WD_P_rater2",
    "WD_consensus",
    "WD_probability",
    "outer_fold",
    "fold_selected_threshold",
}

ERROR_TYPES = ["TP", "TN", "FP", "FN"]


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def classify_row(y: int, pred: int) -> str:
    if y == 1 and pred == 1:
        return "TP"
    if y == 0 and pred == 0:
        return "TN"
    if y == 0 and pred == 1:
        return "FP"
    return "FN"


def add_error_fields(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    # Consensus rows only.
    work = work[work["WD_consensus"].notna()].copy()
    work["WD_consensus"] = work["WD_consensus"].astype(int)
    work["patient_id"] = work["patient_id"].astype(str)
    work["outer_fold"] = work["outer_fold"].astype(int)

    work["predicted_class"] = (
        work["WD_probability"].astype(float)
        >= work["fold_selected_threshold"].astype(float)
    ).astype(int)

    work["error_type"] = [
        classify_row(int(y), int(p))
        for y, p in zip(work["WD_consensus"], work["predicted_class"])
    ]
    work["correct"] = (work["WD_consensus"] == work["predicted_class"]).astype(int)

    # Positive margin = model is on the positive side of its fold threshold.
    work["signed_threshold_margin"] = (
        work["WD_probability"].astype(float)
        - work["fold_selected_threshold"].astype(float)
    )
    work["confidence_margin"] = work["signed_threshold_margin"].abs()

    # Confidence in the actually predicted class. Useful for identifying
    # high-confidence errors. Fold thresholds vary, so threshold margin is
    # the primary ranking variable rather than raw probability alone.
    work["prediction_confidence_score"] = np.where(
        work["predicted_class"] == 1,
        work["WD_probability"].astype(float),
        1.0 - work["WD_probability"].astype(float),
    )

    # Exact rater-pair description is useful because 2/2 and 4/5 are both
    # consensus-positive at threshold >=2 but are clinically different in
    # salience.
    r1 = work["WD_P_rater1"].astype(float)
    r2 = work["WD_P_rater2"].astype(float)
    work["rater_pair"] = (
        r1.map(lambda x: f"{x:g}") + "/" + r2.map(lambda x: f"{x:g}")
    )
    work["rater_mean"] = (r1 + r2) / 2.0
    work["rater_abs_difference"] = (r1 - r2).abs()

    # Percentile is computed within fold, because calibration differs by fold.
    work["confidence_margin_percentile_within_fold"] = (
        work.groupby("outer_fold")["confidence_margin"]
        .rank(method="average", pct=True)
    )

    return work


def confusion_summary(group: pd.DataFrame) -> Dict[str, float]:
    counts = group["error_type"].value_counts()
    tp = int(counts.get("TP", 0))
    tn = int(counts.get("TN", 0))
    fp = int(counts.get("FP", 0))
    fn = int(counts.get("FN", 0))
    n = tp + tn + fp + fn

    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)

    return {
        "n": n,
        "positive_n": int((group["WD_consensus"] == 1).sum()),
        "negative_n": int((group["WD_consensus"] == 0).sum()),
        "prevalence": float(group["WD_consensus"].mean()) if n else float("nan"),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision_ppv": safe_div(tp, tp + fp),
        "npv": safe_div(tn, tn + fn),
        "accuracy": safe_div(tp + tn, n),
        "balanced_accuracy": (
            float(np.nanmean([sensitivity, specificity]))
            if n else float("nan")
        ),
        "mean_probability": float(group["WD_probability"].mean()) if n else float("nan"),
        "mean_probability_positive": (
            float(group.loc[group["WD_consensus"] == 1, "WD_probability"].mean())
            if (group["WD_consensus"] == 1).any()
            else float("nan")
        ),
        "mean_probability_negative": (
            float(group.loc[group["WD_consensus"] == 0, "WD_probability"].mean())
            if (group["WD_consensus"] == 0).any()
            else float("nan")
        ),
    }


def build_group_summary(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for group_value, group in df.groupby(group_column, sort=True):
        row = {group_column: group_value}
        row.update(confusion_summary(group))
        rows.append(row)
    return pd.DataFrame(rows)


def select_diverse_examples(
    df: pd.DataFrame,
    n_per_type: int,
    max_per_patient: int,
) -> pd.DataFrame:
    """Select confident TP/TN/FP/FN while avoiding one-patient domination."""
    selected = []

    for error_type in ERROR_TYPES:
        candidates = df[df["error_type"] == error_type].copy()

        # Primary: far from fold-specific threshold. Secondary: raw model
        # confidence in its predicted class.
        candidates = candidates.sort_values(
            [
                "confidence_margin",
                "prediction_confidence_score",
            ],
            ascending=[False, False],
        )

        patient_counts: Dict[str, int] = {}
        chosen = []

        for _, row in candidates.iterrows():
            pid = str(row["patient_id"])
            if patient_counts.get(pid, 0) >= max_per_patient:
                continue
            chosen.append(row)
            patient_counts[pid] = patient_counts.get(pid, 0) + 1
            if len(chosen) >= n_per_type:
                break

        # If the diversity constraint makes it impossible to reach n_per_type,
        # fill the remaining slots from the most confident unused examples.
        if len(chosen) < n_per_type:
            chosen_ids = {str(row["sample_id"]) for row in chosen}
            for _, row in candidates.iterrows():
                if str(row["sample_id"]) in chosen_ids:
                    continue
                chosen.append(row)
                chosen_ids.add(str(row["sample_id"]))
                if len(chosen) >= n_per_type:
                    break

        if chosen:
            part = pd.DataFrame(chosen)
            part["review_rank_within_type"] = np.arange(1, len(part) + 1)
            selected.append(part)

    if not selected:
        return pd.DataFrame()

    review = pd.concat(selected, ignore_index=True)

    # Structured manual-review fields. These are intentionally descriptive,
    # not inferred labels. Fill them after watching the clips.
    manual_fields = {
        "review_status": "TODO",
        "gaze_visible_notes": "",
        "head_face_visible_notes": "",
        "hands_arms_visible_notes": "",
        "torso_posture_visible_notes": "",
        "movement_temporal_notes": "",
        "other_visible_behavior_notes": "",
        "crop_or_visibility_issue": "",
        "verbal_evidence_if_transcript_checked": "",
        "possible_reason_model_correct_or_wrong": "",
        "reviewer_notes": "",
    }
    for column, default in manual_fields.items():
        review[column] = default

    preferred = [
        "error_type",
        "review_rank_within_type",
        "sample_id",
        "patient_id",
        "outer_fold",
        "video",
        "segment_id",
        "WD_P_rater1",
        "WD_P_rater2",
        "rater_pair",
        "rater_mean",
        "WD_consensus",
        "WD_probability",
        "fold_selected_threshold",
        "predicted_class",
        "signed_threshold_margin",
        "confidence_margin",
        "confidence_margin_percentile_within_fold",
    ] + list(manual_fields.keys())

    remaining = [c for c in review.columns if c not in preferred]
    return review[preferred + remaining]


def build_error_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    for error_type in ERROR_TYPES:
        g = df[df["error_type"] == error_type]
        rows.append(
            {
                "error_type": error_type,
                "n": len(g),
                "fraction_of_all": safe_div(len(g), total),
                "mean_probability": float(g["WD_probability"].mean()) if len(g) else float("nan"),
                "median_probability": float(g["WD_probability"].median()) if len(g) else float("nan"),
                "mean_threshold_margin": float(g["confidence_margin"].mean()) if len(g) else float("nan"),
                "median_threshold_margin": float(g["confidence_margin"].median()) if len(g) else float("nan"),
                "mean_rater_mean": float(g["rater_mean"].mean()) if len(g) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def write_markdown_summary(
    df: pd.DataFrame,
    patient_summary: pd.DataFrame,
    fold_summary: pd.DataFrame,
    review: pd.DataFrame,
    path: Path,
) -> None:
    overall = confusion_summary(df)

    # Hardest patients by balanced accuracy, requiring both classes where
    # possible. Small/one-class patient strata are kept but are not overread.
    hardest = patient_summary.sort_values(
        ["balanced_accuracy", "n"],
        ascending=[True, False],
        na_position="last",
    ).head(5)

    lines = [
        "# WD_P systematic error analysis",
        "",
        "## Overall thresholded OOF confusion",
        "",
        f"- n = {overall['n']}",
        f"- TP = {overall['TP']}",
        f"- TN = {overall['TN']}",
        f"- FP = {overall['FP']}",
        f"- FN = {overall['FN']}",
        f"- Recall/sensitivity = {overall['sensitivity_recall']:.3f}",
        f"- Specificity = {overall['specificity']:.3f}",
        f"- Precision = {overall['precision_ppv']:.3f}",
        f"- Balanced accuracy = {overall['balanced_accuracy']:.3f}",
        "",
        "Thresholds are the validation-derived threshold from each outer fold; they are not a single global 0.5 threshold.",
        "",
        "## Manual review sample",
        "",
        f"Selected {len(review)} examples across TP/TN/FP/FN, ranked by distance from the fold-specific threshold and diversified across patients.",
        "",
        "## Patients to inspect closely",
        "",
    ]

    for _, row in hardest.iterrows():
        bal = row["balanced_accuracy"]
        bal_text = f"{bal:.3f}" if pd.notna(bal) else "NA"
        lines.append(
            f"- {row['patient_id']}: n={int(row['n'])}, "
            f"balanced accuracy={bal_text}, FP={int(row['FP'])}, FN={int(row['FN'])}"
        )

    lines += [
        "",
        "## Suggested manual questions",
        "",
        "1. Do false positives contain visible behaviors that also occur in ordinary non-withdrawal therapy minutes?",
        "2. Do false negatives have weak/ambiguous visual evidence despite positive human ratings?",
        "3. Are errors associated with crop, occlusion, gaze visibility, hands/torso visibility, or minimal movement?",
        "4. Do 2/2 positive segments fail more often than higher-salience positive segments?",
        "5. Are the same visual patterns interpreted differently across patients?",
        "",
        "Do not infer withdrawal directly from any single visible cue; use the review to compare positive and negative contexts.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to oof_consensus_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--n-per-type",
        type=int,
        default=8,
        help="Manual-review examples per TP/TN/FP/FN category.",
    )
    parser.add_argument(
        "--max-per-patient",
        type=int,
        default=2,
        help="Preferred maximum selected examples per patient per category.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "Predictions file is missing required columns: " + ", ".join(missing)
        )

    analysis = add_error_fields(df)
    analysis.to_csv(args.output_dir / "error_analysis_all_consensus.csv", index=False)

    error_summary = build_error_type_summary(analysis)
    error_summary.to_csv(args.output_dir / "error_type_summary.csv", index=False)

    patient_summary = build_group_summary(analysis, "patient_id")
    patient_summary.to_csv(args.output_dir / "error_summary_by_patient.csv", index=False)

    fold_summary = build_group_summary(analysis, "outer_fold")
    fold_summary.to_csv(args.output_dir / "error_summary_by_fold.csv", index=False)

    # Rater-pair breakdown: useful for testing whether low-salience consensus
    # positives (e.g., 2/2) are disproportionately difficult.
    rater_pair_summary = (
        analysis.groupby(["rater_pair", "WD_consensus", "error_type"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    rater_pair_summary.to_csv(
        args.output_dir / "error_summary_by_rater_pair.csv",
        index=False,
    )

    review = select_diverse_examples(
        analysis,
        n_per_type=args.n_per_type,
        max_per_patient=args.max_per_patient,
    )
    review.to_csv(args.output_dir / "manual_review_sample.csv", index=False)

    write_markdown_summary(
        analysis,
        patient_summary,
        fold_summary,
        review,
        args.output_dir / "error_analysis_summary.md",
    )

    overall = confusion_summary(analysis)

    print("SYSTEMATIC WD_P ERROR ANALYSIS")
    print("=" * 72)
    print(f"Input: {args.predictions}")
    print(f"Consensus rows: {len(analysis)}")
    print(
        f"TP={overall['TP']} | TN={overall['TN']} | "
        f"FP={overall['FP']} | FN={overall['FN']}"
    )
    print(f"Recall:      {overall['sensitivity_recall']:.4f}")
    print(f"Specificity: {overall['specificity']:.4f}")
    print(f"Precision:   {overall['precision_ppv']:.4f}")
    print(f"Balanced acc:{overall['balanced_accuracy']:.4f}")
    print()
    print("Error-type counts:")
    print(error_summary.to_string(index=False))
    print()
    print(f"Manual review sample: {len(review)} rows")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()