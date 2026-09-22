#!/usr/bin/env python
"""Report ordinal and binary reliability for all expanded WD_P OOF experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def icc_a1(values: np.ndarray) -> float:
    """ICC(A,1): two-way random-effects, absolute agreement, single rating."""
    x = np.asarray(values, dtype=float)
    n, k = x.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = x.mean()
    row = x.mean(axis=1)
    col = x.mean(axis=0)
    msr = k * np.square(row - grand).sum() / (n - 1)
    msc = n * np.square(col - grand).sum() / (k - 1)
    resid = x - row[:, None] - col[None, :] + grand
    mse = np.square(resid).sum() / ((n - 1) * (k - 1))
    denominator = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denominator) if denominator else float("nan")


def gwet(values: np.ndarray, categories: list[int], quadratic: bool) -> float:
    """Gwet AC2 for ordinal data; unweighted form is binary AC1."""
    x = np.asarray(values, dtype=int)
    n, raters = x.shape
    cats = np.asarray(categories, dtype=int)
    q = len(cats)
    if quadratic:
        weights = 1 - ((cats[:, None] - cats[None, :]) / (q - 1)) ** 2
    else:
        weights = (cats[:, None] == cats[None, :]).astype(float)
    index = {value: i for i, value in enumerate(cats)}
    observed = []
    for row in x:
        positions = [index[int(value)] for value in row]
        observed.append(
            sum(
                weights[positions[i], positions[j]]
                for i in range(raters)
                for j in range(i + 1, raters)
            )
            / (raters * (raters - 1) / 2)
        )
    prevalence = np.array([(x == value).sum() / (n * raters) for value in cats])
    expected = sum(
        weights[i, j] * prevalence[i] * (1 - prevalence[j])
        for i in range(q)
        for j in range(q)
    ) / (q - 1)
    return float((np.mean(observed) - expected) / (1 - expected)) if expected < 1 else float("nan")


def round_half_up(values) -> np.ndarray:
    return np.floor(np.asarray(values, dtype=float) + 0.5).clip(1, 5).astype(int)


def mean_pairwise(function, ai, h1, h2) -> float:
    return float((function(ai, h1) + function(ai, h2)) / 2)


def ordinal_metrics(frame: pd.DataFrame, whole_number: bool = False) -> dict:
    h1 = frame.h1.to_numpy(float)
    h2 = frame.h2.to_numpy(float)
    ai = frame.ai_score.to_numpy(float)
    h1r, h2r, air = round_half_up(h1), round_half_up(h2), round_half_up(ai)
    # In the original report ICC used continuous AI scores.  The optional
    # whole-number mode makes every ordinal statistic use the displayed 1-5
    # integer prediction, including RMSE.
    ai_for_numeric = air.astype(float) if whole_number else ai

    exact = lambda a, b: float(np.mean(np.asarray(a) == np.asarray(b)))
    within = lambda a, b: float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float)) <= 1))
    ac2 = lambda a, b: gwet(np.column_stack([a, b]), list(range(1, 6)), True)
    icc = lambda a, b: icc_a1(np.column_stack([a, b]))
    qwk = lambda a, b: float(cohen_kappa_score(a, b, labels=list(range(1, 6)), weights="quadratic"))

    categorical = np.column_stack([h1r, h2r, air])
    continuous = np.column_stack([h1, h2, ai])
    return {
        "N": len(frame),
        "patients": frame.patient_id.astype(str).nunique(),
        "human_exact": exact(h1r, h2r),
        "human_within_one": within(h1, h2),
        "human_AC2_quadratic": ac2(h1r, h2r),
        "human_ICC_A1": icc(h1, h2),
        "human_quadratic_kappa": qwk(h1r, h2r),
        "AI_human_exact_mean": mean_pairwise(exact, air, h1r, h2r),
        "AI_human_within_one_mean": mean_pairwise(within, ai_for_numeric, h1, h2),
        "AI_human_AC2_quadratic_mean": mean_pairwise(ac2, air, h1r, h2r),
        "AI_human_ICC_A1_mean": mean_pairwise(icc, ai_for_numeric, h1, h2),
        "AI_human_quadratic_kappa_mean": mean_pairwise(qwk, air, h1r, h2r),
        "AI_human_RMSE_mean": float((np.sqrt(np.mean((ai_for_numeric - h1) ** 2)) + np.sqrt(np.mean((ai_for_numeric - h2) ** 2))) / 2),
        "AI_mean_rating_RMSE": float(np.sqrt(np.mean((ai_for_numeric - ((h1 + h2) / 2)) ** 2))),
        "three_rater_unanimous_exact": float(np.mean((h1r == h2r) & (h1r == air))),
        "three_rater_AC2_quadratic": gwet(categorical, list(range(1, 6)), True),
        "three_rater_ICC_A1": icc_a1(np.column_stack([h1, h2, ai_for_numeric])),
        "AI_prediction_min": float(ai_for_numeric.min()),
        "AI_prediction_max": float(ai_for_numeric.max()),
        "AI_prediction_SD": float(ai_for_numeric.std(ddof=1)),
    }


def binary_metrics(frame: pd.DataFrame) -> dict:
    h1 = (frame.h1.to_numpy(float) >= 2).astype(int)
    h2 = (frame.h2.to_numpy(float) >= 2).astype(int)
    ai = frame.ai_binary.to_numpy(int)
    exact = lambda a, b: float(np.mean(np.asarray(a) == np.asarray(b)))
    ac1 = lambda a, b: gwet(np.column_stack([a, b]), [0, 1], False)
    icc = lambda a, b: icc_a1(np.column_stack([a, b]))
    kappa = lambda a, b: float(cohen_kappa_score(a, b, labels=[0, 1]))
    three = np.column_stack([h1, h2, ai])
    return {
        "N": len(frame),
        "patients": frame.patient_id.astype(str).nunique(),
        "human_exact": exact(h1, h2),
        "human_AC1": ac1(h1, h2),
        "human_ICC_A1": icc(h1, h2),
        "human_Cohen_kappa": kappa(h1, h2),
        "AI_human_exact_mean": mean_pairwise(exact, ai, h1, h2),
        "AI_human_AC1_mean": mean_pairwise(ac1, ai, h1, h2),
        "AI_human_ICC_A1_mean": mean_pairwise(icc, ai, h1, h2),
        "AI_human_Cohen_kappa_mean": mean_pairwise(kappa, ai, h1, h2),
        "three_rater_unanimous_exact": float(np.mean((h1 == h2) & (h1 == ai))),
        "three_rater_AC1": gwet(three, [0, 1], False),
        "three_rater_ICC_A1": icc_a1(three),
        "AI_positive_rate": float(ai.mean()),
    }


def choose_key(predictions: pd.DataFrame, labels: pd.DataFrame) -> str:
    for key in ("segment_uid", "sample_id"):
        if key in predictions and key in labels:
            return key
    raise ValueError("Predictions and labels do not share segment_uid or sample_id")


def load_experiment(spec: dict, labels: pd.DataFrame) -> pd.DataFrame:
    predictions = pd.read_csv(spec["path"], encoding="utf-8-sig", low_memory=False)
    if "status" in predictions:
        predictions = predictions[
            predictions.status.astype(str).str.lower().isin(["ok", "success"])
        ].copy()
    key = choose_key(predictions, labels)
    predictions[key] = predictions[key].astype(str)
    base = labels[[key, "patient_id", "h1", "h2"]].copy()
    base[key] = base[key].astype(str)

    keep = [key]
    if spec.get("score"):
        predictions["ai_score"] = pd.to_numeric(predictions[spec["score"]], errors="coerce").clip(1, 5)
        keep.append("ai_score")
    if spec.get("probability"):
        probability = pd.to_numeric(predictions[spec["probability"]], errors="coerce")
        if spec.get("threshold_column"):
            threshold = pd.to_numeric(predictions[spec["threshold_column"]], errors="coerce")
        else:
            threshold = float(spec.get("threshold", 0.5))
        predictions["ai_binary"] = (probability >= threshold).astype(float)
        keep.append("ai_binary")
    elif spec.get("score"):
        predictions["ai_binary"] = (predictions.ai_score >= 2).astype(float)
        keep.append("ai_binary")

    predictions = predictions[keep].dropna().drop_duplicates(key, keep="last")
    merged = base.merge(predictions, on=key, how="inner", validate="one_to_one")
    merged["patient_id"] = merged.patient_id.astype(str)
    return merged


def build_specs(root: Path) -> list[dict]:
    def item(name, modality, folder, score=None, probability=None, threshold_column=None):
        return {
            "experiment": name,
            "modality": modality,
            "path": root / folder / "oof_predictions.csv",
            "score": score,
            "probability": probability,
            "threshold_column": threshold_column,
        }

    return [
        item("zero-shot", "Qwen3-8B transcript", "llm_wd_zero_expanded_cv", "wd_p_score", "WD_probability"),
        item("zero-shot", "Qwen3-VL video", "vlm_wd_zero_expanded_cv", "wd_p_score", "WD_probability"),
        item("3+3 few-shot", "Qwen3-8B transcript", "llm_wd_few_expanded_cv", "wd_p_score", "WD_probability"),
        item("3+3 few-shot", "Qwen3-VL video", "vlm_wd_few_expanded_cv", "wd_p_score", "WD_probability"),
        item("standard regression", "Qwen3-8B transcript", "llm_wd_regression_expanded_cv", "WD_prediction"),
        item("standard regression", "Qwen3-VL video", "vlm_wd_regression_expanded_cv", "WD_P_pred"),
        item("ordinal regression", "Qwen3-8B transcript", "llm_wd_ordinal_qwen3_8b_expanded_cv", "WD_prediction"),
        item("ordinal regression", "Qwen3-14B transcript", "llm_wd_ordinal_qwen3_14b_expanded_cv", "WD_prediction"),
        item("consensus fine-tuning", "Qwen3-8B transcript", "llm_wd_consensus_expanded_cv", probability="WD_probability"),
        item("consensus fine-tuning", "Qwen3-VL video", "vlm_wd_consensus_expanded_paired_cv", probability="WD_probability", threshold_column="fold_selected_threshold"),
        item("soft-label fine-tuning", "Qwen3-8B transcript", "llm_wd_soft_expanded_cv", probability="WD_probability"),
        item("soft-label fine-tuning", "Qwen3-VL video", "vlm_wd_soft_expanded_paired_cv", probability="WD_probability", threshold_column="fold_selected_threshold"),
    ]


def format_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" if i < 2 else "---:" for i in range(len(columns))) + "|"
    lines = [header, divider]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main(args):
    labels = pd.read_csv(args.labels, encoding="utf-8-sig", low_memory=False)
    labels["h1"] = pd.to_numeric(labels.WD_P_rater1, errors="raise")
    labels["h2"] = pd.to_numeric(labels.WD_P_rater2, errors="raise")
    ordinal_rows, binary_rows, coverage = [], [], []
    for spec in build_specs(args.results_root):
        if not spec["path"].exists():
            continue
        data = load_experiment(spec, labels)
        meta = {"experiment": spec["experiment"], "model": spec["modality"]}
        coverage.append({**meta, "N": len(data), "source": str(spec["path"])})
        if "ai_score" in data:
            ordinal_rows.append({**meta, **ordinal_metrics(data.dropna(subset=["ai_score"]), args.whole_number)})
        binary_rows.append({**meta, **binary_metrics(data.dropna(subset=["ai_binary"]))})

    ordinal = pd.DataFrame(ordinal_rows)
    binary = pd.DataFrame(binary_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    ordinal.to_csv(args.output / "ordinal_reliability_all_experiments.csv", index=False, encoding="utf-8-sig")
    binary.to_csv(args.output / "binary_reliability_all_experiments.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage).to_csv(args.output / "coverage.csv", index=False, encoding="utf-8-sig")

    ordinal_columns = [
        "experiment", "model", "N", "human_exact", "human_AC2_quadratic",
        "human_ICC_A1", "human_quadratic_kappa", "AI_human_exact_mean",
        "AI_human_AC2_quadratic_mean", "AI_human_ICC_A1_mean",
        "AI_human_quadratic_kappa_mean", "AI_human_RMSE_mean", "AI_mean_rating_RMSE", "three_rater_unanimous_exact",
        "three_rater_AC2_quadratic", "three_rater_ICC_A1",
    ]
    binary_columns = [
        "experiment", "model", "N", "human_exact", "human_AC1", "human_ICC_A1",
        "human_Cohen_kappa", "AI_human_exact_mean", "AI_human_AC1_mean",
        "AI_human_ICC_A1_mean", "AI_human_Cohen_kappa_mean",
        "three_rater_unanimous_exact", "three_rater_AC1", "three_rater_ICC_A1",
    ]
    report = [
        "# Expanded WD_P reliability across all completed experiments", "",
        ("All ordinal metrics, including ICC and RMSE, use half-up rounded integer AI scores (1-5)." if args.whole_number else "Ordinal AI scores are rounded half-up only for exact agreement, quadratic AC2, and quadratic kappa. ICC uses the original continuous AI score."), "AI-human values are the mean of AI-vs-human1 and AI-vs-human2 on identical rows.", "",
        "For binary data, unweighted Gwet agreement is conventionally AC1; it is the binary counterpart of weighted ordinal AC2. VLM fine-tuned decisions use the saved validation-selected fold threshold, while other binary outputs use 0.5 and regression outputs use score >=2.", "",
        "## Ordinal 1-5 reliability", "",
        *format_table(ordinal, ordinal_columns), "",
        "## Binary reliability", "",
        *format_table(binary, binary_columns), "",
        "## Interpretation warnings", "",
        "- Human baseline values vary when an experiment covers a different subset of rows.",
        "- Zero-shot and few-shot runs cover binary-consensus rows, so human binary agreement is 1 by construction.",
        "- Three-rater coefficients answer a different question from pairwise AI-human coefficients.",
        "- High AC2 under low-score skew can coexist with poor range, rank correlation, or high-severity performance.",
        "- Reliability coefficients do not replace MAE, RMSE, balanced accuracy, calibration, or score-distribution plots.",
    ]
    (args.output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps({
        "ordinal_experiments": len(ordinal), "binary_experiments": len(binary),
        "labels": str(args.labels.resolve()), "results_root": str(args.results_root.resolve())
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Ordinal experiments: {len(ordinal)}")
    print(f"Binary experiments: {len(binary)}")
    print(f"Report: {(args.output / 'report.md').resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("output/wd_multimodal_master_expanded/candidate_label_transcript_ready.csv"))
    parser.add_argument("--results-root", type=Path, default=Path("output/wd_expanded_completed_results_for_transfer"))
    parser.add_argument("--output", type=Path, default=Path("output/wd_expanded_all_reliability"))
    parser.add_argument("--whole-number", action="store_true", help="Round AI ordinal predictions to integer 1-5 before ICC and RMSE.")
    main(parser.parse_args())
