#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_MODELS = ["qwen", "minicpm", "molmo", "internvl35"]
DEFAULT_SEGMENTS = [4, 10, 16, 63, 65]


def parse_csv_list(value: str, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    return re.sub(r"\s+", " ", text)


CONTRADICTION_GROUPS = [
    ({"left"}, {"right"}),
    ({"up", "upward", "upwards"}, {"down", "downward", "downwards"}),
    ({"forward"}, {"backward", "backwards"}),
    ({"toward", "towards"}, {"away"}),
    ({"open", "opened"}, {"closed", "shut"}),
    ({"crossed"}, {"uncrossed"}),
    ({"raised", "raises", "raising"}, {"lowered", "lowers", "lowering"}),
]


def token_present(text: str, token: str) -> bool:
    return re.search(rf"\b{re.escape(token)}\b", text) is not None


def has_any(text: str, terms: set[str]) -> bool:
    return any(token_present(text, t) for t in terms)


def contradicts(a: str, b: str) -> bool:
    a = normalize_text(a)
    b = normalize_text(b)
    for left_terms, right_terms in CONTRADICTION_GROUPS:
        if (
            (has_any(a, left_terms) and has_any(b, right_terms))
            or
            (has_any(a, right_terms) and has_any(b, left_terms))
        ):
            return True
    return False


def temporal_compatible(a_start, a_end, b_start, b_end, tolerance):
    return a_start <= b_end + tolerance and b_start <= a_end + tolerance


def temporal_iou(a_start, a_end, b_start, b_end):
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return 0.0 if union <= 0 else intersection / union


def f1_score(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def load_reference(path: Path, eval_segments: List[int]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"segment_idx", "start_sec", "end_sec", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Reference missing columns: {sorted(missing)}")

    df = df[df["segment_idx"].isin(eval_segments)].copy()
    df["segment_idx"] = df["segment_idx"].astype(int)
    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="raise")
    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="raise")
    df["description"] = df["description"].astype(str).str.strip()

    if "certainty" not in df.columns:
        df["certainty"] = "unknown"

    df = df.reset_index(drop=True)
    df["reference_id"] = [
        f"ref_{seg}_{i:04d}"
        for i, seg in enumerate(df["segment_idx"])
    ]
    return df


def load_predictions(pred_root: Path, model_name: str, eval_segments: List[int]) -> pd.DataFrame:
    path = pred_root / model_name / "open_descriptions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path)
    required = {"segment_idx", "start_sec", "end_sec", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{model_name} predictions missing columns: {sorted(missing)}")

    df = df[df["segment_idx"].isin(eval_segments)].copy()
    df["segment_idx"] = df["segment_idx"].astype(int)
    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="raise")
    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="raise")
    df["description"] = df["description"].astype(str).str.strip()

    if "certainty" not in df.columns:
        df["certainty"] = "unknown"

    df = df.reset_index(drop=True)
    df["prediction_id"] = [
        f"{model_name}_pred_{seg}_{i:04d}"
        for i, seg in enumerate(df["segment_idx"])
    ]
    return df


def load_completeness(pred_root: Path, model_name: str, eval_segments: List[int]) -> pd.DataFrame:
    path = pred_root / model_name / "open_description_details.jsonl"
    latest: Dict[int, dict] = {}

    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            seg = int(row.get("segment_idx"))
            if seg in eval_segments:
                latest[seg] = row

    records = []
    for seg in eval_segments:
        row = latest.get(seg)
        if row is None:
            records.append({
                "model": model_name,
                "segment_idx": seg,
                "status": "missing",
                "n_observations_details": np.nan,
            })
        else:
            records.append({
                "model": model_name,
                "segment_idx": seg,
                "status": row.get("status", "unknown"),
                "n_observations_details": len(row.get("observations", [])),
            })
    return pd.DataFrame(records)


def build_pair_table(reference, predictions, embedder, temporal_tolerance):
    columns = [
        "segment_idx", "reference_id", "prediction_id",
        "reference_start", "reference_end",
        "prediction_start", "prediction_end",
        "reference_description", "prediction_description",
        "semantic_similarity", "time_iou", "contradiction",
    ]

    if reference.empty or predictions.empty:
        return pd.DataFrame(columns=columns)

    rows = []

    for segment_idx in sorted(
        set(reference["segment_idx"]).intersection(set(predictions["segment_idx"]))
    ):
        refs = reference[reference["segment_idx"] == segment_idx].reset_index(drop=True)
        preds = predictions[predictions["segment_idx"] == segment_idx].reset_index(drop=True)

        candidate_pairs = []
        for r_idx, ref in refs.iterrows():
            for p_idx, pred in preds.iterrows():
                if temporal_compatible(
                    float(ref["start_sec"]), float(ref["end_sec"]),
                    float(pred["start_sec"]), float(pred["end_sec"]),
                    temporal_tolerance,
                ):
                    candidate_pairs.append((r_idx, p_idx))

        if not candidate_pairs:
            continue

        unique_texts = []
        text_to_index = {}

        def add_text(text):
            text = str(text)
            if text not in text_to_index:
                text_to_index[text] = len(unique_texts)
                unique_texts.append(text)

        for r_idx, p_idx in candidate_pairs:
            add_text(refs.loc[r_idx, "description"])
            add_text(preds.loc[p_idx, "description"])

        embeddings = embedder.encode(
            unique_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        for r_idx, p_idx in candidate_pairs:
            ref = refs.loc[r_idx]
            pred = preds.loc[p_idx]
            r_text = str(ref["description"])
            p_text = str(pred["description"])

            similarity = float(
                np.dot(
                    embeddings[text_to_index[r_text]],
                    embeddings[text_to_index[p_text]],
                )
            )

            rows.append({
                "segment_idx": int(segment_idx),
                "reference_id": ref["reference_id"],
                "prediction_id": pred["prediction_id"],
                "reference_start": float(ref["start_sec"]),
                "reference_end": float(ref["end_sec"]),
                "prediction_start": float(pred["start_sec"]),
                "prediction_end": float(pred["end_sec"]),
                "reference_description": r_text,
                "prediction_description": p_text,
                "semantic_similarity": similarity,
                "time_iou": temporal_iou(
                    float(ref["start_sec"]), float(ref["end_sec"]),
                    float(pred["start_sec"]), float(pred["end_sec"]),
                ),
                "contradiction": bool(contradicts(r_text, p_text)),
            })

    return pd.DataFrame(rows, columns=columns)


def get_best_matches(reference, predictions, pairs, threshold):
    valid_pairs = pairs[
        (~pairs["contradiction"])
        & (pairs["semantic_similarity"] >= threshold)
    ].copy()

    reference_rows = []
    for _, ref in reference.iterrows():
        candidates = valid_pairs[
            valid_pairs["reference_id"] == ref["reference_id"]
        ].sort_values(["semantic_similarity", "time_iou"], ascending=False)

        if candidates.empty:
            reference_rows.append({
                "segment_idx": int(ref["segment_idx"]),
                "reference_id": ref["reference_id"],
                "reference_start": float(ref["start_sec"]),
                "reference_end": float(ref["end_sec"]),
                "reference_description": ref["description"],
                "covered": False,
                "best_prediction_id": "",
                "best_prediction_description": "",
                "best_similarity": np.nan,
                "best_time_iou": np.nan,
            })
        else:
            best = candidates.iloc[0]
            reference_rows.append({
                "segment_idx": int(ref["segment_idx"]),
                "reference_id": ref["reference_id"],
                "reference_start": float(ref["start_sec"]),
                "reference_end": float(ref["end_sec"]),
                "reference_description": ref["description"],
                "covered": True,
                "best_prediction_id": best["prediction_id"],
                "best_prediction_description": best["prediction_description"],
                "best_similarity": float(best["semantic_similarity"]),
                "best_time_iou": float(best["time_iou"]),
            })

    prediction_rows = []
    for _, pred in predictions.iterrows():
        candidates = valid_pairs[
            valid_pairs["prediction_id"] == pred["prediction_id"]
        ].sort_values(["semantic_similarity", "time_iou"], ascending=False)

        if candidates.empty:
            prediction_rows.append({
                "segment_idx": int(pred["segment_idx"]),
                "prediction_id": pred["prediction_id"],
                "prediction_start": float(pred["start_sec"]),
                "prediction_end": float(pred["end_sec"]),
                "prediction_description": pred["description"],
                "supported": False,
                "best_reference_id": "",
                "best_reference_description": "",
                "best_similarity": np.nan,
                "best_time_iou": np.nan,
            })
        else:
            best = candidates.iloc[0]
            prediction_rows.append({
                "segment_idx": int(pred["segment_idx"]),
                "prediction_id": pred["prediction_id"],
                "prediction_start": float(pred["start_sec"]),
                "prediction_end": float(pred["end_sec"]),
                "prediction_description": pred["description"],
                "supported": True,
                "best_reference_id": best["reference_id"],
                "best_reference_description": best["reference_description"],
                "best_similarity": float(best["semantic_similarity"]),
                "best_time_iou": float(best["time_iou"]),
            })

    return pd.DataFrame(reference_rows), pd.DataFrame(prediction_rows)


def greedy_one_to_one(pairs, threshold):
    valid = pairs[
        (~pairs["contradiction"])
        & (pairs["semantic_similarity"] >= threshold)
    ].copy()

    if valid.empty:
        return valid

    valid = valid.sort_values(
        ["semantic_similarity", "time_iou"],
        ascending=False,
    )

    used_refs = set()
    used_preds = set()
    selected = []

    for _, row in valid.iterrows():
        ref_id = row["reference_id"]
        pred_id = row["prediction_id"]

        if ref_id in used_refs or pred_id in used_preds:
            continue

        used_refs.add(ref_id)
        used_preds.add(pred_id)
        selected.append(row.to_dict())

    return pd.DataFrame(selected)


def compute_metrics(reference, predictions, pairs, threshold):
    ref_best, pred_best = get_best_matches(reference, predictions, pairs, threshold)

    n_ref = len(reference)
    n_pred = len(predictions)

    n_covered = int(ref_best["covered"].sum()) if n_ref else 0
    n_supported = int(pred_best["supported"].sum()) if n_pred else 0

    recall = n_covered / n_ref if n_ref else math.nan
    precision = n_supported / n_pred if n_pred else math.nan

    primary_f1 = (
        math.nan
        if math.isnan(precision) or math.isnan(recall)
        else f1_score(precision, recall)
    )

    strict = greedy_one_to_one(pairs, threshold)
    strict_tp = len(strict)

    strict_precision = strict_tp / n_pred if n_pred else math.nan
    strict_recall = strict_tp / n_ref if n_ref else math.nan

    strict_f1 = (
        math.nan
        if math.isnan(strict_precision) or math.isnan(strict_recall)
        else f1_score(strict_precision, strict_recall)
    )

    covered_rows = ref_best[ref_best["covered"]]

    metrics = {
        "threshold": threshold,
        "n_reference": n_ref,
        "n_predictions": n_pred,
        "n_reference_covered": n_covered,
        "n_predictions_supported": n_supported,
        "reference_recall": recall,
        "support_precision": precision,
        "coverage_support_f1": primary_f1,
        "mean_best_similarity_covered": (
            covered_rows["best_similarity"].mean()
            if not covered_rows.empty else math.nan
        ),
        "mean_time_iou_covered": (
            covered_rows["best_time_iou"].mean()
            if not covered_rows.empty else math.nan
        ),
        "strict_tp": strict_tp,
        "strict_precision": strict_precision,
        "strict_recall": strict_recall,
        "strict_f1": strict_f1,
    }

    return metrics, ref_best, pred_best, strict


def per_segment_metrics(reference, predictions, pairs, eval_segments, threshold):
    rows = []
    for seg in eval_segments:
        refs = reference[reference["segment_idx"] == seg].copy()
        preds = predictions[predictions["segment_idx"] == seg].copy()
        seg_pairs = pairs[pairs["segment_idx"] == seg].copy()

        metrics, _, _, _ = compute_metrics(refs, preds, seg_pairs, threshold)
        metrics["segment_idx"] = seg
        rows.append(metrics)

    return pd.DataFrame(rows)


def borderline_review(pairs, threshold, margin):
    if pairs.empty:
        return pairs.copy()

    return pairs[
        pairs["semantic_similarity"].between(
            threshold - margin,
            threshold + margin,
            inclusive="both",
        )
    ].sort_values(
        ["segment_idx", "semantic_similarity"],
        ascending=[True, False],
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--pred-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--segments", default=",".join(map(str, DEFAULT_SEGMENTS)))
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument(
        "--threshold-sweep",
        default="0.45,0.50,0.55,0.60,0.65",
    )
    parser.add_argument("--temporal-tolerance", type=float, default=2.0)
    parser.add_argument("--borderline-margin", type=float, default=0.05)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
    )

    args = parser.parse_args()

    models = parse_csv_list(args.models, str)
    eval_segments = parse_csv_list(args.segments, int)
    thresholds = parse_csv_list(args.threshold_sweep, float)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print("OPEN DESCRIPTION BENCHMARK EVALUATION")
    print("=" * 64)
    print("Reference:", args.reference)
    print("Prediction root:", args.pred_root)
    print("Segments:", eval_segments)
    print("Models:", models)
    print("Primary threshold:", args.threshold)
    print("Threshold sweep:", thresholds)
    print("Temporal tolerance:", args.temporal_tolerance)
    print("Embedding model:", args.embedding_model)
    print("")

    reference = load_reference(args.reference, eval_segments)
    print(f"Reference observations: {len(reference)}")

    embedder = SentenceTransformer(args.embedding_model)

    overall_rows = []
    sweep_rows = []
    per_segment_frames = []
    completeness_frames = []

    all_ref_best = []
    all_pred_best = []
    all_strict = []
    all_pairs = []
    all_borderline = []

    for model_name in models:
        print("")
        print(f"Evaluating {model_name}...")

        predictions = load_predictions(
            args.pred_root,
            model_name,
            eval_segments,
        )

        completeness = load_completeness(
            args.pred_root,
            model_name,
            eval_segments,
        )
        completeness_frames.append(completeness)

        bad_status = completeness[completeness["status"] != "ok"]
        if not bad_status.empty:
            raise RuntimeError(
                f"{model_name} has incomplete segments:\n"
                f"{bad_status.to_string(index=False)}"
            )

        pairs = build_pair_table(
            reference,
            predictions,
            embedder,
            args.temporal_tolerance,
        )
        pairs["model"] = model_name
        all_pairs.append(pairs)

        metrics, ref_best, pred_best, strict = compute_metrics(
            reference,
            predictions,
            pairs,
            args.threshold,
        )

        metrics["model"] = model_name
        overall_rows.append(metrics)

        ref_best["model"] = model_name
        pred_best["model"] = model_name
        all_ref_best.append(ref_best)
        all_pred_best.append(pred_best)

        if not strict.empty:
            strict["model"] = model_name
            all_strict.append(strict)

        segment_df = per_segment_metrics(
            reference,
            predictions,
            pairs,
            eval_segments,
            args.threshold,
        )
        segment_df["model"] = model_name
        per_segment_frames.append(segment_df)

        border = borderline_review(
            pairs,
            args.threshold,
            args.borderline_margin,
        )
        border["model"] = model_name
        all_borderline.append(border)

        for threshold in thresholds:
            sweep_metrics, _, _, _ = compute_metrics(
                reference,
                predictions,
                pairs,
                threshold,
            )
            sweep_metrics["model"] = model_name
            sweep_rows.append(sweep_metrics)

        print(
            f"  predictions={len(predictions)} | "
            f"recall={metrics['reference_recall']:.3f} | "
            f"precision={metrics['support_precision']:.3f} | "
            f"F1={metrics['coverage_support_f1']:.3f} | "
            f"strict_F1={metrics['strict_f1']:.3f}"
        )

    overall = pd.DataFrame(overall_rows).sort_values(
        "coverage_support_f1",
        ascending=False,
    )
    sweep = pd.DataFrame(sweep_rows).sort_values(
        ["threshold", "coverage_support_f1"],
        ascending=[True, False],
    )
    per_segment = pd.concat(per_segment_frames, ignore_index=True)
    completeness_all = pd.concat(completeness_frames, ignore_index=True)
    ref_best_all = pd.concat(all_ref_best, ignore_index=True)
    pred_best_all = pd.concat(all_pred_best, ignore_index=True)

    pairs_all = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    strict_all = pd.concat(all_strict, ignore_index=True) if all_strict else pd.DataFrame()
    borderline_all = (
        pd.concat(all_borderline, ignore_index=True)
        if all_borderline else pd.DataFrame()
    )

    overall.to_csv(args.output_dir / "overall_metrics.csv", index=False)
    sweep.to_csv(args.output_dir / "threshold_sweep.csv", index=False)
    per_segment.to_csv(args.output_dir / "per_segment_metrics.csv", index=False)
    completeness_all.to_csv(args.output_dir / "inference_completeness.csv", index=False)
    ref_best_all.to_csv(args.output_dir / "reference_best_matches.csv", index=False)
    pred_best_all.to_csv(args.output_dir / "prediction_best_matches.csv", index=False)
    pairs_all.to_csv(args.output_dir / "candidate_pairs.csv", index=False)
    strict_all.to_csv(args.output_dir / "strict_one_to_one_matches.csv", index=False)
    borderline_all.to_csv(args.output_dir / "borderline_manual_review.csv", index=False)

    print("")
    print("PRIMARY RESULTS")
    print("=" * 64)

    display_cols = [
        "model",
        "n_predictions",
        "reference_recall",
        "support_precision",
        "coverage_support_f1",
        "strict_f1",
    ]

    print(
        overall[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("")
    print("Saved evaluation outputs to:")
    print(args.output_dir)


if __name__ == "__main__":
    main()