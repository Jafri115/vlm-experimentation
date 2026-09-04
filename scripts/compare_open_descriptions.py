#!/usr/bin/env python
"""
Compare free-text VLM observations against the human-verified ChatGPT reference.

Why two metric families?
------------------------
1) Coverage/support metrics (PRIMARY):
   - reference recall: each reference observation is covered if ANY temporally
     compatible model observation is semantically similar enough.
   - prediction support precision: each model observation is supported if ANY
     temporally compatible reference observation is semantically similar enough.
   This is robust to different segmentation granularity (e.g. a 60s reference
   state vs four 15s model observations).

2) Strict greedy one-to-one metrics (SENSITIVITY ANALYSIS):
   penalizes repeated/restated observations and gives a more conservative score.

Semantic matching is performed only AFTER temporal filtering.
The evaluator never feeds the reference back into a VLM.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"

ANTONYM_PAIRS = [
    ({"left"}, {"right"}),
    ({"up", "upward", "raises", "raised"}, {"down", "downward", "lowers", "lowered"}),
    ({"toward", "towards"}, {"away"}),
    ({"forward"}, {"backward", "back", "reclined"}),
    ({"open", "opens"}, {"closed", "closes"}),
    ({"crossed", "crosses"}, {"uncrossed", "uncrosses"}),
]


def tokens(text):
    return set(re.findall(r"[a-z]+", str(text).lower()))


def obvious_contradiction(a, b):
    ta, tb = tokens(a), tokens(b)
    for left, right in ANTONYM_PAIRS:
        if (ta & left and tb & right) or (ta & right and tb & left):
            return True
    return False


def interval_gap(a0, a1, b0, b1):
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0.0


def interval_iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 1.0 if abs(a0 - b0) < 1e-9 else 0.0
    return inter / union


def validate_df(df, name):
    required = {"segment_idx", "start_sec", "end_sec", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")

    out = df.copy()
    out["segment_idx"] = pd.to_numeric(out["segment_idx"], errors="raise").astype(int)
    out["start_sec"] = pd.to_numeric(out["start_sec"], errors="raise")
    out["end_sec"] = pd.to_numeric(out["end_sec"], errors="raise")
    out["description"] = out["description"].fillna("").astype(str).str.strip()
    out = out[
        (out["description"] != "")
        & (out["end_sec"] >= out["start_sec"])
    ].copy()
    return out.reset_index(drop=True)


def cosine_matrix(model, ref_texts, pred_texts):
    if not ref_texts or not pred_texts:
        return np.zeros((len(ref_texts), len(pred_texts)), dtype=np.float32)

    all_texts = list(ref_texts) + list(pred_texts)
    emb = model.encode(
        all_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    r = emb[:len(ref_texts)]
    p = emb[len(ref_texts):]
    return r @ p.T


def candidate_table_for_segment(model, ref_seg, pred_seg, tolerance):
    sims = cosine_matrix(
        model,
        ref_seg["description"].tolist(),
        pred_seg["description"].tolist(),
    )
    rows = []

    for ri, r in ref_seg.iterrows():
        for pi, p in pred_seg.iterrows():
            gap = interval_gap(
                float(r.start_sec), float(r.end_sec),
                float(p.start_sec), float(p.end_sec),
            )
            if gap > tolerance:
                continue

            contrad = obvious_contradiction(r.description, p.description)
            rows.append({
                "ref_local": int(ri),
                "pred_local": int(pi),
                "similarity": float(sims[ri, pi]),
                "gap_sec": float(gap),
                "time_iou": float(interval_iou(
                    float(r.start_sec), float(r.end_sec),
                    float(p.start_sec), float(p.end_sec),
                )),
                "contradiction_guard": int(contrad),
            })

    return pd.DataFrame(rows)


def best_matches(model, ref, pred, tolerance):
    ref_best_rows = []
    pred_best_rows = []
    pair_rows = []

    for seg in sorted(set(ref.segment_idx) | set(pred.segment_idx)):
        rseg = ref[ref.segment_idx == seg].reset_index()
        pseg = pred[pred.segment_idx == seg].reset_index()

        if rseg.empty:
            for _, p in pseg.iterrows():
                pred_best_rows.append({
                    "segment_idx": seg,
                    "pred_index": int(p["index"]),
                    "pred_start": p.start_sec,
                    "pred_end": p.end_sec,
                    "pred_description": p.description,
                    "best_ref_index": "",
                    "best_similarity": np.nan,
                    "gap_sec": np.nan,
                    "time_iou": np.nan,
                    "contradiction_guard": 0,
                    "best_ref_description": "",
                })
            continue

        if pseg.empty:
            for _, r in rseg.iterrows():
                ref_best_rows.append({
                    "segment_idx": seg,
                    "ref_index": int(r["index"]),
                    "ref_start": r.start_sec,
                    "ref_end": r.end_sec,
                    "ref_description": r.description,
                    "best_pred_index": "",
                    "best_similarity": np.nan,
                    "gap_sec": np.nan,
                    "time_iou": np.nan,
                    "contradiction_guard": 0,
                    "best_pred_description": "",
                })
            continue

        candidates = candidate_table_for_segment(
            model,
            rseg,
            pseg,
            tolerance,
        )

        for _, r in rseg.iterrows():
            c = candidates[
                (candidates.ref_local == int(r.name))
                & (candidates.contradiction_guard == 0)
            ]
            if c.empty:
                ref_best_rows.append({
                    "segment_idx": seg,
                    "ref_index": int(r["index"]),
                    "ref_start": r.start_sec,
                    "ref_end": r.end_sec,
                    "ref_description": r.description,
                    "best_pred_index": "",
                    "best_similarity": np.nan,
                    "gap_sec": np.nan,
                    "time_iou": np.nan,
                    "contradiction_guard": 0,
                    "best_pred_description": "",
                })
            else:
                best = c.sort_values(
                    ["similarity", "time_iou"],
                    ascending=False,
                ).iloc[0]
                p = pseg.loc[int(best.pred_local)]
                ref_best_rows.append({
                    "segment_idx": seg,
                    "ref_index": int(r["index"]),
                    "ref_start": r.start_sec,
                    "ref_end": r.end_sec,
                    "ref_description": r.description,
                    "best_pred_index": int(p["index"]),
                    "best_similarity": best.similarity,
                    "gap_sec": best.gap_sec,
                    "time_iou": best.time_iou,
                    "contradiction_guard": int(best.contradiction_guard),
                    "best_pred_description": p.description,
                })

        for _, p in pseg.iterrows():
            c = candidates[
                (candidates.pred_local == int(p.name))
                & (candidates.contradiction_guard == 0)
            ]
            if c.empty:
                pred_best_rows.append({
                    "segment_idx": seg,
                    "pred_index": int(p["index"]),
                    "pred_start": p.start_sec,
                    "pred_end": p.end_sec,
                    "pred_description": p.description,
                    "best_ref_index": "",
                    "best_similarity": np.nan,
                    "gap_sec": np.nan,
                    "time_iou": np.nan,
                    "contradiction_guard": 0,
                    "best_ref_description": "",
                })
            else:
                best = c.sort_values(
                    ["similarity", "time_iou"],
                    ascending=False,
                ).iloc[0]
                r = rseg.loc[int(best.ref_local)]
                pred_best_rows.append({
                    "segment_idx": seg,
                    "pred_index": int(p["index"]),
                    "pred_start": p.start_sec,
                    "pred_end": p.end_sec,
                    "pred_description": p.description,
                    "best_ref_index": int(r["index"]),
                    "best_similarity": best.similarity,
                    "gap_sec": best.gap_sec,
                    "time_iou": best.time_iou,
                    "contradiction_guard": int(best.contradiction_guard),
                    "best_ref_description": r.description,
                })

        for _, c in candidates.iterrows():
            r = rseg.loc[int(c.ref_local)]
            p = pseg.loc[int(c.pred_local)]
            pair_rows.append({
                "segment_idx": seg,
                "ref_index": int(r["index"]),
                "pred_index": int(p["index"]),
                "ref_start": r.start_sec,
                "ref_end": r.end_sec,
                "pred_start": p.start_sec,
                "pred_end": p.end_sec,
                "similarity": c.similarity,
                "gap_sec": c.gap_sec,
                "time_iou": c.time_iou,
                "contradiction_guard": int(c.contradiction_guard),
                "ref_description": r.description,
                "pred_description": p.description,
            })

    return (
        pd.DataFrame(ref_best_rows),
        pd.DataFrame(pred_best_rows),
        pd.DataFrame(pair_rows),
    )


def coverage_metrics(ref_best, pred_best, threshold):
    if len(ref_best):
        ref_hit = (
            pd.to_numeric(ref_best.best_similarity, errors="coerce") >= threshold
        )
        recall = float(ref_hit.mean())
        matched_ref = int(ref_hit.sum())
    else:
        recall, matched_ref = 0.0, 0

    if len(pred_best):
        pred_hit = (
            pd.to_numeric(pred_best.best_similarity, errors="coerce") >= threshold
        )
        precision = float(pred_hit.mean())
        supported_pred = int(pred_hit.sum())
    else:
        precision, supported_pred = 0.0, 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    matched_sims = pd.to_numeric(
        ref_best.loc[
            pd.to_numeric(ref_best.best_similarity, errors="coerce") >= threshold,
            "best_similarity",
        ],
        errors="coerce",
    )

    matched_iou = pd.to_numeric(
        ref_best.loc[
            pd.to_numeric(ref_best.best_similarity, errors="coerce") >= threshold,
            "time_iou",
        ],
        errors="coerce",
    )

    return {
        "n_reference": len(ref_best),
        "n_predictions": len(pred_best),
        "matched_reference": matched_ref,
        "supported_predictions": supported_pred,
        "precision_support": precision,
        "recall_coverage": recall,
        "f1_coverage_support": f1,
        "mean_similarity_matched": (
            float(matched_sims.mean()) if len(matched_sims) else np.nan
        ),
        "mean_time_iou_matched": (
            float(matched_iou.mean()) if len(matched_iou) else np.nan
        ),
    }


def greedy_one_to_one(pairs, n_ref, n_pred, threshold):
    if pairs.empty:
        return {
            "strict_tp": 0,
            "strict_fp": n_pred,
            "strict_fn": n_ref,
            "strict_precision": 0.0,
            "strict_recall": 0.0,
            "strict_f1": 0.0,
        }

    c = pairs[
        (pairs.similarity >= threshold)
        & (pairs.contradiction_guard == 0)
    ].sort_values(
        ["similarity", "time_iou"],
        ascending=False,
    )

    used_r, used_p = set(), set()
    tp = 0
    for _, row in c.iterrows():
        r = int(row.ref_index)
        p = int(row.pred_index)
        if r in used_r or p in used_p:
            continue
        used_r.add(r)
        used_p.add(p)
        tp += 1

    fp = n_pred - tp
    fn = n_ref - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "strict_tp": tp,
        "strict_fp": fp,
        "strict_fn": fn,
        "strict_precision": precision,
        "strict_recall": recall,
        "strict_f1": f1,
    }


def evaluate_one(
    embedder,
    model_name,
    ref,
    pred,
    tolerance,
    primary_threshold,
    threshold_sweep,
    out_dir,
):
    ref_best, pred_best, pairs = best_matches(
        embedder,
        ref,
        pred,
        tolerance,
    )

    ref_best.insert(0, "model", model_name)
    pred_best.insert(0, "model", model_name)
    pairs.insert(0, "model", model_name)

    ref_best.to_csv(
        out_dir / f"{model_name}_reference_best_matches.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pred_best.to_csv(
        out_dir / f"{model_name}_prediction_support.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pairs.to_csv(
        out_dir / f"{model_name}_candidate_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = coverage_metrics(
        ref_best,
        pred_best,
        primary_threshold,
    )
    metrics.update(
        greedy_one_to_one(
            pairs,
            n_ref=len(ref),
            n_pred=len(pred),
            threshold=primary_threshold,
        )
    )
    metrics.update({
        "model": model_name,
        "semantic_threshold": primary_threshold,
        "tolerance_sec": tolerance,
    })

    sweep_rows = []
    for th in threshold_sweep:
        row = coverage_metrics(ref_best, pred_best, th)
        row.update(
            greedy_one_to_one(
                pairs,
                n_ref=len(ref),
                n_pred=len(pred),
                threshold=th,
            )
        )
        row.update({
            "model": model_name,
            "semantic_threshold": th,
            "tolerance_sec": tolerance,
        })
        sweep_rows.append(row)

    return metrics, sweep_rows, ref_best, pred_best


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = validate_df(
        pd.read_csv(args.reference),
        "reference",
    )

    specs = [
        ("qwen", args.qwen),
        ("minicpm", args.minicpm),
        ("molmo", args.molmo),
        ("internvl35", args.internvl35),
    ]
    specs = [(name, path) for name, path in specs if path and Path(path).exists()]
    if not specs:
        raise ValueError("No prediction CSVs found.")

    print(f"Loading semantic evaluator: {args.embedding_model}")
    embedder = SentenceTransformer(
        args.embedding_model,
        device=args.device,
    )

    thresholds = [
        float(x.strip())
        for x in args.threshold_sweep.split(",")
        if x.strip()
    ]

    overall = []
    sweep = []
    all_ref_best = []
    all_pred_best = []

    for model_name, path in specs:
        pred = validate_df(
            pd.read_csv(path),
            model_name,
        )

        # Compare only segments represented in the frozen reference.
        pred = pred[pred.segment_idx.isin(set(ref.segment_idx))].copy()

        print(
            f"{model_name}: reference={len(ref)}, predictions={len(pred)}",
            flush=True,
        )

        metrics, sweep_rows, ref_best, pred_best = evaluate_one(
            embedder=embedder,
            model_name=model_name,
            ref=ref,
            pred=pred,
            tolerance=args.tolerance_sec,
            primary_threshold=args.semantic_threshold,
            threshold_sweep=thresholds,
            out_dir=out_dir,
        )
        overall.append(metrics)
        sweep.extend(sweep_rows)
        all_ref_best.append(ref_best)
        all_pred_best.append(pred_best)

    overall_df = pd.DataFrame(overall).sort_values(
        "f1_coverage_support",
        ascending=False,
    )
    sweep_df = pd.DataFrame(sweep)

    overall_df.to_csv(
        out_dir / "open_description_metrics_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sweep_df.to_csv(
        out_dir / "open_description_threshold_sweep.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ref_best_all = pd.concat(all_ref_best, ignore_index=True)
    pred_best_all = pd.concat(all_pred_best, ignore_index=True)

    ref_best_all.to_csv(
        out_dir / "all_models_reference_coverage_details.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pred_best_all.to_csv(
        out_dir / "all_models_prediction_support_details.csv",
        index=False,
        encoding="utf-8-sig",
    )

    low = args.semantic_threshold - 0.08
    high = args.semantic_threshold + 0.08
    borderline = ref_best_all[
        pd.to_numeric(ref_best_all.best_similarity, errors="coerce").between(
            low, high, inclusive="both"
        )
    ].copy()
    borderline.to_csv(
        out_dir / "borderline_matches_for_manual_review.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("")
    print("OPEN DESCRIPTION BENCHMARK")
    print("=" * 100)
    cols = [
        "model",
        "n_reference",
        "n_predictions",
        "precision_support",
        "recall_coverage",
        "f1_coverage_support",
        "mean_similarity_matched",
        "mean_time_iou_matched",
        "strict_precision",
        "strict_recall",
        "strict_f1",
    ]
    print(overall_df[cols].to_string(index=False))
    print("")
    print(f"Primary semantic threshold: {args.semantic_threshold}")
    print(f"Temporal tolerance: ±{args.tolerance_sec}s")
    print(f"Outputs: {out_dir}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--reference",
        default="./reference_human_verified_chatgpt.csv",
    )
    p.add_argument(
        "--qwen",
        default="./output/open_description_benchmark/qwen/open_descriptions.csv",
    )
    p.add_argument(
        "--minicpm",
        default="./output/open_description_benchmark/minicpm/open_descriptions.csv",
    )
    p.add_argument(
        "--molmo",
        default="./output/open_description_benchmark/molmo/open_descriptions.csv",
    )
    p.add_argument(
        "--internvl35",
        default="./output/open_description_benchmark/internvl35/open_descriptions.csv",
    )
    p.add_argument(
        "--output-dir",
        default="./output/open_description_benchmark/evaluation",
    )
    p.add_argument(
        "--embedding-model",
        default=DEFAULT_MODEL,
    )
    p.add_argument(
        "--device",
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    p.add_argument("--tolerance-sec", type=float, default=2.0)
    p.add_argument("--semantic-threshold", type=float, default=0.55)
    p.add_argument(
        "--threshold-sweep",
        default="0.45,0.50,0.55,0.60,0.65",
    )
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
