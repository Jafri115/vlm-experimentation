#!/usr/bin/env python
"""
Compare behavior detection against the same 5-segment manual gold standard.

Primary intended comparison:
    Qwen3-VL-8B V6.3
    InternVL3-8B behavior benchmark

Inputs:
    gold CSV
    Qwen V6.3 flat detections CSV
    InternVL3-8B flat detections CSV

Matching:
    same segment
    same benchmark category
    interval overlap OR <= tolerance seconds apart
    one prediction can match only one gold event
    one gold event can match only one prediction

Non-benchmark expanded cues (gaze, legs, generic gestures, etc.) are ignored.
This prevents penalizing a model for detecting behaviors outside the manually
scored fair cue set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CATEGORY_MAP = {
    # Eye / tear-region behavior
    "hand_to_eye_region": "hand_eye",

    # Face-touch family
    "hand_to_cheek": "face_touch",
    "hand_to_chin": "face_touch",
    "hand_to_mouth": "face_touch",
    "hand_to_forehead": "face_touch",
    "hand_to_ear": "face_touch",
    "hand_to_face_unspecified": "face_touch",

    # Head down
    "head_pitch_down": "head_down",

    # Crying
    "crying_visible": "crying",
    "tear_visible": "crying",

    # Mouth tension
    "mouth_tension": "mouth_tension",
    "lips_pressed": "mouth_tension",

    # Motion / posture
    "movement_reduction": "movement_reduction",
    "shoulders_elevate": "shoulder",
    "shoulders_lower": "shoulder",
    "shoulder_lift_drop": "shoulder",
    "torso_backward": "torso_backward",

    # Head turning family
    "head_yaw_left": "head_turn",
    "head_yaw_right": "head_turn",
    "head_turn": "head_turn",
    "negative_head_shake": "head_turn",
    "abrupt_head_turn": "head_turn",

    # Explicit other
    "other_literal_visual_event": "other",
}

CATEGORY_ORDER = [
    "hand_eye",
    "face_touch",
    "head_down",
    "crying",
    "mouth_tension",
    "movement_reduction",
    "shoulder",
    "torso_backward",
    "head_turn",
    "other",
]


def interval_gap(a_start, a_end, b_start, b_end):
    if a_end >= b_start and b_end >= a_start:
        return 0.0

    if a_end < b_start:
        return b_start - a_end

    return a_start - b_end


def normalize(df, source_name):
    required = {
        "segment_idx",
        "behavior_code",
        "start_sec",
        "end_sec",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{source_name} missing columns: {sorted(missing)}"
        )

    out = df.copy()
    out["segment_idx"] = pd.to_numeric(
        out["segment_idx"], errors="raise"
    ).astype(int)
    out["start_sec"] = pd.to_numeric(
        out["start_sec"], errors="coerce"
    )
    out["end_sec"] = pd.to_numeric(
        out["end_sec"], errors="coerce"
    )
    out["behavior_code"] = out["behavior_code"].astype(str)

    out["category"] = out["behavior_code"].map(CATEGORY_MAP)

    # Fair benchmark: ignore expanded behaviors not represented in gold.
    out = out[out["category"].notna()].copy()

    return out


def greedy_match(gold, pred, tolerance):
    candidates = []

    for gi, g in gold.iterrows():
        for pi, p in pred.iterrows():
            if int(g["segment_idx"]) != int(p["segment_idx"]):
                continue
            if g["category"] != p["category"]:
                continue

            gap = interval_gap(
                float(g["start_sec"]),
                float(g["end_sec"]),
                float(p["start_sec"]),
                float(p["end_sec"]),
            )

            if gap <= tolerance:
                candidates.append(
                    (
                        gap,
                        gi,
                        pi,
                    )
                )

    candidates.sort(key=lambda x: x[0])

    used_g = set()
    used_p = set()
    matches = []

    for gap, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue

        used_g.add(gi)
        used_p.add(pi)
        matches.append(
            {
                "gold_index": gi,
                "pred_index": pi,
                "gap_sec": round(float(gap), 3),
            }
        )

    return matches, used_g, used_p


def score_model(model_name, gold, pred, tolerance):
    matches, used_g, used_p = greedy_match(
        gold,
        pred,
        tolerance,
    )

    match_df = pd.DataFrame(matches)

    rows = []

    for category in CATEGORY_ORDER:
        gold_cat = gold[gold["category"] == category]
        pred_cat = pred[pred["category"] == category]

        gold_ids = set(gold_cat.index)
        pred_ids = set(pred_cat.index)

        tp = sum(
            1
            for m in matches
            if m["gold_index"] in gold_ids
            and m["pred_index"] in pred_ids
        )

        fn = len(gold_ids - used_g)
        fp = len(pred_ids - used_p)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        rows.append(
            {
                "model": model_name,
                "category": category,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            }
        )

    total_tp = len(matches)
    total_fp = len(pred.index.difference(list(used_p)))
    total_fn = len(gold.index.difference(list(used_g)))

    p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0

    overall = {
        "model": model_name,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "precision": round(p, 6),
        "recall": round(r, 6),
        "f1": round(f, 6),
    }

    details = []

    for m in matches:
        g = gold.loc[m["gold_index"]]
        p_row = pred.loc[m["pred_index"]]
        details.append(
            {
                "model": model_name,
                "status": "TP",
                "segment_idx": int(g["segment_idx"]),
                "category": g["category"],
                "gold_behavior": g["behavior_code"],
                "gold_start": g["start_sec"],
                "gold_end": g["end_sec"],
                "pred_behavior": p_row["behavior_code"],
                "pred_start": p_row["start_sec"],
                "pred_end": p_row["end_sec"],
                "gap_sec": m["gap_sec"],
            }
        )

    for gi in gold.index.difference(list(used_g)):
        g = gold.loc[gi]
        details.append(
            {
                "model": model_name,
                "status": "FN",
                "segment_idx": int(g["segment_idx"]),
                "category": g["category"],
                "gold_behavior": g["behavior_code"],
                "gold_start": g["start_sec"],
                "gold_end": g["end_sec"],
                "pred_behavior": "",
                "pred_start": "",
                "pred_end": "",
                "gap_sec": "",
            }
        )

    for pi in pred.index.difference(list(used_p)):
        p_row = pred.loc[pi]
        details.append(
            {
                "model": model_name,
                "status": "FP",
                "segment_idx": int(p_row["segment_idx"]),
                "category": p_row["category"],
                "gold_behavior": "",
                "gold_start": "",
                "gold_end": "",
                "pred_behavior": p_row["behavior_code"],
                "pred_start": p_row["start_sec"],
                "pred_end": p_row["end_sec"],
                "gap_sec": "",
            }
        )

    return overall, rows, details


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold = normalize(
        pd.read_csv(args.gold),
        "gold",
    )

    model_specs = [
        (
            "Qwen3-VL-8B_V6.3",
            args.qwen,
        ),
        (
            "InternVL3-8B",
            args.internvl3,
        ),
    ]

    overall_rows = []
    category_rows = []
    detail_rows = []

    for model_name, path in model_specs:
        pred = normalize(
            pd.read_csv(path),
            model_name,
        )

        overall, categories, details = score_model(
            model_name,
            gold,
            pred,
            args.tolerance_sec,
        )

        overall_rows.append(overall)
        category_rows.extend(categories)
        detail_rows.extend(details)

    overall_df = pd.DataFrame(overall_rows)
    category_df = pd.DataFrame(category_rows)
    details_df = pd.DataFrame(detail_rows)

    overall_df.to_csv(
        out_dir / "behavior_metrics_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )
    category_df.to_csv(
        out_dir / "behavior_metrics_by_category.csv",
        index=False,
        encoding="utf-8-sig",
    )
    details_df.to_csv(
        out_dir / "behavior_match_details.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pivot = category_df.pivot(
        index="category",
        columns="model",
        values="f1",
    ).reindex(CATEGORY_ORDER)

    pivot.to_csv(
        out_dir / "behavior_f1_side_by_side.csv",
        encoding="utf-8-sig",
    )

    print("")
    print("BEHAVIOR DETECTION BENCHMARK")
    print("=" * 70)
    print(f"Gold events: {len(gold)}")
    print(f"Tolerance: ±{args.tolerance_sec}s")
    print("")
    print("OVERALL")
    print(overall_df.to_string(index=False))
    print("")
    print("BY CATEGORY")
    print(category_df.to_string(index=False))
    print("")
    print("F1 SIDE BY SIDE")
    print(pivot.to_string())
    print("")
    print(f"Outputs: {out_dir}")


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--gold",
        default=(
            "./output/visual_cue_benchmark_v6_2/"
            "visual_cue_gold_standard_5_segments_fair_benchmark.csv"
        ),
    )

    p.add_argument(
        "--qwen",
        default=(
            "./output/qwen3vl_visual_experiment_v6_3/"
            "visual_experiment_v6_3_detections.csv"
        ),
    )

    p.add_argument(
        "--internvl3",
        default=(
            "./output/internvl3_8b_behavior_benchmark/"
            "internvl3_8b_behavior_detections.csv"
        ),
    )

    p.add_argument(
        "--output-dir",
        default=(
            "./output/behavior_model_comparison_qwen8_internvl3_8b"
        ),
    )

    p.add_argument(
        "--tolerance-sec",
        type=float,
        default=2.0,
    )

    return p


if __name__ == "__main__":
    main(build_parser().parse_args())