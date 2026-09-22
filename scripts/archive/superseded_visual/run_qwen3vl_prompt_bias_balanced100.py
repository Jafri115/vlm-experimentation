#!/usr/bin/env python
"""
Qwen3-VL zero-shot prompt-bias experiment
=========================================

Research question
-----------------
The original balanced-100 direct zero-shot experiment collapsed to
NO_RUPTURE. Does changing only the decision framing change that behavior?

Conditions
----------
1) original
   Reproduces the conservative original decision rule.

2) rupture_sensitive
   Same visual definition, but explicitly allows subtle/repeated/sustained
   visible evidence to support RUPTURE. It does not require a dramatic cue.

3) balanced_prior
   Same visual definition, plus the diagnostic-set fact that the evaluation
   batch is approximately balanced. The model is told not to use an
   assumed real-world rarity of rupture as a class prior.

IMPORTANT
---------
- Same Qwen3-VL model for all conditions.
- Same 60-s segment.
- Same sampled frames.
- Same video token budget.
- Same deterministic decoding.
- Labels are loaded ONLY AFTER all requested inference is complete.
- The "approximately balanced" statement is global study-design information,
  not a per-segment label.

This script intentionally reuses the stable infrastructure from:
    scripts/run_qwen3vl_visual_baseline_100.py

Put this script in the same scripts/ directory.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from qwen_vl_utils import process_vision_info

import run_qwen3vl_visual_baseline_100 as base


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CONDITIONS = ("original", "rupture_sensitive", "balanced_prior")


# ---------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------

def condition_instruction(condition: str) -> str:
    if condition == "original":
        return """
DECISION FRAMING — ORIGINAL / CONSERVATIVE
- A visible action is not automatically a rupture.
- Require sufficiently clear visually supported evidence.
- If the available visual evidence is ambiguous or insufficient,
  choose NO_RUPTURE.
""".strip()

    if condition == "rupture_sensitive":
        return """
DECISION FRAMING — RUPTURE-SENSITIVE
- A rupture does NOT need to look dramatic.
- Subtle visual evidence may support RUPTURE when it forms a meaningful
  repeated, sustained, or temporally coherent pattern across the segment.
- Do not reject a possible rupture merely because each individual cue is mild.
- Ordinary isolated behaviors still do NOT count as rupture by themselves.
- If there is a credible visible pattern of movement away or movement against,
  choose RUPTURE even when the pattern is relatively subtle.
- If there is no credible visible pattern, choose NO_RUPTURE.
""".strip()

    if condition == "balanced_prior":
        return """
DECISION FRAMING — BALANCED DIAGNOSTIC PRIOR
- This is a controlled diagnostic evaluation batch containing approximately
  equal numbers of RUPTURE and NO_RUPTURE examples.
- This is global study-design information only; it gives no information about
  the current segment.
- Treat RUPTURE and NO_RUPTURE as approximately equally plausible BEFORE
  looking at the video.
- Do not default to NO_RUPTURE because you assume rupture is rare.
- The final decision must still be based only on directly visible evidence.
- Ordinary isolated or ambiguous behaviors are not automatically rupture.
""".strip()

    raise ValueError(f"Unknown condition: {condition}")


def build_messages(
    frames,
    sample_fps: float,
    role_description: str,
    total_pixels: int,
    condition: str,
):
    framing = condition_instruction(condition)

    # Keep the visual definition from the original baseline unchanged.
    prompt = f"""
ROLE LAYOUT:
{role_description}

You are seeing ONE chronological video represented by sampled frames from the
complete psychotherapy segment. The timestamp printed on each frame is the
time within the segment.

{base.VISUAL_DEFINITION}

{framing}

Make ONE binary decision for the complete segment.

LABEL FIREWALL:
Human ratings are not available to you and must not be inferred from filenames,
segment identifiers, patient IDs, session IDs, or metadata. Base the decision
only on the visible frames, the visual definition, and the condition-specific
decision framing above.

Return JSON only:

{{
  "label": "RUPTURE",
  "confidence": 0.00,
  "visual_type": "withdrawal|confrontation|mixed|unclear|none",
  "actor": "patient|therapist|both|unclear|none",
  "evidence_windows": [
    {{
      "start_sec": 0.0,
      "end_sec": 10.0,
      "visual_evidence": "brief literal description of directly visible evidence"
    }}
  ],
  "counterevidence": [
    "brief literal visual observation arguing against rupture"
  ],
  "reason": "short visual-only explanation"
}}

Rules:
- label must be exactly RUPTURE or NO_RUPTURE.
- Do not infer spoken words, speech content, or tone of voice.
- Evidence must be directly visible.
- Prefer literal descriptions of gaze direction, head orientation, posture,
  hand/arm movement, facial action, body movement, and visible mouth movement.
- Do not infer emotion, intention, motivation, resistance, alliance quality,
  criticism, disagreement, or avoidance unless it is the final requested
  rupture classification.
- Judge the complete temporal pattern, not one isolated frame.
""".strip()

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                    "sample_fps": float(sample_fps),
                    "total_pixels": int(total_pixels),
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]


# ---------------------------------------------------------------------
# Qwen inference
# ---------------------------------------------------------------------

class PromptBiasClassifier(base.DirectVisualClassifier):
    def predict_condition(
        self,
        frames,
        sample_fps,
        role_description,
        total_pixels,
        max_new_tokens,
        condition,
    ):
        messages = build_messages(
            frames=frames,
            sample_fps=sample_fps,
            role_description=role_description,
            total_pixels=total_pixels,
            condition=condition,
        )

        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        if video_inputs is not None:
            videos, video_metadatas = zip(*video_inputs)
            videos = list(videos)
            video_metadatas = list(video_metadatas)
        else:
            videos = None
            video_metadatas = None

        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=videos,
            video_metadata=video_metadatas,
            padding=True,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
        inputs = inputs.to(self.device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        started = time.time()

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        inference_sec = time.time() - started

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        raw = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        result = base.validate_result(base.extract_json(raw))

        peak_allocated_gb = None
        peak_reserved_gb = None
        if torch.cuda.is_available():
            peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

        # Explicitly release multimodal processor tensors before next condition.
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (
            result,
            raw,
            inference_sec,
            peak_allocated_gb,
            peak_reserved_gb,
        )


# ---------------------------------------------------------------------
# Resume / save helpers
# ---------------------------------------------------------------------

def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def completed_pairs(df: pd.DataFrame) -> set[tuple[int, str]]:
    if df.empty:
        return set()

    ok = df.copy()
    if "status" in ok.columns:
        ok = ok[ok["status"].astype(str) == "ok"]

    pairs = set()
    for row in ok.itertuples(index=False):
        try:
            pairs.add((int(row.segment_idx), str(row.condition)))
        except Exception:
            pass
    return pairs


def upsert_row(df: pd.DataFrame, row: dict, path: Path) -> pd.DataFrame:
    if not df.empty and {"segment_idx", "condition"}.issubset(df.columns):
        keep = ~(
            (pd.to_numeric(df["segment_idx"], errors="coerce") == int(row["segment_idx"]))
            & (df["condition"].astype(str) == str(row["condition"]))
        )
        df = df.loc[keep].copy()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df["segment_idx"] = pd.to_numeric(df["segment_idx"], errors="coerce")

    df = df.sort_values(["segment_idx", "condition"])
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def append_detail(path: Path, payload: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def binary_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    n = len(y_true)
    accuracy = (tp + tn) / n if n else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    balanced_accuracy = (
        (recall + specificity) / 2
        if np.isfinite(recall) and np.isfinite(specificity)
        else np.nan
    )

    return {
        "N": n,
        "positives": int(np.sum(y_true == 1)),
        "negatives": int(np.sum(y_true == 0)),
        "predicted_rupture_n": int(np.sum(y_pred == 1)),
        "predicted_rupture_fraction": float(np.mean(y_pred)) if n else np.nan,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def evaluate(predictions_csv: Path, labels_csv: Path, output_dir: Path):
    pred = pd.read_csv(predictions_csv)
    pred = pred[pred["status"].astype(str) == "ok"].copy()
    pred["segment_idx"] = pd.to_numeric(pred["segment_idx"], errors="raise").astype(int)
    pred["pred_binary"] = (pred["label"].astype(str).str.upper() == "RUPTURE").astype(int)

    labels = pd.read_csv(labels_csv)

    if "eval_id" in labels.columns:
        labels["segment_idx"] = pd.to_numeric(labels["eval_id"], errors="raise").astype(int)
    elif "segment_idx" in labels.columns:
        labels["segment_idx"] = pd.to_numeric(labels["segment_idx"], errors="raise").astype(int)
    else:
        raise ValueError(
            "Labels CSV must contain eval_id or segment_idx for matching."
        )

    required = {"human_binary", "WD_P_mean", "CF_P_mean"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(
            f"Labels CSV missing columns needed for evaluation: {sorted(missing)}"
        )

    labels["legacy_any_3rs_marker"] = pd.to_numeric(
        labels["human_binary"], errors="coerce"
    )
    labels["patient_only_any_marker"] = (
        (pd.to_numeric(labels["WD_P_mean"], errors="coerce") > 1.0)
        | (pd.to_numeric(labels["CF_P_mean"], errors="coerce") > 1.0)
    ).astype(float)

    merged = pred.merge(
        labels[
            [
                "segment_idx",
                "human_label",
                "legacy_any_3rs_marker",
                "patient_only_any_marker",
                "WD_P_mean",
                "CF_P_mean",
            ]
        ],
        on="segment_idx",
        how="left",
        validate="many_to_one",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(
        output_dir / "predictions_with_labels.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_rows = []
    targets = ("legacy_any_3rs_marker", "patient_only_any_marker")

    for condition in CONDITIONS:
        for target in targets:
            sub = merged[
                (merged["condition"] == condition)
                & merged[target].notna()
            ].copy()

            m = binary_metrics(
                sub[target].astype(int).to_numpy(),
                sub["pred_binary"].astype(int).to_numpy(),
            )
            metric_rows.append(
                {
                    "condition": condition,
                    "target": target,
                    **m,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(
        output_dir / "prompt_condition_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Side-by-side per segment.
    side = (
        merged.pivot_table(
            index=[
                "segment_idx",
                "human_label",
                "legacy_any_3rs_marker",
                "patient_only_any_marker",
                "WD_P_mean",
                "CF_P_mean",
            ],
            columns="condition",
            values="label",
            aggfunc="first",
        )
        .reset_index()
    )
    side.columns.name = None
    side.to_csv(
        output_dir / "side_by_side_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Quantify how each framing moves predictions relative to original.
    change_rows = []
    if "original" in side.columns:
        for variant in ("rupture_sensitive", "balanced_prior"):
            if variant not in side.columns:
                continue

            valid = side[["original", variant]].notna().all(axis=1)
            sub = side.loc[valid].copy()

            o = sub["original"].astype(str)
            v = sub[variant].astype(str)

            change_rows.append(
                {
                    "variant": variant,
                    "N_common": len(sub),
                    "changed_total": int(np.sum(o != v)),
                    "NO_to_RUPTURE": int(
                        np.sum((o == "NO_RUPTURE") & (v == "RUPTURE"))
                    ),
                    "RUPTURE_to_NO": int(
                        np.sum((o == "RUPTURE") & (v == "NO_RUPTURE"))
                    ),
                    "same_NO": int(
                        np.sum((o == "NO_RUPTURE") & (v == "NO_RUPTURE"))
                    ),
                    "same_RUPTURE": int(
                        np.sum((o == "RUPTURE") & (v == "RUPTURE"))
                    ),
                }
            )

    changes = pd.DataFrame(change_rows)
    changes.to_csv(
        output_dir / "prediction_changes_vs_original.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Compact headline using the exact original balanced-100 target.
    headline = metrics[
        metrics["target"] == "legacy_any_3rs_marker"
    ][
        [
            "condition",
            "N",
            "predicted_rupture_n",
            "predicted_rupture_fraction",
            "TP",
            "TN",
            "FP",
            "FN",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
        ]
    ].copy()

    headline.to_csv(
        output_dir / "headline_prompt_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nPROMPT-BIAS COMPARISON — ORIGINAL BALANCED-100 TARGET")
    print("=" * 78)
    print(headline.to_string(index=False))

    print("\nPREDICTION CHANGES VS ORIGINAL")
    print("=" * 78)
    print(changes.to_string(index=False) if not changes.empty else "Not enough common predictions.")

    print(f"\nEvaluation outputs: {output_dir}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_csv = output_dir / "prompt_bias_predictions.csv"
    details_jsonl = output_dir / "prompt_bias_details.jsonl"
    eval_dir = output_dir / "evaluation"

    segments = pd.read_csv(args.segments_csv)

    required = {"segment_idx", "segment_path"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(
            f"segments CSV missing columns: {sorted(missing)}"
        )

    # Keep labels fully outside inference.
    forbidden = {
        "human_label",
        "human_binary",
        "WD_P",
        "WD_T",
        "CF_P",
        "CF_T",
        "WD_P_mean",
        "WD_T_mean",
        "CF_P_mean",
        "CF_T_mean",
    }
    leaked = forbidden.intersection(segments.columns)
    if leaked:
        raise ValueError(
            "LABEL FIREWALL: use the label-free segments_manifest.csv. "
            f"Found label columns: {sorted(leaked)}"
        )

    segments["segment_idx"] = pd.to_numeric(
        segments["segment_idx"], errors="raise"
    ).astype(int)

    requested = base.parse_segment_indices(args.segment_indices)
    if requested is not None:
        segments = segments[segments["segment_idx"].isin(requested)].copy()

    if args.max_segments is not None:
        segments = segments.head(args.max_segments).copy()

    requested_conditions = [
        x.strip() for x in args.conditions.split(",") if x.strip()
    ]
    invalid = [x for x in requested_conditions if x not in CONDITIONS]
    if invalid:
        raise ValueError(
            f"Unknown conditions {invalid}; allowed={CONDITIONS}"
        )

    pred_df = load_predictions(predictions_csv)
    done = completed_pairs(pred_df)

    total_pixels = int(args.video_token_budget * 32 * 32)

    print("\nQWEN3-VL ZERO-SHOT PROMPT-BIAS EXPERIMENT")
    print("=" * 72)
    print(f"Model: {args.model_id}")
    print(f"Segments: {len(segments)}")
    print(f"Conditions: {', '.join(requested_conditions)}")
    print(f"Sampling: {args.sample_fps} FPS")
    print(f"Frame width: {args.frame_width}")
    print(f"Video token budget: {args.video_token_budget}")
    print("LABEL FIREWALL: labels will be loaded only after inference.")
    print()

    classifier = PromptBiasClassifier(args.model_id)

    for pos, row in enumerate(segments.itertuples(index=False), start=1):
        segment_idx = int(row.segment_idx)
        remaining = [
            c for c in requested_conditions
            if (segment_idx, c) not in done
        ]

        if not remaining:
            print(
                f"[{pos}/{len(segments)}] segment {segment_idx}: already complete",
                flush=True,
            )
            continue

        segment_path = Path(row.segment_path)
        print(
            f"\n[{pos}/{len(segments)}] segment {segment_idx}: "
            f"{segment_path.name}",
            flush=True,
        )
        print(f"  remaining: {', '.join(remaining)}", flush=True)

        try:
            frames, timestamps, duration = base.sample_full_segment_frames(
                video_path=segment_path,
                sample_fps=args.sample_fps,
                max_duration=args.max_duration,
                frame_width=args.frame_width,
            )

            print(
                f"  duration={duration:.1f}s | frames={len(frames)}",
                flush=True,
            )

            # IMPORTANT: the exact same in-memory frames are used for all prompts.
            for condition in remaining:
                started = time.time()
                try:
                    (
                        result,
                        raw,
                        inference_sec,
                        peak_allocated_gb,
                        peak_reserved_gb,
                    ) = classifier.predict_condition(
                        frames=frames,
                        sample_fps=args.sample_fps,
                        role_description=args.role_description,
                        total_pixels=total_pixels,
                        max_new_tokens=args.max_new_tokens,
                        condition=condition,
                    )

                    elapsed = time.time() - started

                    out_row = {
                        "segment_idx": segment_idx,
                        "condition": condition,
                        "segment_path": str(segment_path),
                        "video": getattr(row, "video", ""),
                        "patient_id": getattr(row, "patient_id", ""),
                        "session_id": getattr(row, "session_id", ""),
                        "segment_id": getattr(row, "segment_id", ""),
                        "status": "ok",
                        "error": "",
                        "label": result["label"],
                        "confidence": result["confidence"],
                        "visual_type": result["visual_type"],
                        "actor": result["actor"],
                        "evidence_windows": json.dumps(
                            result.get("evidence_windows", []),
                            ensure_ascii=False,
                        ),
                        "counterevidence": json.dumps(
                            result.get("counterevidence", []),
                            ensure_ascii=False,
                        ),
                        "reason": result.get("reason", ""),
                        "frames_sent": len(frames),
                        "sample_fps": args.sample_fps,
                        "frame_width": args.frame_width,
                        "video_token_budget": args.video_token_budget,
                        "inference_sec": round(inference_sec, 3),
                        "elapsed_sec": round(elapsed, 3),
                        "peak_allocated_gb": (
                            round(peak_allocated_gb, 3)
                            if peak_allocated_gb is not None else None
                        ),
                        "peak_reserved_gb": (
                            round(peak_reserved_gb, 3)
                            if peak_reserved_gb is not None else None
                        ),
                    }

                    pred_df = upsert_row(
                        pred_df, out_row, predictions_csv
                    )
                    done.add((segment_idx, condition))

                    append_detail(
                        details_jsonl,
                        {
                            "segment_idx": segment_idx,
                            "condition": condition,
                            "sampled_timestamps_sec": timestamps,
                            "prediction": result,
                            "raw_model_output": raw,
                        },
                    )

                    print(
                        f"  {condition:18s} -> {result['label']:10s} "
                        f"| conf={result['confidence']:.2f} "
                        f"| {inference_sec:.1f}s",
                        flush=True,
                    )

                except Exception as exc:
                    traceback.print_exc()
                    err_row = {
                        "segment_idx": segment_idx,
                        "condition": condition,
                        "segment_path": str(segment_path),
                        "video": getattr(row, "video", ""),
                        "patient_id": getattr(row, "patient_id", ""),
                        "session_id": getattr(row, "session_id", ""),
                        "segment_id": getattr(row, "segment_id", ""),
                        "status": "error",
                        "error": repr(exc),
                    }
                    pred_df = upsert_row(
                        pred_df, err_row, predictions_csv
                    )
                    print(
                        f"  {condition:18s} -> ERROR: {exc}",
                        flush=True,
                    )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            del frames
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            print(
                f"SEGMENT-LEVEL ERROR {segment_idx}: {exc}",
                flush=True,
            )
            traceback.print_exc()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nInference stage finished.")
    print(f"Predictions: {predictions_csv}")

    # -------------------------------------------------------------
    # LABEL FIREWALL OPENS HERE — after all requested inference.
    # -------------------------------------------------------------
    if args.labels_csv:
        labels_path = Path(args.labels_csv)
        if labels_path.exists():
            print("\nLABEL FIREWALL OPENED FOR EVALUATION ONLY.")
            evaluate(
                predictions_csv=predictions_csv,
                labels_csv=labels_path,
                output_dir=eval_dir,
            )
        else:
            print(f"Labels not found: {labels_path}")
            print("Inference outputs remain valid.")

    print("\nFinished.")


def parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--segments-csv",
        default="./output/visual_pilot_100/segments_manifest.csv",
    )
    p.add_argument(
        "--labels-csv",
        default="./output/visual_pilot_100/visual_pilot_100_labels.csv",
    )
    p.add_argument(
        "--output-dir",
        default="./output/qwen3vl_prompt_bias_balanced100",
    )
    p.add_argument(
        "--model-id",
        default=DEFAULT_MODEL,
    )
    p.add_argument(
        "--conditions",
        default="original,rupture_sensitive,balanced_prior",
    )
    p.add_argument(
        "--segment-indices",
        default=None,
        help="Optional comma-separated segment_idx list.",
    )
    p.add_argument(
        "--max-segments",
        type=int,
        default=None,
    )
    p.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--frame-width",
        type=int,
        default=384,
    )
    p.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
    )
    p.add_argument(
        "--video-token-budget",
        type=int,
        default=8192,
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=400,
    )
    p.add_argument(
        "--role-description",
        default=(
            "Patient is the person on the LEFT side of the video. "
            "Therapist is the person on the RIGHT side of the video."
        ),
    )

    return p


if __name__ == "__main__":
    run(parser().parse_args())
