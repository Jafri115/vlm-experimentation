#!/usr/bin/env python
"""
V8 — Qwen3-VL-30B-A3B VLM-only zero-shot rupture ablation
========================================================

Research question
-----------------
Can a general-purpose vision-language model classify psychotherapy alliance
rupture ZERO-SHOT from patient-focused video?

This script stays VLM-ONLY.

It does NOT use:
- Action Units / OpenFace
- head-pose estimators
- pose estimation
- optical flow
- engineered visual features
- trained rupture classifiers
- human labels in any model prompt

Human 3RS labels are loaded ONLY AFTER inference for evaluation.

Three zero-shot conditions are run with the SAME Qwen3-VL-30B-A3B-Instruct model:

A) DIRECT
   One patient-focused 60-s video -> one rupture classification.

B) WINDOWED_DIRECT
   Four independent 15-s video windows -> VLM rupture assessment per window
   -> same Qwen model aggregates the four assessments into a segment decision.

C) DESCRIBE_JUDGE
   Four independent 15-s windows -> literal visual descriptions ONLY
   -> same Qwen model receives only the descriptions + compact visual 3RS rubric
   -> segment rupture decision.

Primary model output:
    NO_RUPTURE / WD_P / CF_P / MIXED_P

Evaluation:
1) legacy_any_3rs_marker
   Uses the existing pilot human_binary label (any WD_P/WD_T/CF_P/CF_T > 1)
   for continuity with the original balanced-100 pilot.

2) patient_only_rupture
   Derived after inference from WD_P_mean and CF_P_mean (>1), which better
   matches a patient-focused visual input.

The distinction is explicit in the output files.

IMPORTANT
---------
Keep this script beside:
    scripts/run_qwen3vl_visual_experiment_v5.py

V5 is imported ONLY for stable infrastructure:
- Qwen3-VL model loading/generation
- YuNet face detection
- patient track selection/cache
- patient ROI cropping
- timestamped frames
- window splitting
- JSON cleanup helpers
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLMoeForConditionalGeneration,
)

import run_qwen3vl_visual_experiment_v5 as v5


DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
ALLOWED_LABELS = {"NO_RUPTURE", "WD_P", "CF_P", "MIXED_P"}
CONDITIONS = ("direct", "windowed_direct", "describe_judge")


# =====================================================================
# Qwen3-VL-30B-A3B 4-bit backend
# =====================================================================

class Qwen30Runner(v5.QwenRunner):
    """
    Drop-in replacement for the V5 QwenRunner.

    The inherited generate() method stays unchanged, so V8 uses exactly the
    same video preparation / chat-template / generation path as V7.

    Only model loading changes:
      - Qwen3-VL-30B-A3B-Instruct (MoE)
      - bitsandbytes NF4 4-bit
      - double quantization
      - BF16 compute
      - SDPA attention (Windows-friendly)

    max_memory intentionally limits MODEL WEIGHTS on GPU to 24 GiB, leaving
    headroom on a 32 GiB GPU for video activations, KV cache, CUDA kernels,
    and generation. If the quantized model fits below 24 GiB, it remains on
    GPU; otherwise Accelerate can place some weights on CPU.
    """

    def __init__(self, model_id):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "V8 expects an NVIDIA CUDA GPU. CUDA is not available."
            )

        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        print(f"Model: {model_id}", flush=True)
        print("Backend: Qwen3VLMoeForConditionalGeneration", flush=True)
        print("Quantization: bitsandbytes NF4 4-bit + double quant", flush=True)
        print(f"Compute dtype: {compute_dtype}", flush=True)
        print(f"CUDA GPU: {torch.cuda.get_device_name(0)}", flush=True)
        total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU total: {total_gib:.2f} GiB", flush=True)
        print("GPU model-weight budget: 24 GiB", flush=True)
        print("CPU model-weight budget: 64 GiB", flush=True)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quant_config,
            dtype=compute_dtype,
            device_map="auto",
            max_memory={
                0: "24GiB",
                "cpu": "64GiB",
            },
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )

        self.model.eval()

        print(
            f"Input device: {next(self.model.parameters()).device}",
            flush=True,
        )

        if hasattr(self.model, "hf_device_map"):
            device_map = self.model.hf_device_map
            gpu_modules = sum(1 for v in device_map.values() if v == 0 or str(v) == "cuda:0")
            cpu_modules = sum(1 for v in device_map.values() if str(v) == "cpu")
            print(
                f"Accelerate device map: GPU entries={gpu_modules}, "
                f"CPU entries={cpu_modules}",
                flush=True,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(
                f"After load: allocated={allocated:.2f} GiB, "
                f"reserved={reserved:.2f} GiB",
                flush=True,
            )


VISUAL_RUPTURE_RUBRIC = r"""
VISUAL-ONLY RUPTURE RUBRIC
--------------------------
This is a compact, manual-grounded visual-only operationalization of 3RS for
this zero-shot experiment.

A rupture is a meaningful visible pattern suggesting movement away from
collaboration or visible interpersonal tension between patient and therapist.

Possible WITHDRAWAL (WD_P) visual evidence can include a PATTERN such as:
- sustained visible pulling away or disengagement from the interaction
- marked reduction in visible responsiveness / participation
- repeated or sustained breaking/avoiding of interpersonal contact
- visibly shutting down, retreating, or becoming markedly less interactive

Possible CONFRONTATION (CF_P) visual evidence can include a PATTERN such as:
- visible pushing back, disagreement, rejection, or challenge
- visibly tense/oppositional behavior directed toward the therapist
- hostile/critical/rejecting interpersonal behavior when the relational
  target is visually clear
- controlling / pressuring / intimidating interaction when visually clear

IMPORTANT NEGATIVE RULE:
Do NOT call rupture from one ordinary or ambiguous behavior alone, including:
- looking down by itself
- looking away briefly
- touching the face
- smiling
- crossed arms or crossed legs by themselves
- neutral posture
- ordinary hand gestures
- ordinary stillness
- a single head turn

Require a meaningful temporal/interpersonal pattern.

There is NO AUDIO and NO TRANSCRIPT.
Do not infer speech content, tone of voice, emotion, motivation, intention, or
clinical meaning that is not visually supported.
""".strip()


# =====================================================================
# Robust JSON parsing
# =====================================================================

def _extra_json_cleanup(candidate: str) -> str:
    cleaned = v5.deterministic_json_cleanup(candidate)

    # Missing commas between objects.
    cleaned = re.sub(r"}\s*{", "},{", cleaned)

    # Missing comma between a completed value and next key.
    cleaned = re.sub(
        r'([}\]])\s*(?="[^"\n]+"\s*:)',
        r'\1,',
        cleaned,
    )

    return cleaned


def parse_json_robust(raw_text, qwen, max_repair_tokens=1600):
    """
    Strict -> V5 deterministic repair -> extra comma repair ->
    Qwen syntax-only repair.

    The repair call uses the SAME Qwen model and is formatting-only.
    """
    candidate = v5.extract_json_candidate(raw_text)

    try:
        return json.loads(candidate), "strict"
    except Exception:
        pass

    cleaned = _extra_json_cleanup(candidate)

    try:
        return json.loads(cleaned), "deterministic_repair"
    except Exception:
        pass

    repair_prompt = f"""
Repair the malformed JSON below.

RULES:
- Return valid JSON only.
- Preserve the exact substantive decision and observations.
- Do NOT add new evidence.
- Do NOT change timestamps unless required for valid JSON syntax.
- Remove markdown/code fences.
- If a duplicate key exists, keep the last complete value.

MALFORMED JSON:
{candidate}
""".strip()

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": repair_prompt}],
        }
    ]

    repaired_raw, _, _, _ = qwen.generate(
        messages,
        max_new_tokens=max_repair_tokens,
    )

    repaired = v5.extract_json_candidate(repaired_raw)
    repaired = _extra_json_cleanup(repaired)

    return json.loads(repaired), "qwen_json_repair"


# =====================================================================
# Label validation
# =====================================================================

def normalize_label(result):
    """
    Normalize a model result without inventing a rupture.
    """
    if not isinstance(result, dict):
        result = {}

    label = str(
        result.get("primary_label", result.get("label", "NO_RUPTURE"))
        or "NO_RUPTURE"
    ).upper().strip()

    aliases = {
        "NONE": "NO_RUPTURE",
        "NO": "NO_RUPTURE",
        "NO RUPTURE": "NO_RUPTURE",
        "WITHDRAWAL": "WD_P",
        "WD": "WD_P",
        "CONFRONTATION": "CF_P",
        "CF": "CF_P",
        "MIXED": "MIXED_P",
    }
    label = aliases.get(label, label)

    if label not in ALLOWED_LABELS:
        # If the model emitted only binary + flags, recover type conservatively.
        rupture = int(bool(result.get("rupture_present", 0)))
        wd = int(bool(result.get("wd_p_present", 0)))
        cf = int(bool(result.get("cf_p_present", 0)))

        if not rupture and not wd and not cf:
            label = "NO_RUPTURE"
        elif wd and cf:
            label = "MIXED_P"
        elif wd:
            label = "WD_P"
        elif cf:
            label = "CF_P"
        else:
            # Generic unsupported "rupture" is not enough for patient type.
            label = "NO_RUPTURE"

    if label == "NO_RUPTURE":
        rupture, wd, cf = 0, 0, 0
    elif label == "WD_P":
        rupture, wd, cf = 1, 1, 0
    elif label == "CF_P":
        rupture, wd, cf = 1, 0, 1
    else:
        rupture, wd, cf = 1, 1, 1

    evidence = result.get("evidence", result.get("evidence_used", []))
    counter = result.get("counterevidence", [])

    if not isinstance(evidence, list):
        evidence = [str(evidence)] if str(evidence).strip() else []
    if not isinstance(counter, list):
        counter = [str(counter)] if str(counter).strip() else []

    strength = result.get("strength", result.get("decision_strength", None))
    try:
        strength = int(round(float(strength)))
        strength = max(1, min(5, strength))
    except Exception:
        strength = None

    return {
        "primary_label": label,
        "rupture_present": rupture,
        "wd_p_present": wd,
        "cf_p_present": cf,
        "strength": strength,
        "temporal_pattern": str(
            result.get("temporal_pattern", "") or ""
        ).strip(),
        "evidence": [str(x).strip() for x in evidence if str(x).strip()],
        "counterevidence": [
            str(x).strip() for x in counter if str(x).strip()
        ],
        "reason": str(result.get("reason", "") or "").strip(),
    }


# =====================================================================
# Prompt helpers
# =====================================================================

def therapist_context(therapist_side):
    if therapist_side in {"left", "right"}:
        return (
            f"The therapist is on IMAGE-{therapist_side.upper()} in the "
            "original scene. Use this only when relational orientation is "
            "visually clear. Do not claim exact eye contact when the eyes "
            "are too small or ambiguous."
        )

    return (
        "The therapist's image-side is UNKNOWN from the current patient crop. "
        "Do not infer gaze toward/away from therapist from image-left/right."
    )


def direct_messages(
    frames,
    sample_fps,
    total_pixels,
    therapist_side,
):
    prompt = f"""
You are performing a ZERO-SHOT VISUAL-ONLY alliance rupture classification.

You see a patient-focused video covering approximately one 60-second
psychotherapy segment. Frame timestamps are global within the segment.

{therapist_context(therapist_side)}

{VISUAL_RUPTURE_RUBRIC}

TASK
----
Judge the WHOLE 60-second segment, not isolated frames.

First consider the temporal pattern:
- what is stable
- what changes
- whether any behavior is sustained or repeated
- whether apparently negative-looking behavior is actually ordinary/ambiguous
- whether there is counterevidence of ongoing participation

Return JSON only:

{{
  "primary_label": "NO_RUPTURE|WD_P|CF_P|MIXED_P",
  "strength": 1,
  "temporal_pattern": "brief whole-segment visual pattern",
  "evidence": [
    "timestamp/range + specific visible evidence supporting the decision"
  ],
  "counterevidence": [
    "timestamp/range + specific visible evidence against rupture"
  ],
  "reason": "short visual-only justification"
}}

strength is 1-5 and describes how strong the visible rupture pattern appears.
It is NOT a calibrated probability.

If visual evidence is insufficient or consists only of isolated ambiguous
behaviors, choose NO_RUPTURE.

Do not output any human score or claim access to 3RS ratings.
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
                {"type": "text", "text": prompt},
            ],
        }
    ]


def window_classification_messages(
    frames,
    window_start,
    window_end,
    sample_fps,
    total_pixels,
    therapist_side,
):
    prompt = f"""
You are performing ZERO-SHOT VISUAL-ONLY rupture assessment of ONE short
window from a psychotherapy segment.

This window covers approximately {window_start:.0f}-{window_end:.0f} seconds.
Timestamps printed on frames are global within the full 60-second segment.

{therapist_context(therapist_side)}

{VISUAL_RUPTURE_RUBRIC}

This is ONLY a window-level assessment.
Do not assume the rest of the 60-second segment has the same pattern.

Return JSON only:

{{
  "window": "{window_start:.0f}-{window_end:.0f}",
  "primary_label": "NO_RUPTURE|WD_P|CF_P|MIXED_P",
  "support": "none|weak|moderate|strong",
  "visible_pattern": "short description of what happens in this window",
  "evidence": [
    "specific visible evidence with approximate timestamp"
  ],
  "counterevidence": [
    "specific visible evidence against rupture"
  ]
}}

Use NO_RUPTURE when the visible evidence is only ordinary/ambiguous behavior.
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
                {"type": "text", "text": prompt},
            ],
        }
    ]


def window_description_messages(
    frames,
    window_start,
    window_end,
    sample_fps,
    total_pixels,
    therapist_side,
):
    prompt = f"""
You are a LITERAL VISUAL OBSERVER.

You see ONLY the patient in a psychotherapy video from approximately
{window_start:.0f}-{window_end:.0f} seconds.
Frame timestamps are global within the 60-second segment.

{therapist_context(therapist_side)}

IMPORTANT
---------
Do NOT classify or discuss:
- rupture
- withdrawal
- confrontation
- alliance quality
- resistance
- engagement
- emotion, intention, motivation, or speech content

Describe only what is visibly happening.

Focus naturally on:
- overall visible participation/stillness
- gaze direction when it is actually discernible
- head/face changes
- hand/arm behavior
- posture/body movement
- changes across the window
- orientation relative to therapist only when visually clear

Do NOT turn every sampled frame into a separate event.
Describe sustained patterns and meaningful visible changes.

Return JSON only:

{{
  "window": "{window_start:.0f}-{window_end:.0f}",
  "visibility": "good|partial|poor",
  "literal_summary": "2-5 sentences describing the visible pattern",
  "notable_changes": [
    {{
      "time": "approximate timestamp/range",
      "observation": "literal visible change"
    }}
  ],
  "ambiguities": [
    "anything that cannot be determined reliably from the video"
  ]
}}
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
                {"type": "text", "text": prompt},
            ],
        }
    ]


def aggregate_window_judge_messages(window_outputs):
    payload = json.dumps(
        {"window_assessments": window_outputs},
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
You are the FINAL ZERO-SHOT VISUAL-ONLY alliance rupture judge.

You do NOT see the video directly.
You receive four independent VLM assessments of consecutive 15-second
patient-video windows.

{VISUAL_RUPTURE_RUBRIC}

WINDOW ASSESSMENTS
------------------
{payload}

TASK
----
Judge the WHOLE 60-second segment.

Do NOT simply majority-vote the window labels.
Consider:
- persistence
- repetition
- escalation/de-escalation
- whether a short event is contradicted by the rest of the segment
- whether the pattern is relationally meaningful rather than ordinary motion

Return JSON only:

{{
  "primary_label": "NO_RUPTURE|WD_P|CF_P|MIXED_P",
  "strength": 1,
  "temporal_pattern": "brief cross-window pattern",
  "evidence": [
    "specific window/timestamp evidence supporting final decision"
  ],
  "counterevidence": [
    "specific evidence against rupture"
  ],
  "reason": "short final visual-only justification"
}}

If evidence is insufficient or only isolated/ambiguous, choose NO_RUPTURE.
""".strip()

    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]


def description_judge_messages(descriptions):
    payload = json.dumps(
        {"literal_window_descriptions": descriptions},
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
You are the FINAL ZERO-SHOT VISUAL-ONLY alliance rupture judge.

You do NOT see the video.
You receive literal visual descriptions made by the SAME general-purpose VLM.
Those descriptions were produced WITHOUT rupture definitions or rupture labels.

{VISUAL_RUPTURE_RUBRIC}

LITERAL WINDOW DESCRIPTIONS
---------------------------
{payload}

TASK
----
Using ONLY the literal descriptions above, classify the whole 60-second
patient segment.

Return JSON only:

{{
  "primary_label": "NO_RUPTURE|WD_P|CF_P|MIXED_P",
  "strength": 1,
  "temporal_pattern": "brief cross-window pattern",
  "evidence": [
    "specific described visual evidence supporting final decision"
  ],
  "counterevidence": [
    "specific described evidence against rupture"
  ],
  "reason": "short final visual-only justification"
}}

Do not invent any behavior not contained in the descriptions.
If evidence is insufficient or only isolated/ambiguous, choose NO_RUPTURE.
""".strip()

    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]


# =====================================================================
# Video / tracking helpers
# =====================================================================

def therapist_side_from_tracks(tracks, selected_track):
    others = [
        t
        for t in tracks
        if t["track_id"] != selected_track["track_id"]
    ]

    if not others:
        return "unknown"

    other = max(
        others,
        key=lambda x: (
            x["detection_count"],
            x["median_face_size_px"],
        ),
    )

    return v5.side_of_track(other)


def subsample_for_direct(
    patient_frames,
    timestamps,
    target_fps,
):
    """
    Deterministically select approximately target_fps frames from the already
    timestamped patient sequence.
    """
    if not patient_frames:
        return [], []

    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")

    step = 1.0 / float(target_fps)
    next_t = 0.0

    out_frames = []
    out_times = []

    for frame, ts in zip(patient_frames, timestamps):
        if ts + 1e-6 >= next_t:
            out_frames.append(frame)
            out_times.append(float(ts))
            next_t += step

    return out_frames, out_times


# =====================================================================
# Inference bookkeeping
# =====================================================================

def load_long_predictions(path):
    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def completed_pairs(pred_df):
    if pred_df.empty:
        return set()

    required = {"segment_idx", "condition", "status"}
    if not required.issubset(pred_df.columns):
        return set()

    ok = pred_df[pred_df["status"] == "ok"].copy()

    pairs = set()

    for row in ok.itertuples(index=False):
        try:
            pairs.add((int(row.segment_idx), str(row.condition)))
        except Exception:
            pass

    return pairs


def upsert_prediction(pred_df, row, path):
    new = pd.DataFrame([row])

    if pred_df.empty:
        out = new
    else:
        out = pred_df.copy()

        if {
            "segment_idx",
            "condition",
        }.issubset(out.columns):
            mask = ~(
                (pd.to_numeric(out["segment_idx"], errors="coerce")
                 == int(row["segment_idx"]))
                & (out["condition"].astype(str) == str(row["condition"]))
            )
            out = out[mask]

        out = pd.concat([out, new], ignore_index=True)

    if not out.empty:
        out["segment_idx"] = pd.to_numeric(
            out["segment_idx"],
            errors="coerce",
        )
        out = out.sort_values(["segment_idx", "condition"])

    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def append_detail(path, obj):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def make_prediction_row(
    segment_idx,
    row,
    condition,
    normalized,
    elapsed_sec,
    peak_allocated_gb,
    peak_reserved_gb,
    patient_side,
    therapist_side,
    patient_track_id,
    selection_method,
):
    return {
        "segment_idx": int(segment_idx),
        "condition": condition,
        "status": "ok",
        "error": "",
        "model": DEFAULT_MODEL,
        "video": getattr(row, "video", ""),
        "patient_id": getattr(row, "patient_id", ""),
        "session_id": getattr(row, "session_id", ""),
        "segment_id": getattr(row, "segment_id", ""),
        "segment_start_sec": getattr(row, "segment_start_sec", ""),
        "primary_label": normalized["primary_label"],
        "rupture_present": normalized["rupture_present"],
        "wd_p_present": normalized["wd_p_present"],
        "cf_p_present": normalized["cf_p_present"],
        "strength": normalized["strength"],
        "temporal_pattern": normalized["temporal_pattern"],
        "evidence": json.dumps(
            normalized["evidence"],
            ensure_ascii=False,
        ),
        "counterevidence": json.dumps(
            normalized["counterevidence"],
            ensure_ascii=False,
        ),
        "reason": normalized["reason"],
        "patient_track_id": patient_track_id,
        "patient_selection_method": selection_method,
        "patient_side": patient_side,
        "therapist_side_for_gaze": therapist_side,
        "elapsed_sec": round(float(elapsed_sec), 3),
        "peak_allocated_gb": (
            round(float(peak_allocated_gb), 3)
            if peak_allocated_gb is not None
            else None
        ),
        "peak_reserved_gb": (
            round(float(peak_reserved_gb), 3)
            if peak_reserved_gb is not None
            else None
        ),
    }


def save_segment_summary(pred_df, path):
    if pred_df.empty:
        return

    ok = pred_df[pred_df["status"] == "ok"].copy()
    if ok.empty:
        return

    base_cols = [
        "segment_idx",
        "video",
        "patient_id",
        "session_id",
        "segment_id",
    ]

    meta = (
        ok.sort_values("segment_idx")
        .drop_duplicates("segment_idx")
        [base_cols]
    )

    piv = ok.pivot_table(
        index="segment_idx",
        columns="condition",
        values="primary_label",
        aggfunc="last",
    ).reset_index()

    piv.columns = [
        str(x)
        if x == "segment_idx"
        else f"{x}_label"
        for x in piv.columns
    ]

    out = meta.merge(piv, on="segment_idx", how="left")
    out.to_csv(path, index=False, encoding="utf-8-sig")


# =====================================================================
# Evaluation — LABELS LOADED ONLY HERE, AFTER INFERENCE
# =====================================================================

def binary_metrics(y_true, y_pred):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "N": tp + tn + fp + fn,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "precision": round(precision, 4),
        "recall_sensitivity": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
    }


def type_label_from_patient_means(row):
    wd = float(row["WD_P_mean"]) > 1.0
    cf = float(row["CF_P_mean"]) > 1.0

    if wd and cf:
        return "MIXED_P"
    if wd:
        return "WD_P"
    if cf:
        return "CF_P"
    return "NO_RUPTURE"


def multiclass_macro_metrics(y_true, y_pred):
    labels = ["NO_RUPTURE", "WD_P", "CF_P", "MIXED_P"]

    rows = []
    f1s = []

    for label in labels:
        tp = sum(
            1 for a, b in zip(y_true, y_pred)
            if a == label and b == label
        )
        fp = sum(
            1 for a, b in zip(y_true, y_pred)
            if a != label and b == label
        )
        fn = sum(
            1 for a, b in zip(y_true, y_pred)
            if a == label and b != label
        )

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0

        f1s.append(f)
        rows.append(
            {
                "class": label,
                "support": sum(1 for a in y_true if a == label),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f, 4),
            }
        )

    accuracy = sum(
        1 for a, b in zip(y_true, y_pred) if a == b
    ) / max(1, len(y_true))

    return round(accuracy, 4), round(sum(f1s) / len(f1s), 4), rows


def evaluate_predictions(predictions_csv, labels_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(predictions_csv)
    pred = pred[pred["status"] == "ok"].copy()

    labels = pd.read_csv(labels_csv)

    if "eval_id" not in labels.columns:
        raise ValueError(
            "labels CSV must contain eval_id matching segment_idx"
        )

    labels = labels.rename(columns={"eval_id": "segment_idx"})
    labels["segment_idx"] = pd.to_numeric(
        labels["segment_idx"],
        errors="coerce",
    ).astype("Int64")

    labels["legacy_any_3rs_marker"] = pd.to_numeric(
        labels["human_binary"],
        errors="coerce",
    )

    labels["patient_only_rupture"] = (
        (pd.to_numeric(labels["WD_P_mean"], errors="coerce") > 1.0)
        | (pd.to_numeric(labels["CF_P_mean"], errors="coerce") > 1.0)
    ).astype(int)

    labels["patient_type_label"] = labels.apply(
        type_label_from_patient_means,
        axis=1,
    )

    keep = [
        "segment_idx",
        "legacy_any_3rs_marker",
        "patient_only_rupture",
        "patient_type_label",
        "WD_P_mean",
        "WD_T_mean",
        "CF_P_mean",
        "CF_T_mean",
    ]

    merged = pred.merge(
        labels[keep],
        on="segment_idx",
        how="inner",
        validate="many_to_one",
    )

    # Binary evaluation.
    metric_rows = []

    for condition in CONDITIONS:
        sub = merged[merged["condition"] == condition].copy()

        for target in [
            "legacy_any_3rs_marker",
            "patient_only_rupture",
        ]:
            eval_sub = sub.dropna(subset=[target]).copy()

            y_true = eval_sub[target].astype(int).tolist()
            y_pred = (
                pd.to_numeric(
                    eval_sub["rupture_present"],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
                .tolist()
            )

            m = binary_metrics(y_true, y_pred)
            metric_rows.append(
                {
                    "condition": condition,
                    "target": target,
                    **m,
                }
            )

    pd.DataFrame(metric_rows).to_csv(
        output_dir / "binary_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Per-segment joined predictions and labels.
    merged.to_csv(
        output_dir / "predictions_with_labels.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Patient-type evaluation.
    type_summary_rows = []
    type_class_rows = []
    confusion_rows = []

    classes = ["NO_RUPTURE", "WD_P", "CF_P", "MIXED_P"]

    for condition in CONDITIONS:
        sub = merged[merged["condition"] == condition].copy()

        y_true = sub["patient_type_label"].astype(str).tolist()
        y_pred = sub["primary_label"].astype(str).tolist()

        accuracy, macro_f1, class_rows = multiclass_macro_metrics(
            y_true,
            y_pred,
        )

        type_summary_rows.append(
            {
                "condition": condition,
                "N": len(sub),
                "type_accuracy": accuracy,
                "macro_f1_4class": macro_f1,
            }
        )

        for r in class_rows:
            type_class_rows.append(
                {"condition": condition, **r}
            )

        for actual in classes:
            for predicted in classes:
                count = sum(
                    1
                    for a, b in zip(y_true, y_pred)
                    if a == actual and b == predicted
                )
                confusion_rows.append(
                    {
                        "condition": condition,
                        "actual": actual,
                        "predicted": predicted,
                        "count": count,
                    }
                )

    pd.DataFrame(type_summary_rows).to_csv(
        output_dir / "patient_type_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(type_class_rows).to_csv(
        output_dir / "patient_type_metrics_by_class.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(confusion_rows).to_csv(
        output_dir / "patient_type_confusion.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return pd.DataFrame(metric_rows)


# =====================================================================
# Main experiment
# =====================================================================

def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_csv = output_dir / "v7_condition_predictions.csv"
    summary_csv = output_dir / "v7_segment_summary.csv"
    details_jsonl = output_dir / "v7_details.jsonl"

    track_preview_dir = output_dir / "track_previews"
    patient_preview_dir = output_dir / "patient_crop_previews"
    parse_failure_dir = output_dir / "json_failures"

    role_cache_path = Path(args.role_cache)
    role_cache = v5.load_role_cache(role_cache_path)

    segments = pd.read_csv(args.segments_csv)

    required = {"segment_idx", "segment_path"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(
            f"segments CSV missing columns: {sorted(missing)}"
        )

    # Explicit label firewall.
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
    present_forbidden = forbidden.intersection(segments.columns)
    if present_forbidden:
        raise ValueError(
            "LABEL FIREWALL: inference manifest contains human-label "
            f"columns: {sorted(present_forbidden)}. "
            "Use the label-free segments_manifest.csv."
        )

    segments["segment_idx"] = pd.to_numeric(
        segments["segment_idx"],
        errors="raise",
    ).astype(int)

    requested_segments = v5.parse_segment_indices(
        args.segment_indices
    )
    if requested_segments is not None:
        segments = segments[
            segments["segment_idx"].isin(requested_segments)
        ].copy()

    if args.max_segments is not None:
        segments = segments.head(args.max_segments)

    requested_conditions = [
        x.strip()
        for x in args.conditions.split(",")
        if x.strip()
    ]

    invalid_conditions = [
        x for x in requested_conditions
        if x not in CONDITIONS
    ]
    if invalid_conditions:
        raise ValueError(
            f"Unknown conditions: {invalid_conditions}. "
            f"Allowed: {CONDITIONS}"
        )

    pred_df = load_long_predictions(predictions_csv)
    done = completed_pairs(pred_df)

    window_total_pixels = int(
        args.window_video_token_budget * 32 * 32
    )
    direct_total_pixels = int(
        args.direct_video_token_budget * 32 * 32
    )

    face_detector = v5.get_face_detector(
        model_path=args.yunet_model,
        score_threshold=args.yunet_score_threshold,
        nms_threshold=args.yunet_nms_threshold,
        top_k=args.yunet_top_k,
    )

    print("", flush=True)
    print("V8 — QWEN3-VL-30B-A3B 4-BIT ZERO-SHOT RUPTURE ABLATION", flush=True)
    print("=" * 49, flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Segments selected: {len(segments)}", flush=True)
    print(f"Conditions: {', '.join(requested_conditions)}", flush=True)
    print(
        "NO AUs / OpenFace / pose / engineered CV features.",
        flush=True,
    )
    print(
        "LABEL FIREWALL: human labels are NOT loaded until "
        "all requested model inference has finished.",
        flush=True,
    )
    print(
        f"Base video sampling: {args.sample_fps} FPS; "
        f"window={args.window_seconds:.0f}s",
        flush=True,
    )
    print(
        f"Direct condition: {args.direct_fps} FPS "
        f"(VRAM-safe one-call 60-s view)",
        flush=True,
    )
    print(f"Output: {output_dir}", flush=True)
    print("", flush=True)

    qwen = Qwen30Runner(args.model_id)

    for pos, row in enumerate(
        segments.itertuples(index=False),
        start=1,
    ):
        segment_idx = int(row.segment_idx)

        remaining = [
            c
            for c in requested_conditions
            if (segment_idx, c) not in done
        ]

        if not remaining:
            print(
                f"[{pos}/{len(segments)}] segment {segment_idx}: "
                "all requested conditions already done",
                flush=True,
            )
            continue

        segment_path = Path(row.segment_path)

        print(
            f"\n[{pos}/{len(segments)}] segment {segment_idx}: "
            f"{segment_path.name}",
            flush=True,
        )
        print(
            f"  remaining: {', '.join(remaining)}",
            flush=True,
        )

        segment_started = time.time()

        try:
            # ---------------------------------------------------------
            # Shared patient-focused visual input.
            # ---------------------------------------------------------
            full_frames, timestamps, duration = v5.sample_full_frames(
                segment_path,
                sample_fps=args.sample_fps,
                max_duration=args.max_duration,
            )

            fw, fh = full_frames[0].size

            detections = []

            for frame_idx, (frame, ts) in enumerate(
                zip(full_frames, timestamps)
            ):
                faces = v5.detect_faces(
                    frame,
                    face_detector,
                    min_face_px=args.min_face_px,
                )

                for face in faces:
                    detections.append(
                        {
                            "frame_idx": frame_idx,
                            "timestamp": float(ts),
                            "bbox": face,
                        }
                    )

            tracks = v5.cluster_static_faces(
                detections,
                frame_width=fw,
                frame_height=fh,
                center_threshold=args.track_center_threshold,
            )

            tracks = v5.filter_candidate_tracks(
                tracks,
                total_frames=len(full_frames),
                min_detection_count=args.min_track_detections,
                min_detection_fraction=args.min_track_fraction,
            )

            if not tracks:
                raise RuntimeError(
                    "No persistent face candidates detected."
                )

            track_preview_path = (
                track_preview_dir
                / f"segment_{segment_idx:03d}_tracks.jpg"
            )

            v5.annotate_tracks(
                full_frames,
                timestamps,
                tracks,
                track_preview_path,
                max_frames=4,
            )

            selected_track, selection_method = v5.select_patient_track(
                tracks=tracks,
                row=row,
                preview_path=track_preview_path,
                cache=role_cache,
                cache_path=role_cache_path,
                mode=args.patient_selection,
                forced_side=args.patient_side,
            )

            patient_side = v5.side_of_track(selected_track)
            therapist_side = therapist_side_from_tracks(
                tracks,
                selected_track,
            )

            patient_roi = v5.face_to_person_roi(
                selected_track["median_face_bbox"],
                frame_width=fw,
                frame_height=fh,
                width_mult=args.roi_width_face_mult,
                top_mult=args.roi_top_face_mult,
                bottom_mult=args.roi_bottom_face_mult,
            )

            patient_frames = v5.crop_patient_frames(
                full_frames,
                timestamps,
                patient_roi,
                frame_width=args.frame_width,
            )

            patient_preview_path = (
                patient_preview_dir
                / f"segment_{segment_idx:03d}_patient.jpg"
            )
            v5.make_contact_sheet(
                patient_frames,
                patient_preview_path,
                cols=4,
                max_frames=12,
                thumb_width=220,
            )

            windows = v5.split_windows(
                patient_frames,
                timestamps,
                window_seconds=args.window_seconds,
                max_duration=min(args.max_duration, duration),
            )

            print(
                f"  patient TRACK {selected_track['track_id']} "
                f"via {selection_method}; "
                f"patient_side={patient_side}; "
                f"therapist_side={therapist_side}",
                flush=True,
            )

            # =========================================================
            # CONDITION A: DIRECT
            # =========================================================
            if "direct" in remaining:
                cond_started = time.time()
                peak_a = []
                peak_r = []

                try:
                    direct_frames, direct_times = subsample_for_direct(
                        patient_frames,
                        timestamps,
                        args.direct_fps,
                    )

                    print(
                        f"  [direct] one 60-s call | "
                        f"{len(direct_frames)} frames",
                        flush=True,
                    )

                    raw, sec, pa, pr = qwen.generate(
                        direct_messages(
                            direct_frames,
                            sample_fps=args.direct_fps,
                            total_pixels=direct_total_pixels,
                            therapist_side=therapist_side,
                        ),
                        max_new_tokens=args.final_max_new_tokens,
                    )

                    if pa is not None:
                        peak_a.append(pa)
                    if pr is not None:
                        peak_r.append(pr)

                    try:
                        parsed, parse_method = parse_json_robust(
                            raw,
                            qwen,
                        )
                    except Exception as exc:
                        parse_failure_dir.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        failure = (
                            parse_failure_dir
                            / f"segment_{segment_idx:03d}_direct_raw.txt"
                        )
                        failure.write_text(raw, encoding="utf-8")
                        raise RuntimeError(
                            f"Direct JSON parse failed; raw saved to {failure}"
                        ) from exc

                    normalized = normalize_label(parsed)

                    row_out = make_prediction_row(
                        segment_idx=segment_idx,
                        row=row,
                        condition="direct",
                        normalized=normalized,
                        elapsed_sec=time.time() - cond_started,
                        peak_allocated_gb=max(peak_a) if peak_a else None,
                        peak_reserved_gb=max(peak_r) if peak_r else None,
                        patient_side=patient_side,
                        therapist_side=therapist_side,
                        patient_track_id=selected_track["track_id"],
                        selection_method=selection_method,
                    )

                    pred_df = upsert_prediction(
                        pred_df,
                        row_out,
                        predictions_csv,
                    )
                    done.add((segment_idx, "direct"))

                    append_detail(
                        details_jsonl,
                        {
                            "segment_idx": segment_idx,
                            "condition": "direct",
                            "direct_timestamps": direct_times,
                            "parsed": parsed,
                            "normalized": normalized,
                            "parse_method": parse_method,
                            "raw": raw,
                            "inference_sec": sec,
                        },
                    )

                    print(
                        f"    -> {normalized['primary_label']} "
                        f"| {sec:.1f}s",
                        flush=True,
                    )

                except Exception as exc:
                    traceback.print_exc()

                    err_row = {
                        "segment_idx": segment_idx,
                        "condition": "direct",
                        "status": "error",
                        "error": repr(exc),
                        "model": args.model_id,
                        "video": getattr(row, "video", ""),
                        "patient_id": getattr(row, "patient_id", ""),
                        "session_id": getattr(row, "session_id", ""),
                    }
                    pred_df = upsert_prediction(
                        pred_df,
                        err_row,
                        predictions_csv,
                    )

                    print(
                        f"    DIRECT ERROR: {exc}",
                        flush=True,
                    )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # =========================================================
            # CONDITION B: WINDOWED DIRECT
            # =========================================================
            if "windowed_direct" in remaining:
                cond_started = time.time()
                peak_a = []
                peak_r = []
                window_results = []
                window_details = []

                try:
                    for w_no, w in enumerate(windows, start=1):
                        print(
                            f"  [windowed_direct] window {w_no}/"
                            f"{len(windows)} "
                            f"{w['start']:.0f}-{w['end']:.0f}s",
                            flush=True,
                        )

                        raw, sec, pa, pr = qwen.generate(
                            window_classification_messages(
                                w["frames"],
                                w["start"],
                                w["end"],
                                sample_fps=args.sample_fps,
                                total_pixels=window_total_pixels,
                                therapist_side=therapist_side,
                            ),
                            max_new_tokens=args.window_max_new_tokens,
                        )

                        if pa is not None:
                            peak_a.append(pa)
                        if pr is not None:
                            peak_r.append(pr)

                        parsed, method = parse_json_robust(
                            raw,
                            qwen,
                        )

                        # Keep window output mostly as generated.
                        parsed["window_start"] = w["start"]
                        parsed["window_end"] = w["end"]

                        window_results.append(parsed)
                        window_details.append(
                            {
                                "window_start": w["start"],
                                "window_end": w["end"],
                                "parsed": parsed,
                                "raw": raw,
                                "parse_method": method,
                                "inference_sec": sec,
                            }
                        )

                    final_raw, final_sec, pa, pr = qwen.generate(
                        aggregate_window_judge_messages(
                            window_results
                        ),
                        max_new_tokens=args.final_max_new_tokens,
                    )

                    if pa is not None:
                        peak_a.append(pa)
                    if pr is not None:
                        peak_r.append(pr)

                    final_parsed, final_method = parse_json_robust(
                        final_raw,
                        qwen,
                    )
                    normalized = normalize_label(final_parsed)

                    row_out = make_prediction_row(
                        segment_idx=segment_idx,
                        row=row,
                        condition="windowed_direct",
                        normalized=normalized,
                        elapsed_sec=time.time() - cond_started,
                        peak_allocated_gb=max(peak_a) if peak_a else None,
                        peak_reserved_gb=max(peak_r) if peak_r else None,
                        patient_side=patient_side,
                        therapist_side=therapist_side,
                        patient_track_id=selected_track["track_id"],
                        selection_method=selection_method,
                    )

                    pred_df = upsert_prediction(
                        pred_df,
                        row_out,
                        predictions_csv,
                    )
                    done.add((segment_idx, "windowed_direct"))

                    append_detail(
                        details_jsonl,
                        {
                            "segment_idx": segment_idx,
                            "condition": "windowed_direct",
                            "windows": window_details,
                            "final_parsed": final_parsed,
                            "final_normalized": normalized,
                            "final_raw": final_raw,
                            "final_parse_method": final_method,
                            "final_inference_sec": final_sec,
                        },
                    )

                    print(
                        f"    FINAL -> {normalized['primary_label']}",
                        flush=True,
                    )

                except Exception as exc:
                    traceback.print_exc()

                    err_row = {
                        "segment_idx": segment_idx,
                        "condition": "windowed_direct",
                        "status": "error",
                        "error": repr(exc),
                        "model": args.model_id,
                        "video": getattr(row, "video", ""),
                        "patient_id": getattr(row, "patient_id", ""),
                        "session_id": getattr(row, "session_id", ""),
                    }
                    pred_df = upsert_prediction(
                        pred_df,
                        err_row,
                        predictions_csv,
                    )

                    print(
                        f"    WINDOWED_DIRECT ERROR: {exc}",
                        flush=True,
                    )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # =========================================================
            # CONDITION C: DESCRIBE -> JUDGE
            # =========================================================
            if "describe_judge" in remaining:
                cond_started = time.time()
                peak_a = []
                peak_r = []
                descriptions = []
                description_details = []

                try:
                    for w_no, w in enumerate(windows, start=1):
                        print(
                            f"  [describe_judge] describe window {w_no}/"
                            f"{len(windows)} "
                            f"{w['start']:.0f}-{w['end']:.0f}s",
                            flush=True,
                        )

                        raw, sec, pa, pr = qwen.generate(
                            window_description_messages(
                                w["frames"],
                                w["start"],
                                w["end"],
                                sample_fps=args.sample_fps,
                                total_pixels=window_total_pixels,
                                therapist_side=therapist_side,
                            ),
                            max_new_tokens=args.description_max_new_tokens,
                        )

                        if pa is not None:
                            peak_a.append(pa)
                        if pr is not None:
                            peak_r.append(pr)

                        parsed, method = parse_json_robust(
                            raw,
                            qwen,
                        )

                        parsed["window_start"] = w["start"]
                        parsed["window_end"] = w["end"]

                        descriptions.append(parsed)
                        description_details.append(
                            {
                                "window_start": w["start"],
                                "window_end": w["end"],
                                "parsed": parsed,
                                "raw": raw,
                                "parse_method": method,
                                "inference_sec": sec,
                            }
                        )

                    final_raw, final_sec, pa, pr = qwen.generate(
                        description_judge_messages(descriptions),
                        max_new_tokens=args.final_max_new_tokens,
                    )

                    if pa is not None:
                        peak_a.append(pa)
                    if pr is not None:
                        peak_r.append(pr)

                    final_parsed, final_method = parse_json_robust(
                        final_raw,
                        qwen,
                    )

                    normalized = normalize_label(final_parsed)

                    row_out = make_prediction_row(
                        segment_idx=segment_idx,
                        row=row,
                        condition="describe_judge",
                        normalized=normalized,
                        elapsed_sec=time.time() - cond_started,
                        peak_allocated_gb=max(peak_a) if peak_a else None,
                        peak_reserved_gb=max(peak_r) if peak_r else None,
                        patient_side=patient_side,
                        therapist_side=therapist_side,
                        patient_track_id=selected_track["track_id"],
                        selection_method=selection_method,
                    )

                    pred_df = upsert_prediction(
                        pred_df,
                        row_out,
                        predictions_csv,
                    )
                    done.add((segment_idx, "describe_judge"))

                    append_detail(
                        details_jsonl,
                        {
                            "segment_idx": segment_idx,
                            "condition": "describe_judge",
                            "literal_descriptions": description_details,
                            "final_parsed": final_parsed,
                            "final_normalized": normalized,
                            "final_raw": final_raw,
                            "final_parse_method": final_method,
                            "final_inference_sec": final_sec,
                        },
                    )

                    print(
                        f"    FINAL -> {normalized['primary_label']}",
                        flush=True,
                    )

                except Exception as exc:
                    traceback.print_exc()

                    err_row = {
                        "segment_idx": segment_idx,
                        "condition": "describe_judge",
                        "status": "error",
                        "error": repr(exc),
                        "model": args.model_id,
                        "video": getattr(row, "video", ""),
                        "patient_id": getattr(row, "patient_id", ""),
                        "session_id": getattr(row, "session_id", ""),
                    }
                    pred_df = upsert_prediction(
                        pred_df,
                        err_row,
                        predictions_csv,
                    )

                    print(
                        f"    DESCRIBE_JUDGE ERROR: {exc}",
                        flush=True,
                    )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            save_segment_summary(
                pred_df,
                summary_csv,
            )

            print(
                f"  segment total: "
                f"{time.time() - segment_started:.1f}s",
                flush=True,
            )

            del full_frames
            del patient_frames

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

    # -----------------------------------------------------------------
    # Evaluation is deliberately AFTER all model inference.
    # -----------------------------------------------------------------
    print("\nInference stage finished.", flush=True)
    print(f"Predictions: {predictions_csv}", flush=True)

    save_segment_summary(
        pred_df,
        summary_csv,
    )

    if args.labels_csv:
        labels_path = Path(args.labels_csv)

        if labels_path.exists():
            print(
                "\nLABEL FIREWALL OPENED FOR EVALUATION ONLY.",
                flush=True,
            )

            metrics = evaluate_predictions(
                predictions_csv=predictions_csv,
                labels_csv=labels_path,
                output_dir=output_dir / "evaluation",
            )

            print("\nBinary evaluation:", flush=True)
            print(
                metrics.to_string(index=False),
                flush=True,
            )
            print(
                f"\nEvaluation outputs: "
                f"{output_dir / 'evaluation'}",
                flush=True,
            )
        else:
            print(
                f"\nLabels file not found: {labels_path}",
                flush=True,
            )
            print(
                "Inference outputs are still valid. "
                "Run again after placing the evaluation-only labels file.",
                flush=True,
            )

    print("\nV8 finished.", flush=True)


# =====================================================================
# CLI
# =====================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "V8: Qwen3-VL-30B-A3B 4-bit VLM-only zero-shot rupture ablation "
            "(Direct vs Windowed Direct vs Describe->Judge)."
        )
    )

    p.add_argument(
        "--segments-csv",
        required=True,
    )

    p.add_argument(
        "--labels-csv",
        default=(
            "./output/visual_pilot_100/"
            "visual_pilot_100_labels.csv"
        ),
        help=(
            "Evaluation-only labels. NEVER loaded during inference."
        ),
    )

    p.add_argument(
        "--output-dir",
        default="./output/qwen3vl_v8_qwen30_zeroshot_ablation",
    )

    p.add_argument(
        "--model-id",
        default=DEFAULT_MODEL,
    )

    p.add_argument(
        "--conditions",
        default="direct,windowed_direct,describe_judge",
    )

    p.add_argument(
        "--segment-indices",
        default=None,
        help=(
            "Comma-separated segment_idx values. "
            "Omit for all manifest rows."
        ),
    )

    p.add_argument(
        "--max-segments",
        type=int,
        default=None,
    )

    # Shared patient-crop sampling.
    p.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
    )

    p.add_argument(
        "--window-seconds",
        type=float,
        default=15.0,
    )

    p.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
    )

    p.add_argument(
        "--frame-width",
        type=int,
        default=320,
    )

    p.add_argument(
        "--window-video-token-budget",
        type=int,
        default=4096,
    )

    # Direct full-segment view is deliberately lighter for VRAM safety.
    p.add_argument(
        "--direct-fps",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--direct-video-token-budget",
        type=int,
        default=6144,
    )

    # Existing V5 patient-side cache.
    p.add_argument(
        "--role-cache",
        default=(
            "./output/qwen3vl_visual_experiment_v5/"
            "patient_role_cache.json"
        ),
    )

    # YuNet / persistent face tracking.
    p.add_argument(
        "--yunet-model",
        default="./models/face_detection_yunet_2026may.onnx",
    )
    p.add_argument(
        "--yunet-score-threshold",
        type=float,
        default=0.60,
    )
    p.add_argument(
        "--yunet-nms-threshold",
        type=float,
        default=0.30,
    )
    p.add_argument(
        "--yunet-top-k",
        type=int,
        default=5000,
    )
    p.add_argument(
        "--min-face-px",
        type=int,
        default=24,
    )
    p.add_argument(
        "--track-center-threshold",
        type=float,
        default=0.14,
    )
    p.add_argument(
        "--min-track-detections",
        type=int,
        default=6,
    )
    p.add_argument(
        "--min-track-fraction",
        type=float,
        default=0.08,
    )

    # Same patient ROI as V5/V6.3.
    p.add_argument(
        "--roi-width-face-mult",
        type=float,
        default=4.5,
    )
    p.add_argument(
        "--roi-top-face-mult",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--roi-bottom-face-mult",
        type=float,
        default=5.0,
    )

    p.add_argument(
        "--patient-selection",
        choices=[
            "interactive",
            "leftmost",
            "rightmost",
            "largest",
        ],
        default="interactive",
    )

    p.add_argument(
        "--patient-side",
        choices=["left", "right"],
        default=None,
    )

    # Generation budgets.
    p.add_argument(
        "--window-max-new-tokens",
        type=int,
        default=650,
    )
    p.add_argument(
        "--description-max-new-tokens",
        type=int,
        default=650,
    )
    p.add_argument(
        "--final-max-new-tokens",
        type=int,
        default=800,
    )

    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
