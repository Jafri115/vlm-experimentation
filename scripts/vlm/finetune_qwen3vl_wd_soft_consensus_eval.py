#!/usr/bin/env python
"""
Qwen3-VL WD_P binary / soft-binary QLoRA experiment.

Primary scientific question
---------------------------
Can Qwen3-VL distinguish clear patient withdrawal from low/no withdrawal
more reliably than it can regress the exact 1-5 WD_P salience score?

The script supports three supervised targets:

1) soft
   Threshold EACH rater independently at the selected 3RS boundary
   (default >=2) and average:
       both below threshold -> 0.0
       one at/above         -> 0.5
       both at/above        -> 1.0

2) hard
   Threshold the mean of the two WD_P ratings at the same boundary.

3) consensus
   Train ONLY on segments where both raters agree on the binary side
   of the selected threshold:
       both below threshold -> 0
       both at/above         -> 1
       disagreement          -> EXCLUDED from training

   Validation checkpointing and threshold selection are also performed
   on consensus-only validation cases. Test output reports both the
   complete held-out set and the consensus-only subset.

Supported boundaries:
   >=2 : any/subtle withdrawal evidence versus rating 1
   >=3 : clear/somewhat-salient-or-greater withdrawal

The same existing patient-disjoint manifest is reused, so the experiment
does not change the train/validation/test patients.

Training
--------
- Qwen/Qwen3-VL-8B-Instruct
- 4-bit QLoRA
- 16 chronological patient frames across the labelled minute
- mean_all multimodal pooling by default
- one binary logit
- BCEWithLogitsLoss
- automatic positive-class weighting from the TRAIN split
- best checkpoint selected by validation AUPRC
- probability threshold selected on validation F1, then frozen for test

Evaluation
----------
The script reports:
- AUPRC / AUROC
- precision / recall / F1
- soft-label BCE and Brier score
- consensus-only metrics (rater agreement only)
- prediction probability spread
- patient-wise metrics
- periodic train/validation curves

Place in your VLM_experiments/scripts directory.

Requires the tested helper:
    finetune_qwen3vl_rupture_pilot.py
in the same scripts directory.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
    brier_score_loss,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vlm import finetune_qwen3vl_rupture_pilot as base


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
SEGMENT_KEY = ["patient_id", "session_id", "video", "segment_id"]

def build_binary_prompt(positive_threshold: float) -> str:
    """
    Build a 3RS-informed visual-only prompt whose decision boundary
    matches the label threshold used to construct the binary target.

    threshold = 2:
        "any/subtle withdrawal evidence" versus rating 1.

    threshold = 3:
        "clear/somewhat-salient-or-greater withdrawal" versus <3.
    """
    threshold = float(positive_threshold)

    if threshold == 2.0:
        boundary_text = """
3RS DECISION BOUNDARY: RATING >= 2
---------------------------------
The positive class includes Patient Moves Away ratings of 2 or higher.

A rating of 2 lies between:
- 1 = movements away are not salient, and
- 3 = movements away are somewhat salient.

Therefore, for this >=2 experiment, a positive example does NOT require
a fully clear withdrawal marker. It can include subtle but meaningful
visible evidence that is stronger than a rating of 1.

The negative class corresponds to rating 1:
- no withdrawal marker is visible, OR
- only one possible marker is present and it is very low in intensity
  and clarity and does not stand out.

When evidence is subtle, preserve it in the representation rather than
discarding it merely because it would not yet warrant a rating of 3.
""".strip()

    elif threshold == 3.0:
        boundary_text = """
3RS DECISION BOUNDARY: RATING >= 3
---------------------------------
The positive class corresponds to somewhat salient or greater
Patient Moves Away.

There should be at least one CLEAR withdrawal marker.

The negative class corresponds to ratings below 3:
- no clear withdrawal marker, OR
- only weak, brief, ambiguous, or isolated possible evidence.
""".strip()

    else:
        raise ValueError(
            "This experiment supports only 3RS binary thresholds 2 or 3."
        )

    return f"""
You are observing ONLY the PATIENT in chronological frames sampled
across one labelled minute of a psychotherapy session.

There is NO audio and NO transcript.

The target is based on the 3RS v2022 construct
"Patient moves away" (withdrawal).

3RS CONCEPT
-----------
Withdrawal means movement away from the therapist and/or from the
work of therapy.

For this visual-only task, assess ONLY directly visible evidence.
Do not infer speech content, internal states, intentions, emotions,
motivation, resistance, or alliance quality.

VISUAL EVIDENCE TO INSPECT
--------------------------
Pay close attention to persistent states and changes involving:

GAZE / EYES
- sustained gaze away from the therapist
- sustained downward gaze
- prolonged gaze disengagement
- eyes closed for a sustained period
- changes from visually engaged gaze to disengaged gaze

HEAD / FACE
- head angled or turned away for a sustained period
- sustained head-down posture
- visible reduction in facial responsiveness
- visible facial behavior that accompanies disengagement
- changes from active visible responding to reduced visible responding

BODY / POSTURE
- collapsed or slumped posture
- body turning away
- torso moving backward or withdrawing from the interaction
- arms held close to the body when part of a broader withdrawal pattern
- visible reduction in movement across the minute
- prolonged unusual stillness when the interaction appears to continue

GESTURES / MOVEMENT
- shrugging
- giving-up-like visible gestures
- reduced gesturing or movement relative to earlier in the minute
- hands or arms becoming less active when this occurs together with
  other disengagement cues

TEMPORAL RULE
-------------
Use the full minute.

Distinguish:
- sustained or repeated patterns
from
- brief isolated movements.

Do not decide from one sampled frame alone.

IMPORTANT NON-INFERENCE RULES
-----------------------------
Do NOT automatically classify any of the following as withdrawal:
- a brief gaze shift
- briefly looking down
- a normal pause
- ordinary listening
- ordinary thinking
- a neutral facial expression
- crossed arms by themselves
- hands on lap by themselves
- stillness by itself
- a single shrug
- a single smile
- head orientation without supporting context

These can occur without withdrawal.

Also do NOT infer verbal withdrawal markers that cannot be observed
visually here, including:
- avoidant storytelling
- abstract communication
- topic shifting
- verbal minimal responses
- deferential verbal agreement
- verbal denial
- content/affect mismatch that requires knowing speech content

{boundary_text}

Your role is not to output the final label in text. Build an internal
representation that preserves these 3RS-informed visual distinctions
so that the supervised classifier can learn from the provided target.
""".strip()


# ---------------------------------------------------------------------------
# JSON / reproducibility utilities
# ---------------------------------------------------------------------------

def clean_json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (
        math.isnan(value) or math.isinf(value)
    ):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {
            str(k): clean_json_value(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            clean_json_value(v)
            for v in value
        ]
    return value


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            clean_json_value(obj),
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Build hard + soft labels from the two raw coder rows
# ---------------------------------------------------------------------------

def normalize_key_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["patient_id"] = (
        out["patient_id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )
    out["session_id"] = (
        out["session_id"]
        .astype(str)
        .str.strip()
    )
    out["video"] = (
        out["video"]
        .astype(str)
        .str.strip()
    )
    out["segment_id"] = pd.to_numeric(
        out["segment_id"],
        errors="coerce",
    ).astype("Int64")

    return out


def build_two_rater_targets(
    labels_csv: Path,
    threshold: float,
) -> pd.DataFrame:
    raw = pd.read_csv(labels_csv)

    required = set(
        SEGMENT_KEY
        + ["coder", "WD_P"]
    )

    missing = required - set(raw.columns)

    if missing:
        raise RuntimeError(
            "Ratings CSV is missing required columns: "
            f"{sorted(missing)}"
        )

    ratings = raw[
        SEGMENT_KEY + ["coder", "WD_P"]
    ].copy()

    ratings = normalize_key_types(
        ratings
    )

    ratings["coder"] = (
        ratings["coder"]
        .astype(str)
        .str.strip()
        .replace(
            {
                "segments Alex": "Alex",
            }
        )
    )

    ratings["WD_P"] = pd.to_numeric(
        ratings["WD_P"],
        errors="coerce",
    )

    ratings = ratings.dropna(
        subset=[
            "patient_id",
            "session_id",
            "video",
            "segment_id",
            "coder",
            "WD_P",
        ]
    ).copy()

    # There should be one row per physical segment × coder.
    duplicate_counts = (
        ratings.groupby(
            SEGMENT_KEY + ["coder"],
            dropna=False,
        )
        .size()
    )

    duplicate_rows = int(
        (duplicate_counts > 1).sum()
    )

    if duplicate_rows:
        raise RuntimeError(
            f"Found {duplicate_rows} duplicated "
            "segment × coder combinations. "
            "Resolve these before constructing soft labels."
        )

    grouped_rows = []

    for key, group in ratings.groupby(
        SEGMENT_KEY,
        sort=False,
        dropna=False,
    ):
        group = group.sort_values(
            "coder"
        )

        if group["coder"].nunique() != 2:
            continue

        if len(group) != 2:
            continue

        r1 = group.iloc[0]
        r2 = group.iloc[1]

        score1 = float(r1["WD_P"])
        score2 = float(r2["WD_P"])

        bin1 = int(
            score1 >= threshold
        )
        bin2 = int(
            score2 >= threshold
        )

        soft = (
            bin1 + bin2
        ) / 2.0

        mean_score = (
            score1 + score2
        ) / 2.0

        hard_mean = int(
            mean_score >= threshold
        )

        if bin1 == bin2:
            consensus = float(bin1)
        else:
            consensus = np.nan

        grouped_rows.append(
            {
                "patient_id": str(key[0]),
                "session_id": str(key[1]),
                "video": str(key[2]),
                "segment_id": int(key[3]),
                "coder_1": str(r1["coder"]),
                "coder_2": str(r2["coder"]),
                "WD_P_rater1": score1,
                "WD_P_rater2": score2,
                "WD_P_mean_from_raters": mean_score,
                "WD_soft": float(soft),
                "WD_hard_mean": int(hard_mean),
                "WD_consensus": consensus,
                "WD_binary_disagreement": int(
                    bin1 != bin2
                ),
                "WD_absolute_rater_difference": abs(
                    score1 - score2
                ),
            }
        )

    targets = pd.DataFrame(
        grouped_rows
    )

    if targets.empty:
        raise RuntimeError(
            "No exactly-two-rater WD_P segments found."
        )

    return targets


def merge_targets_with_manifest(
    manifest_path: Path,
    labels_csv: Path,
    threshold: float,
    output_dir: Path,
) -> pd.DataFrame:
    manifest = pd.read_csv(
        manifest_path
    )

    required = set(
        SEGMENT_KEY
        + [
            "sample_id",
            "split",
            "video_path",
            "patient_side",
        ]
    )

    missing = required - set(
        manifest.columns
    )

    if missing:
        raise RuntimeError(
            "Manifest is missing required columns: "
            f"{sorted(missing)}"
        )

    manifest = normalize_key_types(
        manifest
    )

    targets = build_two_rater_targets(
        labels_csv=labels_csv,
        threshold=threshold,
    )

    merged = manifest.merge(
        targets,
        on=SEGMENT_KEY,
        how="left",
        validate="one_to_one",
    )

    missing_targets = int(
        merged["WD_soft"].isna().sum()
    )

    print(
        f"Manifest rows: {len(manifest)}"
    )
    print(
        f"Rows with two-rater targets: "
        f"{len(merged) - missing_targets}"
    )
    print(
        f"Rows missing two-rater target: "
        f"{missing_targets}"
    )

    if missing_targets:
        examples = (
            merged.loc[
                merged["WD_soft"].isna(),
                [
                    "sample_id",
                    "patient_id",
                    "session_id",
                    "video",
                    "segment_id",
                ],
            ]
            .head(10)
        )

        print(
            "\nExamples missing raw two-rater labels:"
        )
        print(
            examples.to_string(index=False)
        )

    merged = merged.dropna(
        subset=["WD_soft"]
    ).copy()

    merged["WD_hard_mean"] = (
        merged["WD_hard_mean"]
        .astype(int)
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        output_dir
        / "manifest_with_binary_targets.csv",
        index=False,
    )

    print("\nTARGET DISTRIBUTION")
    print("=" * 72)

    for split in [
        "train",
        "val",
        "test",
    ]:
        sub = merged[
            merged["split"] == split
        ]

        if sub.empty:
            continue

        counts = (
            sub["WD_soft"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        print(
            f"{split:5s}: n={len(sub):4d} | "
            f"soft 0/0.5/1={counts} | "
            f"hard positives="
            f"{int(sub['WD_hard_mean'].sum())} | "
            f"binary disagreements="
            f"{int(sub['WD_binary_disagreement'].sum())}"
        )

    return merged


# ---------------------------------------------------------------------------
# Binary head + Qwen pooling
# ---------------------------------------------------------------------------

class BinaryHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size,
                256,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                256,
                1,
            ),
        )

    def forward(
        self,
        pooled_hidden: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(
            pooled_hidden.float()
        )


def prepare_inputs(
    processor,
    frames: Sequence[Image.Image],
    device: torch.device,
    positive_threshold: float,
):
    content = [
        {
            "type": "image",
            "image": frame,
        }
        for frame in frames
    ]

    content.append(
        {
            "type": "text",
            "text": build_binary_prompt(
                positive_threshold
            ),
        }
    )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    moved = {}

    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue

        if key in {
            "pixel_values",
            "pixel_values_videos",
        }:
            moved[key] = value.to(
                device=device,
                dtype=torch.bfloat16,
            )
        else:
            moved[key] = value.to(
                device=device
            )

    return moved


def pool_hidden(
    model,
    inputs,
    hidden: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    attention_mask = inputs.get(
        "attention_mask"
    )

    if attention_mask is None:
        valid_mask = torch.ones(
            hidden.shape[:2],
            dtype=torch.bool,
            device=hidden.device,
        )
    else:
        valid_mask = (
            attention_mask.bool()
        )

    if pooling == "mean_all":
        weights = (
            valid_mask
            .unsqueeze(-1)
            .to(hidden.dtype)
        )

        return (
            (hidden * weights).sum(dim=1)
            / weights.sum(dim=1)
            .clamp_min(1.0)
        )

    if pooling == "mean_image":
        input_ids = inputs.get(
            "input_ids"
        )

        if input_ids is None:
            raise RuntimeError(
                "mean_image pooling requires input_ids."
            )

        base_model = (
            model.get_base_model()
            if hasattr(
                model,
                "get_base_model",
            )
            else model
        )

        image_token_id = None

        for obj in (
            getattr(
                base_model,
                "config",
                None,
            ),
            getattr(
                getattr(
                    base_model,
                    "model",
                    None,
                ),
                "config",
                None,
            ),
        ):
            if (
                obj is not None
                and hasattr(
                    obj,
                    "image_token_id",
                )
            ):
                image_token_id = getattr(
                    obj,
                    "image_token_id",
                )

                if image_token_id is not None:
                    break

        if image_token_id is None:
            raise RuntimeError(
                "Could not determine image_token_id."
            )

        mask = (
            (input_ids == int(image_token_id))
            & valid_mask
        )

        if torch.any(
            mask.sum(dim=1) == 0
        ):
            raise RuntimeError(
                "No image-token positions found."
            )

        weights = (
            mask.unsqueeze(-1)
            .to(hidden.dtype)
        )

        return (
            (hidden * weights).sum(dim=1)
            / weights.sum(dim=1)
            .clamp_min(1.0)
        )

    raise ValueError(
        f"Unsupported pooling: {pooling}"
    )


def predict_logits(
    model,
    head,
    processor,
    frames,
    device,
    pooling: str,
    positive_threshold: float,
):
    inputs = prepare_inputs(
        processor,
        frames,
        device,
        positive_threshold,
    )

    backbone = base.get_backbone(
        model
    )

    outputs = backbone(
        **inputs,
        use_cache=False,
        return_dict=True,
    )

    hidden = outputs.last_hidden_state

    pooled = pool_hidden(
        model=model,
        inputs=inputs,
        hidden=hidden,
        pooling=pooling,
    )

    return head(pooled)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def safe_auc(
    y_true,
    prob,
) -> float:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    if len(np.unique(y_true)) < 2:
        return float("nan")

    try:
        return float(
            roc_auc_score(
                y_true,
                prob,
            )
        )
    except Exception:
        return float("nan")


def safe_auprc(
    y_true,
    prob,
) -> float:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    if len(np.unique(y_true)) < 2:
        return float("nan")

    try:
        return float(
            average_precision_score(
                y_true,
                prob,
            )
        )
    except Exception:
        return float("nan")


def binary_scores(
    y_true,
    prob,
    threshold: float,
) -> dict:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )
    prob = np.asarray(
        prob,
        dtype=float,
    )

    pred = (
        prob >= threshold
    ).astype(int)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            pred,
            average="binary",
            zero_division=0,
        )
    )

    tp = int(
        np.sum(
            (y_true == 1)
            & (pred == 1)
        )
    )
    fp = int(
        np.sum(
            (y_true == 0)
            & (pred == 1)
        )
    )
    fn = int(
        np.sum(
            (y_true == 1)
            & (pred == 0)
        )
    )
    tn = int(
        np.sum(
            (y_true == 0)
            & (pred == 0)
        )
    )

    return {
        "n": int(len(y_true)),
        "positive_n": int(
            y_true.sum()
        ),
        "threshold": float(
            threshold
        ),
        "AUPRC": safe_auprc(
            y_true,
            prob,
        ),
        "AUROC": safe_auc(
            y_true,
            prob,
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(f1),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def soft_bce(
    soft_target,
    prob,
) -> float:
    y = np.asarray(
        soft_target,
        dtype=float,
    )
    p = np.clip(
        np.asarray(
            prob,
            dtype=float,
        ),
        1e-7,
        1.0 - 1e-7,
    )

    return float(
        -np.mean(
            y * np.log(p)
            + (1.0 - y)
            * np.log(1.0 - p)
        )
    )


def soft_brier(
    soft_target,
    prob,
) -> float:
    y = np.asarray(
        soft_target,
        dtype=float,
    )
    p = np.asarray(
        prob,
        dtype=float,
    )

    return float(
        np.mean(
            (p - y) ** 2
        )
    )


def compute_metrics(
    rows: List[dict],
    threshold: float,
) -> dict:
    if not rows:
        return {}

    df = pd.DataFrame(rows)

    hard = df[
        "WD_hard_mean"
    ].to_numpy(
        dtype=int
    )

    soft = df[
        "WD_soft"
    ].to_numpy(
        dtype=float
    )

    prob = df[
        "WD_probability"
    ].to_numpy(
        dtype=float
    )

    result = {
        "n": int(len(df)),
        "soft_BCE": soft_bce(
            soft,
            prob,
        ),
        "soft_Brier": soft_brier(
            soft,
            prob,
        ),
        "prob_mean": float(
            np.mean(prob)
        ),
        "prob_std": float(
            np.std(prob)
        ),
        "prob_min": float(
            np.min(prob)
        ),
        "prob_max": float(
            np.max(prob)
        ),
        "hard_mean": binary_scores(
            hard,
            prob,
            threshold,
        ),
    }

    consensus_mask = (
        df["WD_consensus"]
        .notna()
        .to_numpy()
    )

    if consensus_mask.sum() > 0:
        consensus_y = (
            df.loc[
                consensus_mask,
                "WD_consensus",
            ]
            .to_numpy(
                dtype=int
            )
        )

        consensus_prob = prob[
            consensus_mask
        ]

        result[
            "consensus_only"
        ] = binary_scores(
            consensus_y,
            consensus_prob,
            threshold,
        )

    result[
        "binary_disagreement_n"
    ] = int(
        df[
            "WD_binary_disagreement"
        ].sum()
    )

    return result


def consensus_only_rows(
    rows: List[dict],
) -> List[dict]:
    """Return only examples where both raters agree on the binary class."""
    return [
        row
        for row in rows
        if not pd.isna(
            row.get("WD_consensus", np.nan)
        )
    ]


def choose_threshold(
    rows: List[dict],
) -> Tuple[float, pd.DataFrame]:
    if not rows:
        return 0.5, pd.DataFrame()

    df = pd.DataFrame(rows)

    y = df[
        "WD_hard_mean"
    ].to_numpy(
        dtype=int
    )

    prob = df[
        "WD_probability"
    ].to_numpy(
        dtype=float
    )

    thresholds = np.linspace(
        0.05,
        0.95,
        181,
    )

    records = []

    for threshold in thresholds:
        metrics = binary_scores(
            y,
            prob,
            float(threshold),
        )

        records.append(
            {
                "threshold": float(
                    threshold
                ),
                "f1": metrics["f1"],
                "precision": (
                    metrics["precision"]
                ),
                "recall": metrics["recall"],
            }
        )

    table = pd.DataFrame(
        records
    )

    best_f1 = table[
        "f1"
    ].max()

    candidates = table[
        np.isclose(
            table["f1"],
            best_f1,
        )
    ].copy()

    # Among equal-F1 thresholds, prefer higher recall,
    # then the threshold closest to 0.5.
    candidates[
        "distance_to_half"
    ] = np.abs(
        candidates["threshold"]
        - 0.5
    )

    best = (
        candidates.sort_values(
            [
                "recall",
                "distance_to_half",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]
    )

    return (
        float(best["threshold"]),
        table,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    df: pd.DataFrame,
    split_name: str,
    model,
    head,
    processor,
    frame_cache,
    device,
    args,
    max_examples: Optional[int] = None,
    quiet: bool = False,
):
    model.eval()
    head.eval()

    work = df.copy()

    if (
        max_examples is not None
        and max_examples > 0
    ):
        work = work.head(
            max_examples
        )

    rows = []

    if not quiet:
        print(
            f"\nEvaluating {split_name}: "
            f"{len(work)} segments"
        )

    for i, row in enumerate(
        work.itertuples(index=False),
        start=1,
    ):
        try:
            frames = frame_cache.build(
                row
            )

            logits = predict_logits(
                model=model,
                head=head,
                processor=processor,
                frames=frames,
                device=device,
                pooling=args.pooling,
                positive_threshold=args.positive_threshold,
            )

            probability = float(
                torch.sigmoid(
                    logits[0, 0]
                )
                .detach()
                .float()
                .cpu()
            )

            record = {
                "sample_id": row.sample_id,
                "patient_id": row.patient_id,
                "video": row.video,
                "segment_id": int(
                    row.segment_id
                ),
                "WD_P_rater1": float(
                    row.WD_P_rater1
                ),
                "WD_P_rater2": float(
                    row.WD_P_rater2
                ),
                "WD_soft": float(
                    row.WD_soft
                ),
                "WD_hard_mean": int(
                    row.WD_hard_mean
                ),
                "WD_consensus": (
                    float(row.WD_consensus)
                    if pd.notna(
                        row.WD_consensus
                    )
                    else np.nan
                ),
                "WD_binary_disagreement": int(
                    row.WD_binary_disagreement
                ),
                "WD_probability": probability,
            }

            rows.append(
                record
            )

            if not quiet:
                print(
                    f"  [{i:04d}/{len(work):04d}] "
                    f"{row.sample_id} | "
                    f"raters={row.WD_P_rater1:.1f},"
                    f"{row.WD_P_rater2:.1f} | "
                    f"soft={row.WD_soft:.1f} | "
                    f"p={probability:.3f}",
                    flush=True,
                )

        except Exception as exc:
            if not quiet:
                print(
                    f"  ERROR "
                    f"{getattr(row, 'sample_id', '?')}: "
                    f"{type(exc).__name__}: "
                    f"{exc}",
                    flush=True,
                )

        torch.cuda.empty_cache()

    return rows


def make_curve_subset(
    val_df: pd.DataFrame,
    n: int,
    seed: int,
) -> pd.DataFrame:
    n = min(
        int(n),
        len(val_df),
    )

    if n <= 0:
        return val_df.iloc[:0].copy()

    # Keep all/most hard positives when possible and then fill negatives.
    positives = val_df[
        val_df["WD_hard_mean"] == 1
    ]

    disagreements = val_df[
        val_df["WD_binary_disagreement"] == 1
    ]

    selected_indices = set()

    for pool, desired in [
        (
            positives,
            min(
                len(positives),
                max(1, n // 3),
            ),
        ),
        (
            disagreements,
            min(
                len(disagreements),
                max(1, n // 4),
            ),
        ),
    ]:
        available = pool.loc[
            ~pool.index.isin(
                selected_indices
            )
        ]

        take = min(
            desired,
            len(available),
            n - len(
                selected_indices
            ),
        )

        if take > 0:
            sample = available.sample(
                n=take,
                random_state=(
                    seed
                    + len(
                        selected_indices
                    )
                    + 100
                ),
            )

            selected_indices.update(
                sample.index.tolist()
            )

    remaining_n = (
        n - len(
            selected_indices
        )
    )

    if remaining_n > 0:
        remaining = val_df.loc[
            ~val_df.index.isin(
                selected_indices
            )
        ]

        sample = remaining.sample(
            n=min(
                remaining_n,
                len(remaining),
            ),
            random_state=seed + 909,
        )

        selected_indices.update(
            sample.index.tolist()
        )

    return val_df.loc[
        sorted(selected_indices)
    ].copy()


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------

def save_curves(
    history: List[dict],
    output_dir: Path,
):
    if not history:
        return

    hist = pd.DataFrame(
        history
    )

    hist.to_csv(
        output_dir
        / "learning_curves.csv",
        index=False,
    )

    fig = plt.figure(
        figsize=(7, 4.5)
    )
    ax = fig.add_subplot(111)

    ax.plot(
        hist["optimizer_step"],
        hist["train_loss_recent"],
        marker="o",
    )
    ax.set_xlabel(
        "Optimizer step"
    )
    ax.set_ylabel(
        "Recent weighted BCE loss"
    )
    ax.set_title(
        "WD binary QLoRA training loss"
    )
    ax.grid(
        True,
        alpha=0.25,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir
        / "training_loss_curve.png",
        dpi=180,
    )
    plt.close(fig)

    fig = plt.figure(
        figsize=(7, 4.5)
    )
    ax = fig.add_subplot(111)

    ax.plot(
        hist["optimizer_step"],
        hist["val_AUPRC"],
        marker="o",
        label="AUPRC",
    )

    if "val_AUROC" in hist:
        ax.plot(
            hist["optimizer_step"],
            hist["val_AUROC"],
            marker="o",
            label="AUROC",
        )

    ax.set_xlabel(
        "Optimizer step"
    )
    ax.set_ylabel(
        "Validation metric"
    )
    ax.set_title(
        "WD binary validation curve"
    )
    ax.legend()
    ax.grid(
        True,
        alpha=0.25,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir
        / "validation_auc_curves.png",
        dpi=180,
    )
    plt.close(fig)

    fig = plt.figure(
        figsize=(7, 4.5)
    )
    ax = fig.add_subplot(111)

    ax.plot(
        hist["optimizer_step"],
        hist["val_soft_BCE"],
        marker="o",
    )

    ax.set_xlabel(
        "Optimizer step"
    )
    ax.set_ylabel(
        "Validation soft-label BCE"
    )
    ax.set_title(
        "WD soft-label validation loss"
    )
    ax.grid(
        True,
        alpha=0.25,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir
        / "validation_soft_bce_curve.png",
        dpi=180,
    )
    plt.close(fig)


def save_probability_plot(
    predictions: pd.DataFrame,
    output_path: Path,
    title: str,
):
    if predictions.empty:
        return

    fig = plt.figure(
        figsize=(7, 4.5)
    )
    ax = fig.add_subplot(111)

    groups = []

    labels = []

    for value, label in [
        (0.0, "soft=0"),
        (0.5, "soft=0.5"),
        (1.0, "soft=1"),
    ]:
        values = (
            predictions.loc[
                np.isclose(
                    predictions[
                        "WD_soft"
                    ],
                    value,
                ),
                "WD_probability",
            ]
            .to_numpy()
        )

        if len(values):
            groups.append(
                values
            )
            labels.append(
                label
            )

    if groups:
        ax.boxplot(
            groups,
            tick_labels=labels,
        )

    ax.set_ylabel(
        "Predicted P(clear withdrawal)"
    )
    ax.set_title(title)
    ax.grid(
        True,
        alpha=0.2,
    )
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def parse_pos_weight(
    value: str,
    train_targets: np.ndarray,
) -> float:
    if value.lower() == "auto":
        positive_mass = float(
            train_targets.sum()
        )
        negative_mass = float(
            (1.0 - train_targets)
            .sum()
        )

        if positive_mass <= 0:
            raise RuntimeError(
                "Training target has no positive mass."
            )

        return (
            negative_mass
            / positive_mass
        )

    number = float(value)

    if number <= 0:
        raise ValueError(
            "--pos-weight must be "
            "'auto' or > 0."
        )

    return number


def train(
    args,
    manifest: pd.DataFrame,
):
    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base.seed_everything(
        args.seed
    )

    train_df = manifest[
        manifest["split"] == "train"
    ].copy()

    val_df = manifest[
        manifest["split"] == "val"
    ].copy()

    test_df = manifest[
        manifest["split"] == "test"
    ].copy()

    original_train_n = len(train_df)

    if args.target_mode == "consensus":
        train_df = train_df[
            train_df["WD_consensus"].notna()
        ].copy()

        print(
            "\nCONSENSUS-ONLY TRAINING FILTER"
        )
        print("=" * 72)
        print(
            f"Train before filter: {original_train_n}"
        )
        print(
            f"Train retained:      {len(train_df)}"
        )
        print(
            f"Train excluded:      "
            f"{original_train_n - len(train_df)} "
            "(binary rater disagreements)"
        )

    for name, df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        if df.empty:
            raise RuntimeError(
                f"{name} split is empty."
            )

    if args.target_mode == "soft":
        target_column = "WD_soft"
    elif args.target_mode == "hard":
        target_column = "WD_hard_mean"
    else:
        target_column = "WD_consensus"

    train_targets_np = (
        train_df[target_column]
        .to_numpy(
            dtype=np.float32
        )
    )

    pos_weight = parse_pos_weight(
        args.pos_weight,
        train_targets_np,
    )

    print("\nLOADING QWEN3-VL")
    print("=" * 72)

    # Reuse the already-tested 4-bit + LoRA setup from the pilot helper.
    model, processor, old_head = (
        base.build_model_and_processor(
            args
        )
    )

    del old_head
    gc.collect()
    torch.cuda.empty_cache()

    base_model = (
        model.get_base_model()
    )

    hidden_size = int(
        base_model.config
        .text_config
        .hidden_size
    )

    head = BinaryHead(
        hidden_size=hidden_size,
        dropout=args.head_dropout,
    ).to(
        "cuda",
        dtype=torch.float32,
    )

    cropper = base.PatientCropper(
        yunet_model=Path(
            args.yunet_model
        ),
        output_width=args.frame_width,
    )

    frame_cache = base.FrameCache(
        cache_root=Path(
            args.frame_cache
        ),
        cropper=cropper,
        num_frames=args.num_frames,
    )

    device = torch.device(
        "cuda:0"
    )

    loss_fn = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [pos_weight],
                device=device,
                dtype=torch.float32,
            )
        )
    )

    lora_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_params,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": list(
                    head.parameters()
                ),
                "lr": args.head_learning_rate,
                "weight_decay": args.weight_decay,
            },
        ]
    )

    steps_per_epoch = math.ceil(
        len(train_df)
        / args.grad_accum_steps
    )

    planned_steps = (
        steps_per_epoch
        * args.epochs
    )

    if args.max_train_steps > 0:
        planned_steps = min(
            planned_steps,
            args.max_train_steps,
        )

    print("\nTRAINING PLAN")
    print("=" * 72)
    print(
        "Target mode:",
        args.target_mode,
    )
    print(
        "Target column:",
        target_column,
    )
    print(
        f"Train: {len(train_df)} segments / "
        f"{train_df['patient_id'].nunique()} patients"
    )
    print(
        f"Val:   {len(val_df)} segments / "
        f"{val_df['patient_id'].nunique()} patients"
    )
    print(
        f"Test:  {len(test_df)} segments / "
        f"{test_df['patient_id'].nunique()} patients"
    )
    print(
        "Pooling:",
        args.pooling,
    )
    print(
        f"Frames: {args.num_frames} | "
        f"width: {args.frame_width}"
    )
    print(
        f"Positive BCE weight: "
        f"{pos_weight:.4f}"
    )
    print(
        "Checkpoint criterion: "
        + (
            "consensus-only validation AUPRC"
            if args.target_mode in {"consensus", "soft"}
            else "validation AUPRC"
        )
    )
    print(
        "Planned optimizer steps:",
        planned_steps,
    )

    config = vars(args).copy()

    config.update(
        {
            "target_column": (
                target_column
            ),
            "soft_label_definition": (
                f"(I(rater1>={args.positive_threshold:g})+"
                f"I(rater2>={args.positive_threshold:g}))/2"
            ),
            "hard_label_definition": (
                f"mean(rater1,rater2)>={args.positive_threshold:g}"
            ),
            "binary_target_interpretation": (
                "any/subtle withdrawal evidence (rating >=2)"
                if float(args.positive_threshold) == 2.0
                else "clear/somewhat-salient withdrawal (rating >=3)"
            ),
            "prompt_text": build_binary_prompt(
                args.positive_threshold
            ),
            "resolved_pos_weight": (
                pos_weight
            ),
            "checkpoint_metric": (
                "consensus-only validation AUPRC"
                if args.target_mode in {"consensus", "soft"}
                else "validation hard-label AUPRC"
            ),
            "consensus_training": (
                args.target_mode == "consensus"
            ),
            "soft_training_uses_disagreement_rows": (
                args.target_mode == "soft"
            ),
            "primary_model_selection_population": (
                "binary-consensus validation rows"
                if args.target_mode == "soft"
                else None
            ),
            "consensus_train_definition": (
                "keep only segments where both raters are on the same "
                "binary side of the selected 3RS threshold"
                if args.target_mode == "consensus"
                else None
            ),
        }
    )

    write_json(
        output_dir
        / "run_config.json",
        config,
    )

    best_score = -float("inf")
    best_epoch = None
    best_step = None
    best_lora_state = None
    best_head_state = None
    best_threshold = 0.5
    best_val_metrics = None

    optimizer_step = 0
    accum_counter = 0
    optimizer.zero_grad(
        set_to_none=True
    )

    history = []
    recent_losses = []

    curve_source_df = val_df

    if args.target_mode in {"consensus", "soft"}:
        curve_source_df = val_df[
            val_df["WD_consensus"].notna()
        ].copy()

    curve_val_df = (
        make_curve_subset(
            curve_source_df,
            n=args.curve_val_examples,
            seed=args.seed,
        )
    )

    stop = False

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        print(
            f"\nEPOCH {epoch}/{args.epochs}"
        )
        print("=" * 72)

        model.train()
        head.train()

        epoch_df = train_df.sample(
            frac=1.0,
            random_state=(
                args.seed + epoch
            ),
        ).reset_index(
            drop=True
        )

        running_loss = 0.0
        successful = 0

        for idx, row in enumerate(
            epoch_df.itertuples(index=False),
            start=1,
        ):
            if (
                args.max_train_steps > 0
                and optimizer_step
                >= args.max_train_steps
            ):
                stop = True
                break

            try:
                frames = frame_cache.build(
                    row
                )

                target_value = float(
                    getattr(
                        row,
                        target_column,
                    )
                )

                target = torch.tensor(
                    [[target_value]],
                    device=device,
                    dtype=torch.float32,
                )

                logits = predict_logits(
                    model=model,
                    head=head,
                    processor=processor,
                    frames=frames,
                    device=device,
                    pooling=args.pooling,
                    positive_threshold=args.positive_threshold,
                )

                loss = loss_fn(
                    logits.float(),
                    target,
                )

                (
                    loss
                    / args.grad_accum_steps
                ).backward()

                detached_loss = float(
                    loss.detach().cpu()
                )

                running_loss += (
                    detached_loss
                )
                recent_losses.append(
                    detached_loss
                )
                successful += 1
                accum_counter += 1

                should_step = (
                    accum_counter
                    >= args.grad_accum_steps
                    or idx == len(
                        epoch_df
                    )
                )

                if should_step:
                    trainable = [
                        p
                        for p in model.parameters()
                        if p.requires_grad
                    ] + list(
                        head.parameters()
                    )

                    torch.nn.utils.clip_grad_norm_(
                        trainable,
                        max_norm=args.max_grad_norm,
                    )

                    optimizer.step()

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    optimizer_step += 1
                    accum_counter = 0

                    if (
                        args.eval_every_steps > 0
                        and optimizer_step
                        % args.eval_every_steps
                        == 0
                    ):
                        recent_loss = float(
                            np.mean(
                                recent_losses
                            )
                        )

                        print(
                            "\nPERIODIC VALIDATION "
                            f"@ step "
                            f"{optimizer_step}",
                            flush=True,
                        )

                        curve_rows = evaluate(
                            df=curve_val_df,
                            split_name=(
                                "curve_val"
                            ),
                            model=model,
                            head=head,
                            processor=processor,
                            frame_cache=frame_cache,
                            device=device,
                            args=args,
                            max_examples=None,
                            quiet=True,
                        )

                        curve_metrics = (
                            compute_metrics(
                                curve_rows,
                                threshold=0.5,
                            )
                        )

                        if args.target_mode == "consensus":
                            hard_metrics = (
                                curve_metrics.get(
                                    "consensus_only",
                                    {},
                                )
                            )
                        else:
                            hard_metrics = (
                                curve_metrics.get(
                                    "hard_mean",
                                    {},
                                )
                            )

                        history.append(
                            {
                                "optimizer_step": (
                                    optimizer_step
                                ),
                                "epoch": epoch,
                                "train_loss_recent": (
                                    recent_loss
                                ),
                                "val_AUPRC": (
                                    hard_metrics.get(
                                        "AUPRC",
                                        np.nan,
                                    )
                                ),
                                "val_AUROC": (
                                    hard_metrics.get(
                                        "AUROC",
                                        np.nan,
                                    )
                                ),
                                "val_soft_BCE": (
                                    curve_metrics.get(
                                        "soft_BCE",
                                        np.nan,
                                    )
                                ),
                                "val_prob_std": (
                                    curve_metrics.get(
                                        "prob_std",
                                        np.nan,
                                    )
                                ),
                            }
                        )

                        save_curves(
                            history,
                            output_dir,
                        )

                        print(
                            "  train BCE="
                            f"{recent_loss:.4f} | "
                            "val AUPRC="
                            f"{hard_metrics.get('AUPRC', float('nan')):.4f} | "
                            "val AUROC="
                            f"{hard_metrics.get('AUROC', float('nan')):.4f} | "
                            "val soft BCE="
                            f"{curve_metrics.get('soft_BCE', float('nan')):.4f}",
                            flush=True,
                        )

                        recent_losses = []

                        model.train()
                        head.train()

                probability = float(
                    torch.sigmoid(
                        logits[0, 0]
                    )
                    .detach()
                    .float()
                    .cpu()
                )

                print(
                    f"  epoch={epoch} "
                    f"sample="
                    f"{idx:04d}/"
                    f"{len(epoch_df):04d} "
                    f"opt_step="
                    f"{optimizer_step:04d} "
                    f"loss="
                    f"{detached_loss:.4f} "
                    f"target="
                    f"{target_value:.1f} "
                    f"p={probability:.3f}",
                    flush=True,
                )

            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(
                    set_to_none=True
                )
                accum_counter = 0
                torch.cuda.empty_cache()
                raise

            except Exception as exc:
                print(
                    f"  ERROR "
                    f"{getattr(row, 'sample_id', '?')}: "
                    f"{type(exc).__name__}: "
                    f"{exc}",
                    flush=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )
                accum_counter = 0

            finally:
                gc.collect()
                torch.cuda.empty_cache()

        mean_train_loss = (
            running_loss
            / max(
                successful,
                1,
            )
        )

        print(
            f"\nEpoch {epoch} mean "
            f"training BCE: "
            f"{mean_train_loss:.4f}"
        )

        val_rows = evaluate(
            df=val_df,
            split_name="val",
            model=model,
            head=head,
            processor=processor,
            frame_cache=frame_cache,
            device=device,
            args=args,
            max_examples=(
                args.max_val_examples
            ),
            quiet=False,
        )

        threshold_rows = val_rows

        if args.target_mode in {"consensus", "soft"}:
            threshold_rows = consensus_only_rows(
                val_rows
            )

            if not threshold_rows:
                raise RuntimeError(
                    "No consensus-only validation rows are available for "
                    "threshold selection."
                )

        selected_threshold, threshold_table = (
            choose_threshold(
                threshold_rows
            )
        )

        threshold_table.to_csv(
            output_dir
            / f"val_threshold_search_epoch_{epoch}.csv",
            index=False,
        )

        val_metrics = compute_metrics(
            val_rows,
            threshold=(
                selected_threshold
            ),
        )

        print(
            "\nVAL METRICS"
        )
        print(
            json.dumps(
                clean_json_value(
                    val_metrics
                ),
                indent=2,
            )
        )

        metric_group = (
            "consensus_only"
            if args.target_mode in {"consensus", "soft"}
            else "hard_mean"
        )

        val_auprc = (
            val_metrics
            .get(
                metric_group,
                {},
            )
            .get(
                "AUPRC",
                float("nan"),
            )
        )

        score = (
            float(val_auprc)
            if (
                val_auprc is not None
                and np.isfinite(
                    val_auprc
                )
            )
            else -float("inf")
        )

        checkpoint_dir = (
            output_dir
            / f"checkpoint_epoch_{epoch}"
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.save_pretrained(
            checkpoint_dir
            / "adapter"
        )

        processor.save_pretrained(
            checkpoint_dir
            / "processor"
        )

        torch.save(
            head.state_dict(),
            checkpoint_dir
            / "binary_head.pt",
        )

        write_json(
            checkpoint_dir
            / "metrics.json",
            {
                "epoch": epoch,
                "optimizer_step": (
                    optimizer_step
                ),
                "mean_train_loss": (
                    mean_train_loss
                ),
                "selected_threshold": (
                    selected_threshold
                ),
                "val_metrics": (
                    val_metrics
                ),
            },
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_step = optimizer_step
            best_threshold = (
                selected_threshold
            )
            best_val_metrics = (
                copy.deepcopy(
                    val_metrics
                )
            )

            best_lora_state = {
                key: (
                    value.detach()
                    .cpu()
                    .clone()
                )
                for key, value
                in get_peft_model_state_dict(
                    model
                ).items()
            }

            best_head_state = {
                key: (
                    value.detach()
                    .cpu()
                    .clone()
                )
                for key, value
                in head.state_dict().items()
            }

            best_dir = (
                output_dir / "best"
            )

            best_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            model.save_pretrained(
                best_dir / "adapter"
            )

            processor.save_pretrained(
                best_dir
                / "processor"
            )

            torch.save(
                head.state_dict(),
                best_dir
                / "binary_head.pt",
            )

            write_json(
                best_dir
                / "metrics.json",
                {
                    "epoch": epoch,
                    "optimizer_step": (
                        optimizer_step
                    ),
                    "selected_threshold": (
                        selected_threshold
                    ),
                    "val_AUPRC": score,
                    "val_metrics": (
                        val_metrics
                    ),
                },
            )

            print(
                "New best checkpoint: "
                f"epoch {epoch} | "
                + (
                    "consensus-val AUPRC="
                    if args.target_mode == "consensus"
                    else "val AUPRC="
                )
                + f"{score:.4f} | "
                f"threshold={selected_threshold:.3f}"
            )

        if stop:
            break

    save_curves(
        history,
        output_dir,
    )

    if (
        best_lora_state is None
        or best_head_state is None
    ):
        raise RuntimeError(
            "No best checkpoint captured."
        )

    print(
        "\nRESTORING BEST "
        "VALIDATION-AUPRC CHECKPOINT"
    )
    print("=" * 72)

    set_peft_model_state_dict(
        model,
        best_lora_state,
        adapter_name="default",
    )

    head.load_state_dict(
        best_head_state
    )

    head.to(
        device=device,
        dtype=torch.float32,
    )

    model.eval()
    head.eval()

    print(
        f"Restored epoch {best_epoch}, "
        f"step {best_step}, "
        f"val AUPRC="
        f"{best_score:.4f}, "
        f"threshold="
        f"{best_threshold:.3f}"
    )

    # Save validation predictions from the restored best model.
    best_val_rows = evaluate(
        df=val_df,
        split_name="best_val",
        model=model,
        head=head,
        processor=processor,
        frame_cache=frame_cache,
        device=device,
        args=args,
        max_examples=(
            args.max_val_examples
        ),
        quiet=True,
    )

    best_val_predictions = (
        pd.DataFrame(
            best_val_rows
        )
    )

    best_val_predictions.to_csv(
        output_dir
        / "val_predictions.csv",
        index=False,
    )

    save_probability_plot(
        best_val_predictions,
        output_dir
        / "val_probability_by_soft_label.png",
        "Validation probability by rater-derived soft label",
    )

    print(
        "\nFINAL HELD-OUT TEST"
    )
    print("=" * 72)

    test_rows = evaluate(
        df=test_df,
        split_name="test",
        model=model,
        head=head,
        processor=processor,
        frame_cache=frame_cache,
        device=device,
        args=args,
        max_examples=(
            args.max_test_examples
        ),
        quiet=False,
    )

    test_predictions = (
        pd.DataFrame(
            test_rows
        )
    )

    test_predictions.to_csv(
        output_dir
        / "test_predictions.csv",
        index=False,
    )

    test_metrics = compute_metrics(
        test_rows,
        threshold=best_threshold,
    )

    print(
        "\nTEST METRICS"
    )
    print(
        json.dumps(
            clean_json_value(
                test_metrics
            ),
            indent=2,
        )
    )

    save_probability_plot(
        test_predictions,
        output_dir
        / "test_probability_by_soft_label.png",
        "Test probability by rater-derived soft label",
    )

    patient_rows = []

    for patient_id, group in (
        test_predictions.groupby(
            "patient_id"
        )
    ):
        rows_group = (
            group.to_dict(
                orient="records"
            )
        )

        patient_metrics = (
            compute_metrics(
                rows_group,
                threshold=(
                    best_threshold
                ),
            )
        )

        hard = (
            patient_metrics.get(
                "hard_mean",
                {},
            )
        )

        patient_rows.append(
            {
                "patient_id": (
                    str(patient_id)
                ),
                "n": len(group),
                "positive_n": (
                    hard.get(
                        "positive_n",
                        0,
                    )
                ),
                "AUPRC": hard.get(
                    "AUPRC",
                    np.nan,
                ),
                "AUROC": hard.get(
                    "AUROC",
                    np.nan,
                ),
                "precision": hard.get(
                    "precision",
                    np.nan,
                ),
                "recall": hard.get(
                    "recall",
                    np.nan,
                ),
                "f1": hard.get(
                    "f1",
                    np.nan,
                ),
                "prob_mean": (
                    group[
                        "WD_probability"
                    ].mean()
                ),
                "prob_std": (
                    group[
                        "WD_probability"
                    ].std(
                        ddof=0
                    )
                ),
            }
        )

    pd.DataFrame(
        patient_rows
    ).to_csv(
        output_dir
        / "test_patient_metrics.csv",
        index=False,
    )

    write_json(
        output_dir
        / "final_summary.json",
        {
            "target_mode": (
                args.target_mode
            ),
            "best_epoch": (
                best_epoch
            ),
            "best_optimizer_step": (
                best_step
            ),
            "best_val_AUPRC": (
                best_score
            ),
            "best_val_metric_population": (
                "consensus_only"
                if args.target_mode == "consensus"
                else "all_hard_mean"
            ),
            "selected_probability_threshold": (
                best_threshold
            ),
            "best_val_metrics": (
                best_val_metrics
            ),
            "test_metrics": (
                test_metrics
            ),
        },
    )

    print("\nFINISHED")
    print(
        "Output:",
        output_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_parser():
    p = argparse.ArgumentParser(
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
        description=(
            "Qwen3-VL WD_P hard/soft "
            "binary QLoRA."
        ),
    )

    p.add_argument(
        "--labels-csv",
        required=True,
    )

    p.add_argument(
        "--manifest",
        required=True,
        help=(
            "Existing patient-disjoint "
            "WD-only manifest."
        ),
    )

    p.add_argument(
        "--target-mode",
        choices=[
            "soft",
            "hard",
            "consensus",
        ],
        default="soft",
        help=(
            "soft: 0/0.5/1 from the two raters; "
            "hard: threshold the two-rater mean; "
            "consensus: exclude binary disagreements from training and "
            "use only unanimous 0/1 labels."
        ),
    )

    p.add_argument(
        "--positive-threshold",
        type=float,
        choices=[2.0, 3.0],
        default=2.0,
        help=(
            "3RS binary boundary. "
            "2 = any/subtle withdrawal evidence versus rating 1; "
            "3 = clear/somewhat-salient-or-greater withdrawal."
        ),
    )

    p.add_argument(
        "--video-root",
        default=(
            r"C:\Data\Sequence_model"
            r"\Memopsy_videos\CONVERTED"
        ),
    )

    p.add_argument(
        "--yunet-model",
        default=(
            r".\models\face_detection_yunet"
            r"\face_detection_yunet_2026may.onnx"
        ),
    )

    p.add_argument(
        "--frame-cache",
        default=(
            r".\output\qwen3vl_rupture_finetune_16f_stable"
            r"\frame_cache_16"
        ),
    )

    p.add_argument(
        "--output-dir",
        default=(
            r".\output\qwen3vl_wd_soft_binary"
            r"\run1"
        ),
    )

    p.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
    )

    p.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=[
            "sdpa",
            "eager",
            "flash_attention_2",
        ],
    )

    p.add_argument(
        "--no-4bit",
        action="store_true",
    )

    p.add_argument(
        "--num-frames",
        type=int,
        default=16,
    )

    p.add_argument(
        "--frame-width",
        type=int,
        default=224,
    )

    p.add_argument(
        "--pooling",
        choices=[
            "mean_all",
            "mean_image",
        ],
        default="mean_all",
    )

    p.add_argument(
        "--lora-r",
        type=int,
        default=4,
    )

    p.add_argument(
        "--lora-alpha",
        type=int,
        default=8,
    )

    p.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--head-dropout",
        type=float,
        default=0.10,
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    p.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
    )

    p.add_argument(
        "--head-learning-rate",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
    )

    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--pos-weight",
        default="auto",
        help=(
            "'auto' balances positive "
            "target mass in the train split, "
            "or provide a numeric value."
        ),
    )

    p.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help=(
            "0 = train all requested epochs."
        ),
    )

    p.add_argument(
        "--max-val-examples",
        type=int,
        default=0,
    )

    p.add_argument(
        "--max-test-examples",
        type=int,
        default=0,
    )

    p.add_argument(
        "--eval-every-steps",
        type=int,
        default=20,
        help=(
            "0 disables periodic "
            "validation curves."
        ),
    )

    p.add_argument(
        "--curve-val-examples",
        type=int,
        default=40,
    )

    p.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Build/print hard + soft "
            "targets without loading Qwen."
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return p


def main():
    args = make_parser().parse_args()

    if args.max_val_examples <= 0:
        args.max_val_examples = None

    if args.max_test_examples <= 0:
        args.max_test_examples = None

    output_dir = Path(
        args.output_dir
    )

    print(
        "QWEN3-VL WD_P "
        "HARD / SOFT BINARY QLORA"
    )
    print("=" * 72)
    print(
        "Target mode:",
        args.target_mode,
    )
    print(
        "3RS positive threshold:",
        args.positive_threshold,
    )
    print(
        "Target interpretation:",
        (
            "rating >=2: any/subtle withdrawal evidence"
            if float(args.positive_threshold) == 2.0
            else "rating >=3: clear/somewhat-salient withdrawal"
        ),
    )
    print(
        "Manifest:",
        args.manifest,
    )
    print(
        "Labels:",
        args.labels_csv,
    )

    manifest = merge_targets_with_manifest(
        manifest_path=Path(
            args.manifest
        ),
        labels_csv=Path(
            args.labels_csv
        ),
        threshold=(
            args.positive_threshold
        ),
        output_dir=output_dir,
    )

    if args.prepare_only:
        if args.target_mode == "consensus":
            print(
                "\nCONSENSUS-ONLY COUNTS"
            )
            print("=" * 72)

            for split in [
                "train",
                "val",
                "test",
            ]:
                sub = manifest[
                    manifest["split"] == split
                ]
                consensus = sub[
                    sub["WD_consensus"].notna()
                ]

                negatives = int(
                    (consensus["WD_consensus"] == 0).sum()
                )
                positives = int(
                    (consensus["WD_consensus"] == 1).sum()
                )

                print(
                    f"{split:5s}: "
                    f"retained={len(consensus):4d} | "
                    f"negative={negatives:4d} | "
                    f"positive={positives:4d} | "
                    f"excluded_disagreement="
                    f"{len(sub) - len(consensus):4d}"
                )

        print(
            "\nPREPARE-ONLY complete."
        )
        return

    train(
        args,
        manifest,
    )


if __name__ == "__main__":
    main()