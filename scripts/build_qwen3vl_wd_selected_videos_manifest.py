#!/usr/bin/env python
"""
Build a substantially larger Qwen3-VL WD_P manifest from the annotation CSV.

Optionally restrict the dataset to an explicit planning/selection CSV using
--video-list-csv. When supplied, ONLY filenames listed in that CSV are eligible.

Purpose
-------
The earlier VLM experiments reused a 951-segment manifest. The annotation CSV
contains many more exactly-two-rater WD_P segments. This script builds a new
manifest from ALL eligible exactly-two-rater segments, resolves the source
videos, assigns the patient side from the existing role cache, and optionally
expands patient-side coverage only when the role-cache evidence for that
patient is unanimous.

No Qwen model is loaded.

Expected annotation-only counts for the currently analysed CSV
---------------------------------------------------------------
These are checked at runtime; they are NOT hard-coded:
- ~3,896 exactly-two-rater WD_P segments
- ~19 patients
- ~78 videos
- at threshold >=2:
    both raters <2      -> consensus negative
    one rater >=2       -> disagreement
    both raters >=2     -> consensus positive
- ~2,700 consensus segments before media/role filtering

Outputs
-------
output_dir/
    full_double_rated_targets.csv
        Every exactly-two-rater WD_P segment, before video/role filtering.

    full_media_resolved_manifest.csv
        Exactly-two-rater segments whose video file was found.
        May still contain unresolved patient_side values.

    training_manifest.csv
        All exactly-two-rater segments with BOTH:
            - resolved video_path
            - resolved patient_side
        This is the manifest to use with patient-grouped CV.

    consensus_thr2_manifest.csv
        Convenience subset of training_manifest containing only unanimous
        threshold-2 binary targets.

    unresolved_patient_side.csv
        Media-resolved rows for which patient side could not be established.

    missing_videos.csv
        Annotation rows whose video file could not be resolved.

    role_resolution_by_video.csv
        One row per annotated video showing exact/inferred/unresolved side.

    patient_role_evidence.csv
        Audit of exact role-cache evidence per patient.

    manifest_summary.json
        Counts and coverage diagnostics.

Optional frame cache
--------------------
Use --build-frame-cache to create 16 patient-cropped frames per segment using
the same PatientCropper / FrameCache implementation as the prior experiments.

By default, frame caching is NOT run because it can take a long time.

Safe role inference
-------------------
Exact role-cache entries are always preferred.

For videos missing an exact role-cache entry, the default behaviour is to infer
a patient's side ONLY if all exact role-cache videos available for that patient
agree on the same side. The source is stored as:
    exact_role_cache
or
    patient_unanimous_inference

If a patient has mixed exact left/right evidence, or no exact role evidence,
the side remains unresolved.

Use --no-patient-side-inference to disable even this conservative inference.

This inference should still be audited before final modelling.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import finetune_qwen3vl_rupture_pilot as base


SEGMENT_KEY = [
    "patient_id",
    "session_id",
    "video",
    "segment_id",
]

VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_id(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def normalize_coder(value) -> str:
    text = str(value).strip()
    mapping = {
        "segments Alex": "Alex",
    }
    return mapping.get(text, text)


def parse_hms(value) -> float:
    if pd.isna(value):
        return float("nan")

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()

    # Numeric seconds encoded as string.
    try:
        return float(text)
    except ValueError:
        pass

    parts = text.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return (
            float(h) * 3600.0
            + float(m) * 60.0
            + float(s)
        )

    if len(parts) == 2:
        m, s = parts
        return float(m) * 60.0 + float(s)

    raise ValueError(f"Could not parse time value: {value!r}")


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
    path.write_text(
        json.dumps(
            json_clean(obj),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def safe_sample_id(
    patient_id: str,
    session_id: str,
    segment_id: int,
) -> str:
    raw = f"{patient_id}_{session_id}_seg{int(segment_id):03d}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def load_requested_video_list(
    csv_path: Path,
    video_column: str = "video",
) -> Tuple[set[str], pd.DataFrame]:
    """
    Load the exact set of requested videos from a planning/selection CSV.

    All rows in the specified video column are used. No additional status,
    decision, or scheduling column is applied unless the user creates a
    separately filtered CSV first.
    """
    df = pd.read_csv(csv_path)

    if video_column not in df.columns:
        raise RuntimeError(
            f"Video-list CSV has no column {video_column!r}. "
            f"Available columns: {list(df.columns)}"
        )

    videos = (
        df[video_column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    videos = videos[
        videos.ne("")
    ]

    audit = pd.DataFrame(
        {
            "video": sorted(
                videos.unique().tolist()
            )
        }
    )

    return set(audit["video"].tolist()), audit


# ---------------------------------------------------------------------------
# Build exactly-two-rater targets
# ---------------------------------------------------------------------------

def build_full_two_rater_targets(
    labels_csv: Path,
    threshold: float,
    allowed_videos: Optional[set[str]] = None,
) -> pd.DataFrame:
    raw = pd.read_csv(labels_csv)

    required = [
        "video",
        "patient_id",
        "session_id",
        "coder",
        "segment_id",
        "segment_start",
        "segment_end",
        "WD_P",
    ]

    missing = [
        col for col in required
        if col not in raw.columns
    ]

    if missing:
        raise RuntimeError(
            f"Labels CSV is missing columns: {missing}"
        )

    raw = raw.dropna(
        subset=[
            "video",
            "patient_id",
            "session_id",
            "coder",
            "segment_id",
            "WD_P",
        ]
    ).copy()

    raw["patient_id"] = raw["patient_id"].map(normalize_id)
    raw["session_id"] = raw["session_id"].map(normalize_id)
    raw["video"] = raw["video"].astype(str).str.strip()

    if allowed_videos is not None:
        allowed_videos = {
            str(v).strip()
            for v in allowed_videos
            if str(v).strip()
        }
        raw = raw[
            raw["video"].isin(allowed_videos)
        ].copy()

        if raw.empty:
            raise RuntimeError(
                "No annotation rows remain after applying the requested "
                "video-list filter."
            )

    raw["coder"] = raw["coder"].map(normalize_coder)
    raw["segment_id"] = pd.to_numeric(
        raw["segment_id"],
        errors="coerce",
    )
    raw["WD_P"] = pd.to_numeric(
        raw["WD_P"],
        errors="coerce",
    )

    raw = raw.dropna(
        subset=[
            "segment_id",
            "WD_P",
        ]
    ).copy()

    raw["segment_id"] = raw["segment_id"].astype(int)

    # Verify each physical segment's coder count before selecting exactly 2.
    counts = (
        raw.groupby(
            SEGMENT_KEY,
            dropna=False,
        )
        .agg(
            n_rows=("coder", "size"),
            n_coders=("coder", "nunique"),
        )
        .reset_index()
    )

    exactly_two_keys = counts[
        counts["n_coders"] == 2
    ][SEGMENT_KEY].copy()

    two = raw.merge(
        exactly_two_keys,
        on=SEGMENT_KEY,
        how="inner",
        validate="many_to_one",
    )

    # Current dataset should be two physical rows for exactly-two-coder cases.
    duplicate_check = (
        two.groupby(SEGMENT_KEY)
        .agg(
            n_rows=("coder", "size"),
            n_coders=("coder", "nunique"),
        )
        .reset_index()
    )

    bad = duplicate_check[
        (duplicate_check["n_rows"] != 2)
        | (duplicate_check["n_coders"] != 2)
    ]

    if not bad.empty:
        raise RuntimeError(
            "Some exactly-two-rater segments do not contain exactly two "
            "rows after coder normalization. Inspect the labels first.\n"
            + bad.head(20).to_string(index=False)
        )

    rows = []

    for key, group in two.groupby(
        SEGMENT_KEY,
        sort=True,
        dropna=False,
    ):
        group = group.sort_values(
            "coder"
        ).reset_index(drop=True)

        r1 = group.iloc[0]
        r2 = group.iloc[1]

        score1 = float(r1["WD_P"])
        score2 = float(r2["WD_P"])

        bin1 = int(score1 >= threshold)
        bin2 = int(score2 >= threshold)

        if bin1 == bin2:
            consensus = float(bin1)
        else:
            consensus = np.nan

        start_sec = parse_hms(r1["segment_start"])
        end_inclusive = parse_hms(r1["segment_end"])

        # Annotation convention is usually 00:00:00 -> 00:00:59.
        # Convert to half-open [start, end) timing.
        end_sec = end_inclusive + 1.0

        if not np.isfinite(start_sec):
            start_sec = float((int(key[3]) - 1) * 60)

        if (
            not np.isfinite(end_sec)
            or end_sec <= start_sec
        ):
            end_sec = start_sec + 60.0

        patient_id = normalize_id(key[0])
        session_id = normalize_id(key[1])
        video = str(key[2]).strip()
        segment_id = int(key[3])

        rows.append(
            {
                "sample_id": safe_sample_id(
                    patient_id,
                    session_id,
                    segment_id,
                ),
                "patient_id": patient_id,
                "session_id": session_id,
                "video": video,
                "segment_id": segment_id,
                "segment_start": r1["segment_start"],
                "segment_end": r1["segment_end"],
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "coder_1": str(r1["coder"]),
                "coder_2": str(r2["coder"]),
                "WD_P_rater1": score1,
                "WD_P_rater2": score2,
                "WD_P_mean": float(
                    (score1 + score2) / 2.0
                ),
                "WD_P_min": float(min(score1, score2)),
                "WD_P_max": float(max(score1, score2)),
                "WD_P_absolute_difference": float(
                    abs(score1 - score2)
                ),
                "WD_binary_rater1": bin1,
                "WD_binary_rater2": bin2,
                "WD_soft": float(
                    (bin1 + bin2) / 2.0
                ),
                "WD_consensus": consensus,
                "WD_binary_disagreement": int(
                    bin1 != bin2
                ),
                # Placeholder only. Patient-grouped CV overwrites this.
                "split": "train",
            }
        )

    targets = pd.DataFrame(rows)

    if targets.empty:
        raise RuntimeError(
            "No exactly-two-rater WD_P segments were found."
        )

    if targets["sample_id"].duplicated().any():
        dup = targets[
            targets["sample_id"].duplicated(
                keep=False
            )
        ]
        raise RuntimeError(
            "sample_id collision detected.\n"
            + dup.head(20).to_string(index=False)
        )

    return targets


# ---------------------------------------------------------------------------
# Media + patient side resolution
# ---------------------------------------------------------------------------

def index_videos(
    video_root: Path,
) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    if not video_root.exists():
        raise FileNotFoundError(
            f"Video root does not exist: {video_root}"
        )

    by_name: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}

    paths: List[Path] = []

    for ext in VIDEO_EXTENSIONS:
        paths.extend(
            video_root.rglob(f"*{ext}")
        )
        paths.extend(
            video_root.rglob(f"*{ext.upper()}")
        )

    for path in sorted(set(paths)):
        key = path.name.lower()

        if key in by_name:
            duplicates.setdefault(
                key,
                [by_name[key]],
            ).append(path)
        else:
            by_name[key] = path

    return by_name, duplicates


def exact_role_side(
    role_cache: Dict[str, dict],
    video_name: str,
) -> Optional[str]:
    return base.role_side_for_video(
        role_cache,
        video_name,
    )


def patient_role_evidence(
    targets: pd.DataFrame,
    role_cache: Dict[str, dict],
) -> pd.DataFrame:
    # One row per annotated video first.
    video_rows = (
        targets[
            [
                "patient_id",
                "video",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    video_rows["exact_patient_side"] = (
        video_rows["video"]
        .map(
            lambda name: exact_role_side(
                role_cache,
                str(name),
            )
            or ""
        )
    )

    rows = []

    for patient_id, group in video_rows.groupby(
        "patient_id",
        sort=True,
    ):
        known = group[
            group["exact_patient_side"].isin(
                ["left", "right"]
            )
        ]

        sides = sorted(
            known["exact_patient_side"]
            .unique()
            .tolist()
        )

        if len(sides) == 1:
            unanimous_side = sides[0]
        else:
            unanimous_side = ""

        rows.append(
            {
                "patient_id": str(patient_id),
                "annotated_videos": int(
                    group["video"].nunique()
                ),
                "exact_role_videos": int(
                    len(known)
                ),
                "exact_left_videos": int(
                    (
                        known["exact_patient_side"]
                        == "left"
                    ).sum()
                ),
                "exact_right_videos": int(
                    (
                        known["exact_patient_side"]
                        == "right"
                    ).sum()
                ),
                "exact_sides_seen": "|".join(sides),
                "unanimous_inferred_side": (
                    unanimous_side
                ),
                "safe_for_patient_inference": bool(
                    len(sides) == 1
                    and len(known) >= 1
                ),
            }
        )

    return pd.DataFrame(rows)


def resolve_manifest(
    targets: pd.DataFrame,
    video_root: Path,
    role_cache_path: Path,
    infer_patient_side: bool,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    video_index, duplicates = index_videos(
        video_root
    )

    if duplicates:
        print(
            f"WARNING: {len(duplicates)} duplicated video basenames found. "
            "The first sorted path will be used."
        )

    role_cache = base.load_role_cache(
        role_cache_path
    )

    evidence = patient_role_evidence(
        targets,
        role_cache,
    )

    patient_inference = {
        str(row.patient_id): str(
            row.unanimous_inferred_side
        )
        for row in evidence.itertuples(
            index=False
        )
        if bool(
            row.safe_for_patient_inference
        )
        and str(
            row.unanimous_inferred_side
        )
        in {"left", "right"}
    }

    video_audit_rows = []

    for row in (
        targets[
            [
                "patient_id",
                "video",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "patient_id",
                "video",
            ]
        )
        .itertuples(index=False)
    ):
        video = str(row.video)
        patient_id = str(row.patient_id)

        path = video_index.get(
            Path(video).name.lower()
        )

        side = exact_role_side(
            role_cache,
            video,
        )

        if side in {"left", "right"}:
            source = "exact_role_cache"
        elif (
            infer_patient_side
            and patient_id in patient_inference
        ):
            side = patient_inference[
                patient_id
            ]
            source = (
                "patient_unanimous_inference"
            )
        else:
            side = ""
            source = "unresolved"

        video_audit_rows.append(
            {
                "patient_id": patient_id,
                "video": video,
                "video_found": bool(
                    path is not None
                ),
                "video_path": (
                    str(path)
                    if path is not None
                    else ""
                ),
                "patient_side": side,
                "patient_side_source": source,
            }
        )

    video_audit = pd.DataFrame(
        video_audit_rows
    )

    resolved = targets.merge(
        video_audit,
        on=[
            "patient_id",
            "video",
        ],
        how="left",
        validate="many_to_one",
    )

    missing_videos = resolved[
        ~resolved["video_found"].fillna(False)
    ].copy()

    media_resolved = resolved[
        resolved["video_found"].fillna(False)
    ].copy()

    unresolved_side = media_resolved[
        ~media_resolved["patient_side"].isin(
            ["left", "right"]
        )
    ].copy()

    training_manifest = media_resolved[
        media_resolved["patient_side"].isin(
            ["left", "right"]
        )
    ].copy()

    # Stable sort helps reproducibility.
    sort_cols = [
        "patient_id",
        "session_id",
        "video",
        "segment_id",
    ]

    for frame in (
        missing_videos,
        media_resolved,
        unresolved_side,
        training_manifest,
    ):
        frame.sort_values(
            sort_cols,
            inplace=True,
        )
        frame.reset_index(
            drop=True,
            inplace=True,
        )

    return (
        training_manifest,
        media_resolved,
        missing_videos,
        unresolved_side,
        video_audit,
        evidence,
    )


# ---------------------------------------------------------------------------
# Optional frame-cache expansion
# ---------------------------------------------------------------------------

def build_frame_cache(
    manifest: pd.DataFrame,
    cache_root: Path,
    yunet_model: Path,
    num_frames: int,
    frame_width: int,
    scope: str,
    threshold: float,
    output_dir: Path,
) -> dict:
    if scope == "consensus":
        to_cache = manifest[
            manifest["WD_consensus"].notna()
        ].copy()
    elif scope == "all":
        to_cache = manifest.copy()
    else:
        raise ValueError(
            f"Unknown cache scope: {scope}"
        )

    print("\nBUILDING PATIENT FRAME CACHE")
    print("=" * 72)
    print("Scope:", scope)
    print("Rows:", len(to_cache))
    print("Frames per segment:", num_frames)
    print("Frame width:", frame_width)
    print("Cache root:", cache_root)

    cropper = base.PatientCropper(
        yunet_model=yunet_model,
        output_width=frame_width,
    )

    cache = base.FrameCache(
        cache_root=cache_root,
        cropper=cropper,
        num_frames=num_frames,
    )

    failures = []
    success = 0

    for i, row in enumerate(
        to_cache.itertuples(index=False),
        start=1,
    ):
        try:
            cache.build(row)
            success += 1
        except Exception as exc:
            failures.append(
                {
                    "sample_id": str(row.sample_id),
                    "patient_id": str(row.patient_id),
                    "session_id": str(row.session_id),
                    "video": str(row.video),
                    "segment_id": int(row.segment_id),
                    "video_path": str(row.video_path),
                    "patient_side": str(
                        row.patient_side
                    ),
                    "error": repr(exc),
                }
            )

        if (
            i == 1
            or i % 50 == 0
            or i == len(to_cache)
        ):
            print(
                f"[{i:4d}/{len(to_cache):4d}] "
                f"success={success} "
                f"failures={len(failures)}"
            )

    failure_df = pd.DataFrame(
        failures
    )

    failure_df.to_csv(
        output_dir
        / "frame_cache_failures.csv",
        index=False,
    )

    return {
        "scope": scope,
        "requested_segments": int(
            len(to_cache)
        ),
        "cached_success": int(success),
        "cached_failures": int(
            len(failures)
        ),
        "cache_root": str(cache_root),
        "num_frames": int(num_frames),
        "frame_width": int(frame_width),
        "positive_threshold": float(
            threshold
        ),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def make_summary(
    targets: pd.DataFrame,
    media_resolved: pd.DataFrame,
    training_manifest: pd.DataFrame,
    missing_videos: pd.DataFrame,
    unresolved_side: pd.DataFrame,
    video_audit: pd.DataFrame,
    threshold: float,
    inference_enabled: bool,
) -> dict:
    consensus_all = targets[
        targets["WD_consensus"].notna()
    ]

    consensus_training = training_manifest[
        training_manifest[
            "WD_consensus"
        ].notna()
    ]

    exact_rows = training_manifest[
        training_manifest[
            "patient_side_source"
        ]
        == "exact_role_cache"
    ]

    inferred_rows = training_manifest[
        training_manifest[
            "patient_side_source"
        ]
        == "patient_unanimous_inference"
    ]

    return {
        "positive_threshold": float(
            threshold
        ),
        "patient_side_inference_enabled": bool(
            inference_enabled
        ),
        "annotation_pool": {
            "exactly_two_rater_segments": int(
                len(targets)
            ),
            "patients": int(
                targets["patient_id"].nunique()
            ),
            "videos": int(
                targets["video"].nunique()
            ),
            "consensus_segments": int(
                len(consensus_all)
            ),
            "consensus_negative": int(
                (
                    consensus_all[
                        "WD_consensus"
                    ]
                    == 0
                ).sum()
            ),
            "consensus_positive": int(
                (
                    consensus_all[
                        "WD_consensus"
                    ]
                    == 1
                ).sum()
            ),
            "binary_disagreements": int(
                targets[
                    "WD_binary_disagreement"
                ].sum()
            ),
        },
        "media_resolution": {
            "resolved_segments": int(
                len(media_resolved)
            ),
            "missing_video_segments": int(
                len(missing_videos)
            ),
            "resolved_videos": int(
                media_resolved[
                    "video"
                ].nunique()
            ),
            "missing_videos": int(
                missing_videos[
                    "video"
                ].nunique()
            ),
        },
        "role_resolution": {
            "training_manifest_segments": int(
                len(training_manifest)
            ),
            "training_manifest_patients": int(
                training_manifest[
                    "patient_id"
                ].nunique()
            ),
            "training_manifest_videos": int(
                training_manifest[
                    "video"
                ].nunique()
            ),
            "exact_role_cache_segments": int(
                len(exact_rows)
            ),
            "patient_unanimous_inferred_segments": int(
                len(inferred_rows)
            ),
            "unresolved_side_segments": int(
                len(unresolved_side)
            ),
            "exact_role_cache_videos": int(
                (
                    video_audit[
                        "patient_side_source"
                    ]
                    == "exact_role_cache"
                ).sum()
            ),
            "patient_unanimous_inferred_videos": int(
                (
                    video_audit[
                        "patient_side_source"
                    ]
                    == "patient_unanimous_inference"
                ).sum()
            ),
            "unresolved_side_videos": int(
                (
                    video_audit[
                        "patient_side_source"
                    ]
                    == "unresolved"
                ).sum()
            ),
        },
        "usable_consensus_pool": {
            "segments": int(
                len(consensus_training)
            ),
            "patients": int(
                consensus_training[
                    "patient_id"
                ].nunique()
            ),
            "videos": int(
                consensus_training[
                    "video"
                ].nunique()
            ),
            "negative": int(
                (
                    consensus_training[
                        "WD_consensus"
                    ]
                    == 0
                ).sum()
            ),
            "positive": int(
                (
                    consensus_training[
                        "WD_consensus"
                    ]
                    == 1
                ).sum()
            ),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build the full exactly-two-rater WD_P "
            "Qwen3-VL manifest."
        )
    )

    p.add_argument(
        "--labels-csv",
        required=True,
    )

    p.add_argument(
        "--video-list-csv",
        default=None,
        help=(
            "Optional CSV restricting the dataset to only the videos "
            "listed in a column (default column: video)."
        ),
    )

    p.add_argument(
        "--video-column",
        default="video",
        help=(
            "Column in --video-list-csv containing video filenames."
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
        "--role-cache",
        default=(
            r".\output\qwen3vl_visual_experiment_v5"
            r"\patient_role_cache.json"
        ),
    )

    p.add_argument(
        "--output-dir",
        default=(
            r".\output\qwen3vl_wd_full_dataset_thr2"
        ),
    )

    p.add_argument(
        "--positive-threshold",
        type=float,
        choices=[
            2.0,
            3.0,
        ],
        default=2.0,
    )

    p.add_argument(
        "--no-patient-side-inference",
        action="store_true",
        help=(
            "Use ONLY exact role-cache video entries. "
            "Do not infer missing video sides even when a patient's "
            "known role-cache videos are unanimous."
        ),
    )

    p.add_argument(
        "--build-frame-cache",
        action="store_true",
        help=(
            "After building the manifest, also build patient-cropped "
            "frame cache. Qwen itself is not loaded."
        ),
    )

    p.add_argument(
        "--cache-scope",
        choices=[
            "consensus",
            "all",
        ],
        default="consensus",
        help=(
            "consensus = cache only unanimous threshold labels; "
            "all = cache all exactly-two-rater rows in training_manifest."
        ),
    )

    p.add_argument(
        "--frame-cache",
        default=(
            r".\output\qwen3vl_wd_full_dataset_thr2"
            r"\frame_cache_16"
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
        "--num-frames",
        type=int,
        default=16,
    )

    p.add_argument(
        "--frame-width",
        type=int,
        default=224,
    )

    return p


def main():
    args = make_parser().parse_args()

    labels_csv = Path(
        args.labels_csv
    )

    requested_videos = None
    requested_video_audit = None

    if args.video_list_csv:
        requested_videos, requested_video_audit = (
            load_requested_video_list(
                Path(args.video_list_csv),
                video_column=str(args.video_column),
            )
        )

    video_root = Path(
        args.video_root
    )

    role_cache_path = Path(
        args.role_cache
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    infer_patient_side = not bool(
        args.no_patient_side_inference
    )

    print(
        "BUILDING FULL QWEN3-VL WD_P DATASET"
    )
    print("=" * 72)
    print("Labels:", labels_csv)

    if args.video_list_csv:
        print(
            "Video-list CSV:",
            args.video_list_csv,
        )
        print(
            "Requested unique videos:",
            len(requested_videos),
        )
    else:
        print(
            "Video-list CSV: none "
            "(all annotation videos are eligible)"
        )

    print("Video root:", video_root)
    print("Role cache:", role_cache_path)
    print(
        "Patient-side inference:",
        (
            "enabled (unanimous exact evidence only)"
            if infer_patient_side
            else "disabled"
        ),
    )
    print(
        "Binary threshold:",
        args.positive_threshold,
    )

    targets = build_full_two_rater_targets(
        labels_csv=labels_csv,
        threshold=float(
            args.positive_threshold
        ),
        allowed_videos=requested_videos,
    )

    targets.to_csv(
        output_dir
        / "full_double_rated_targets.csv",
        index=False,
    )

    video_filter_summary = None

    if requested_video_audit is not None:
        requested_video_audit.to_csv(
            output_dir / "requested_video_list.csv",
            index=False,
        )

        target_videos = set(
            targets["video"]
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        requested_with_two_rater = (
            requested_video_audit[
                requested_video_audit["video"].isin(
                    target_videos
                )
            ].copy()
        )

        requested_without_two_rater = (
            requested_video_audit[
                ~requested_video_audit["video"].isin(
                    target_videos
                )
            ].copy()
        )

        requested_with_two_rater.to_csv(
            output_dir
            / "requested_videos_with_two_rater_wd.csv",
            index=False,
        )

        requested_without_two_rater.to_csv(
            output_dir
            / "requested_videos_without_two_rater_wd.csv",
            index=False,
        )

        video_filter_summary = {
            "source_csv": str(args.video_list_csv),
            "video_column": str(args.video_column),
            "requested_unique_videos": int(
                len(requested_video_audit)
            ),
            "requested_videos_with_exactly_two_rater_wd": int(
                len(requested_with_two_rater)
            ),
            "requested_videos_without_exactly_two_rater_wd": int(
                len(requested_without_two_rater)
            ),
        }

    print("\nANNOTATION POOL")
    print("=" * 72)
    print(
        "Exactly-two-rater segments:",
        len(targets),
    )
    print(
        "Patients:",
        targets[
            "patient_id"
        ].nunique(),
    )
    print(
        "Videos:",
        targets[
            "video"
        ].nunique(),
    )
    print(
        "Soft-label distribution:",
        targets[
            "WD_soft"
        ]
        .value_counts()
        .sort_index()
        .to_dict(),
    )

    (
        training_manifest,
        media_resolved,
        missing_videos,
        unresolved_side,
        video_audit,
        evidence,
    ) = resolve_manifest(
        targets=targets,
        video_root=video_root,
        role_cache_path=role_cache_path,
        infer_patient_side=infer_patient_side,
    )

    media_resolved.to_csv(
        output_dir
        / "full_media_resolved_manifest.csv",
        index=False,
    )

    training_manifest.to_csv(
        output_dir
        / "training_manifest.csv",
        index=False,
    )

    training_manifest[
        training_manifest[
            "WD_consensus"
        ].notna()
    ].to_csv(
        output_dir
        / (
            f"consensus_thr"
            f"{int(args.positive_threshold)}"
            f"_manifest.csv"
        ),
        index=False,
    )

    missing_videos.to_csv(
        output_dir
        / "missing_videos.csv",
        index=False,
    )

    unresolved_side.to_csv(
        output_dir
        / "unresolved_patient_side.csv",
        index=False,
    )

    video_audit.to_csv(
        output_dir
        / "role_resolution_by_video.csv",
        index=False,
    )

    evidence.to_csv(
        output_dir
        / "patient_role_evidence.csv",
        index=False,
    )

    summary = make_summary(
        targets=targets,
        media_resolved=media_resolved,
        training_manifest=training_manifest,
        missing_videos=missing_videos,
        unresolved_side=unresolved_side,
        video_audit=video_audit,
        threshold=float(
            args.positive_threshold
        ),
        inference_enabled=infer_patient_side,
    )

    if video_filter_summary is not None:
        summary["video_filter"] = (
            video_filter_summary
        )

    frame_cache_summary = None

    if args.build_frame_cache:
        frame_cache_summary = build_frame_cache(
            manifest=training_manifest,
            cache_root=Path(
                args.frame_cache
            ),
            yunet_model=Path(
                args.yunet_model
            ),
            num_frames=int(
                args.num_frames
            ),
            frame_width=int(
                args.frame_width
            ),
            scope=str(
                args.cache_scope
            ),
            threshold=float(
                args.positive_threshold
            ),
            output_dir=output_dir,
        )

        summary[
            "frame_cache"
        ] = frame_cache_summary

    write_json(
        output_dir
        / "manifest_summary.json",
        summary,
    )

    print("\nFINAL SUMMARY")
    print("=" * 72)
    print(
        json.dumps(
            json_clean(summary),
            indent=2,
        )
    )

    print("\nKEY FILE FOR NEXT CV EXPERIMENT")
    print("=" * 72)
    print(
        output_dir
        / "training_manifest.csv"
    )

    if len(unresolved_side):
        print(
            "\nIMPORTANT: patient-side unresolved rows remain."
        )
        print(
            "Review:",
            output_dir
            / "unresolved_patient_side.csv",
        )

    if len(missing_videos):
        print(
            "\nIMPORTANT: some annotation videos were not found."
        )
        print(
            "Review:",
            output_dir
            / "missing_videos.csv",
        )

    print("\nDONE")


if __name__ == "__main__":
    main()