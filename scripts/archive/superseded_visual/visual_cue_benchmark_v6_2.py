#!/usr/bin/env python
"""
V6.2 — Visual Cue Gold Standard + Perception Benchmark
======================================================

This script does TWO things:

1) PREPARE
   Creates a manual cue-level gold-standard CSV and a cue codebook for the
   diagnostic psychotherapy clips.

2) EVALUATE
   Compares existing Qwen3-VL-8B V5 and InternVL3.5-38B V6/V6.1 literal
   behavior outputs against the manually completed gold standard.

IMPORTANT
---------
This script intentionally evaluates VISUAL PERCEPTION only.
It does NOT use WD_P / CF_P / rupture labels.
It does NOT decide whether a visual cue is clinically meaningful.

The manual gold standard should contain literal observable events such as:
    head pitches downward
    right hand contacts cheek
    lips press together
    shoulders lift then drop
NOT:
    withdrawal
    disengagement
    confrontation
    resistance
    rupture

Recommended workflow
--------------------
A. Create template:
    python visual_cue_benchmark_v6_2.py prepare ...

B. Manually fill visual_cue_gold_standard.csv.

C. Evaluate:
    python visual_cue_benchmark_v6_2.py evaluate ...

Outputs include:
    cue_metrics_overall.csv
    cue_metrics_by_category.csv
    cue_metrics_by_segment.csv
    model_detection_events.csv
    cue_match_details.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Manual annotation vocabulary
# ---------------------------------------------------------------------

CODEBOOK = [
    # Head / face orientation
    ("head_pitch_down", "Head angle moves or remains visibly downward.", "head_down"),
    ("head_pitch_up", "Head angle moves visibly upward.", "other"),
    ("head_yaw_left", "Head turns toward image-left.", "head_turn"),
    ("head_yaw_right", "Head turns toward image-right.", "head_turn"),
    ("head_turn", "Visible head turn; exact image direction uncertain/not coded.", "head_turn"),
    ("negative_head_shake", "Repeated side-to-side head movement consistent with a head shake; describe movement only.", "head_shake"),
    ("abrupt_head_turn", "Rapid visible head turn.", "head_turn"),

    # Hand-to-face
    ("hand_to_eye_region", "Hand contacts or passes over eye/upper-cheek region.", "hand_eye"),
    ("hand_to_cheek", "Hand contacts cheek.", "face_touch"),
    ("hand_to_chin", "Hand contacts chin/jaw.", "face_touch"),
    ("hand_to_mouth", "Hand contacts/covers mouth area.", "face_touch"),
    ("hand_to_forehead", "Hand contacts forehead/eyebrow region.", "face_touch"),
    ("hand_to_face_unspecified", "Hand contacts face; exact region uncertain.", "face_touch"),
    ("hand_away_from_face", "Hand visibly moves away from face after contact.", "face_touch"),

    # Face / mouth
    ("lips_pressed", "Lips visibly press/compress together.", "mouth_tension"),
    ("mouth_tension", "Visible lower-face or mouth tension not better specified.", "mouth_tension"),
    ("crying_visible", "Visible crying behavior; annotate only if visually observable.", "crying"),
    ("tear_visible", "Visible tear/tear track.", "crying"),
    ("smile", "Visible smile.", "other"),

    # Shoulder / body
    ("shoulders_elevate", "One/both shoulders visibly elevate.", "shoulder"),
    ("shoulders_lower", "One/both shoulders visibly lower.", "shoulder"),
    ("shoulder_lift_drop", "Shoulder lift followed by drop in one short sequence.", "shoulder"),
    ("torso_backward", "Torso visibly moves backward.", "torso_backward"),
    ("torso_forward", "Torso visibly moves forward.", "other"),
    ("slump", "Visible postural collapse/slumping.", "slump"),
    ("body_turn_left", "Torso/body turns toward image-left.", "body_turn"),
    ("body_turn_right", "Torso/body turns toward image-right.", "body_turn"),
    ("body_turn", "Visible body turn; exact direction uncertain.", "body_turn"),
    ("arms_cross", "Arms become or remain crossed/closed.", "arms_cross"),
    ("push_away_gesture", "Visible pushing-away/rejecting hand or arm movement; describe literal motion in description.", "push_away"),
    ("movement_reduction", "Marked visible reduction in movement relative to nearby behavior.", "movement_reduction"),

    # Generic fallback
    ("other_literal_visual_event", "Other literal visible event not covered above.", "other"),
]

CODEBOOK_COLUMNS = [
    "behavior_code",
    "definition",
    "evaluation_category",
]

# Existing V5/V6 behavior labels -> evaluation categories.
MODEL_BEHAVIOR_TO_CATEGORY = {
    "head_down": "head_down",
    "face_or_head_away": "head_turn",
    "negative_head_shake": "head_shake",
    "abrupt_head_turn_away": "head_turn",
    "crying_visible": "crying",
    "face_or_tear_wiping": "hand_eye",
    "lip_compression_or_mouth_tension": "mouth_tension",
    "face_or_chin_touching": "face_touch",
    "shoulder_shrug_or_lift_drop": "shoulder",
    "lean_or_pull_backward": "torso_backward",
    "slump_or_collapse": "slump",
    "body_turn_away": "body_turn",
    "arms_become_closed_or_crossed": "arms_cross",
    "pushing_away_or_rejecting_gesture": "push_away",
    "marked_reduction_in_movement": "movement_reduction",
}

GOLD_CODE_TO_CATEGORY = {
    behavior_code: eval_category
    for behavior_code, _, eval_category in CODEBOOK
}


GOLD_COLUMNS = [
    "segment_idx",
    "segment_path",
    "video",
    "patient_id",
    "session_id",
    "segment_id",
    "segment_start_sec",
    "cue_id",
    "start_sec",
    "end_sec",
    "actor",
    "behavior_code",
    "literal_description",
    "visibility",
    "certainty",
    "annotator",
    "notes",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def parse_indices(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def load_json_maybe(value):
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def parse_window_bounds(window: dict, fallback_index: int, fallback_size: float = 15.0) -> Tuple[float, float]:
    if "window_start_global" in window and "window_end_global" in window:
        try:
            return float(window["window_start_global"]), float(window["window_end_global"])
        except Exception:
            pass

    label = str(window.get("window", "") or "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", label)
    if m:
        return float(m.group(1)), float(m.group(2))

    start = float(fallback_index) * fallback_size
    return start, start + fallback_size


def normalize_model_time(raw_t, window_start: float, window_end: float) -> Optional[float]:
    try:
        t = float(raw_t)
    except Exception:
        return None

    duration = window_end - window_start

    # Already global.
    if window_start - 0.6 <= t <= window_end + 0.6:
        return round(max(window_start, min(t, window_end)), 3)

    # Window-relative.
    if -0.1 <= t <= duration + 0.6:
        return round(max(window_start, min(window_start + t, window_end)), 3)

    return round(t, 3)


@dataclass
class Detection:
    model: str
    segment_idx: int
    category: str
    start_sec: float
    end_sec: float
    source_behavior: str
    source_note: str = ""


@dataclass
class GoldEvent:
    segment_idx: int
    cue_id: str
    category: str
    behavior_code: str
    start_sec: float
    end_sec: float
    literal_description: str


# ---------------------------------------------------------------------
# PREPARE
# ---------------------------------------------------------------------

def prepare(args):
    segments = pd.read_csv(args.segments_csv)
    indices = set(parse_indices(args.segment_indices))

    if "segment_idx" not in segments.columns:
        raise ValueError("segments CSV must contain segment_idx")

    segments["segment_idx"] = pd.to_numeric(segments["segment_idx"], errors="raise").astype(int)
    selected = segments[segments["segment_idx"].isin(indices)].copy()
    selected = selected.sort_values("segment_idx")

    missing = indices - set(selected["segment_idx"])
    if missing:
        raise ValueError(f"These segment_idx values were not found: {sorted(missing)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_path = out_dir / "visual_cue_gold_standard.csv"
    codebook_path = out_dir / "visual_cue_codebook.csv"
    checklist_path = out_dir / "manual_annotation_checklist.csv"

    # Do not overwrite a manually edited gold file unless explicitly requested.
    if gold_path.exists() and not args.overwrite:
        print(f"Gold file already exists; leaving unchanged: {gold_path}")
    else:
        rows = []
        for row in selected.itertuples(index=False):
            # One intentionally blank annotation row per segment.
            # Duplicate rows manually when there are multiple events.
            rows.append({
                "segment_idx": int(row.segment_idx),
                "segment_path": getattr(row, "segment_path", ""),
                "video": getattr(row, "video", ""),
                "patient_id": getattr(row, "patient_id", ""),
                "session_id": getattr(row, "session_id", ""),
                "segment_id": getattr(row, "segment_id", ""),
                "segment_start_sec": getattr(row, "segment_start_sec", ""),
                "cue_id": f"{int(row.segment_idx):03d}_001",
                "start_sec": "",
                "end_sec": "",
                "actor": "patient",
                "behavior_code": "",
                "literal_description": "",
                "visibility": "",
                "certainty": "",
                "annotator": args.annotator,
                "notes": "",
            })

        pd.DataFrame(rows, columns=GOLD_COLUMNS).to_csv(
            gold_path, index=False, encoding="utf-8-sig"
        )

    pd.DataFrame(CODEBOOK, columns=CODEBOOK_COLUMNS).to_csv(
        codebook_path, index=False, encoding="utf-8-sig"
    )

    checklist_rows = []
    for row in selected.itertuples(index=False):
        for behavior_code, definition, eval_category in CODEBOOK:
            checklist_rows.append({
                "segment_idx": int(row.segment_idx),
                "video": getattr(row, "video", ""),
                "behavior_code": behavior_code,
                "evaluation_category": eval_category,
                "definition": definition,
                "observed": "",
                "notes": "",
            })

    pd.DataFrame(checklist_rows).to_csv(
        checklist_path, index=False, encoding="utf-8-sig"
    )

    readme = out_dir / "README_visual_cue_gold_standard.txt"
    readme.write_text(
        """V6.2 MANUAL VISUAL CUE GOLD STANDARD

1. Watch each segment manually.
2. Add ONE CSV row per observable event/state.
3. Use GLOBAL time within the 60-second segment (0-60 sec), not original-session time.
4. Duplicate the segment row when multiple events are present.
5. cue_id must be unique, e.g. 010_001, 010_002, 010_003.
6. Use behavior_code values from visual_cue_codebook.csv.
7. Keep descriptions literal:
      GOOD: "right hand moves to cheek and then away"
      GOOD: "head remains angled downward for ~6 sec"
      BAD:  "patient withdraws"
      BAD:  "patient is resistant"
8. visibility: good | partial | poor
9. certainty: clear | possible | ambiguous
10. Do NOT look at VLM outputs while creating the gold standard if possible.
    That reduces confirmation bias.

Important:
The gold standard is for VISUAL PERCEPTION, not 3RS labels.
""",
        encoding="utf-8",
    )

    print("Prepared V6.2 annotation package")
    print(f"Gold standard: {gold_path}")
    print(f"Codebook:      {codebook_path}")
    print(f"Checklist:     {checklist_path}")
    print(f"Instructions:  {readme}")


# ---------------------------------------------------------------------
# EXTRACT model events from V5 / V6 CSVs
# ---------------------------------------------------------------------

def extract_window_detections(row: pd.Series, model_name: str) -> List[Detection]:
    segment_idx = int(row["segment_idx"])
    windows = load_json_maybe(row.get("window_observations"))
    detections: List[Detection] = []

    if not isinstance(windows, list):
        return detections

    for wi, window in enumerate(windows):
        if not isinstance(window, dict):
            continue

        ws, we = parse_window_bounds(window, wi)

        # InternVL V6/V6.1 structure.
        detected_behaviors = window.get("detected_behaviors")
        if isinstance(detected_behaviors, list):
            for item in detected_behaviors:
                if not isinstance(item, dict):
                    continue
                source_behavior = str(item.get("type", "")).strip()
                category = MODEL_BEHAVIOR_TO_CATEGORY.get(source_behavior)
                if not category:
                    continue
                note = str(item.get("notes", "") or "")
                times = item.get("times_sec", []) or []
                for raw_t in times:
                    t = normalize_model_time(raw_t, ws, we)
                    if t is not None:
                        detections.append(
                            Detection(
                                model=model_name,
                                segment_idx=segment_idx,
                                category=category,
                                start_sec=t,
                                end_sec=t,
                                source_behavior=source_behavior,
                                source_note=note,
                            )
                        )

        # Qwen V5 structure.
        behavior_flags = window.get("behavior_flags")
        if isinstance(behavior_flags, dict):
            for source_behavior, payload in behavior_flags.items():
                category = MODEL_BEHAVIOR_TO_CATEGORY.get(source_behavior)
                if not category or not isinstance(payload, dict):
                    continue
                if not int(bool(payload.get("present", 0))):
                    continue
                note = str(payload.get("notes", "") or "")
                times = payload.get("times_sec", []) or []
                for raw_t in times:
                    t = normalize_model_time(raw_t, ws, we)
                    if t is not None:
                        detections.append(
                            Detection(
                                model=model_name,
                                segment_idx=segment_idx,
                                category=category,
                                start_sec=t,
                                end_sec=t,
                                source_behavior=source_behavior,
                                source_note=note,
                            )
                        )

    return detections


def cluster_detections(
    detections: List[Detection],
    max_gap_sec: float,
) -> List[Detection]:
    """
    Convert per-frame/per-sample detections into contiguous intervals.

    This is important because V6.1 sometimes emitted:
        15.5, 16.5, 17.5, ... 29.5
    for one persistent state. Those should be one interval, not 15 events.
    """
    if not detections:
        return []

    groups: Dict[Tuple[str, int, str, str], List[Detection]] = {}
    for d in detections:
        key = (d.model, d.segment_idx, d.category, d.source_behavior)
        groups.setdefault(key, []).append(d)

    clustered: List[Detection] = []

    for (model, segment_idx, category, source_behavior), items in groups.items():
        items = sorted(items, key=lambda x: x.start_sec)

        cur_start = items[0].start_sec
        cur_end = items[0].end_sec
        notes = [items[0].source_note] if items[0].source_note else []

        for d in items[1:]:
            if d.start_sec - cur_end <= max_gap_sec:
                cur_end = max(cur_end, d.end_sec)
                if d.source_note:
                    notes.append(d.source_note)
            else:
                clustered.append(
                    Detection(
                        model=model,
                        segment_idx=segment_idx,
                        category=category,
                        start_sec=round(cur_start, 3),
                        end_sec=round(cur_end, 3),
                        source_behavior=source_behavior,
                        source_note=" | ".join(dict.fromkeys(notes)),
                    )
                )
                cur_start = d.start_sec
                cur_end = d.end_sec
                notes = [d.source_note] if d.source_note else []

        clustered.append(
            Detection(
                model=model,
                segment_idx=segment_idx,
                category=category,
                start_sec=round(cur_start, 3),
                end_sec=round(cur_end, 3),
                source_behavior=source_behavior,
                source_note=" | ".join(dict.fromkeys(notes)),
            )
        )

    return sorted(
        clustered,
        key=lambda x: (x.model, x.segment_idx, x.category, x.start_sec),
    )


def extract_model_csv(
    path: Path,
    model_name: str,
    cluster_gap_sec: float,
) -> List[Detection]:
    df = pd.read_csv(path)
    if "segment_idx" not in df.columns:
        raise ValueError(f"{path} missing segment_idx")

    if "status" in df.columns:
        df = df[df["status"].fillna("") == "ok"]

    raw: List[Detection] = []
    for _, row in df.iterrows():
        raw.extend(extract_window_detections(row, model_name))

    return cluster_detections(raw, cluster_gap_sec)


# ---------------------------------------------------------------------
# Gold loading and matching
# ---------------------------------------------------------------------

def load_gold(path: Path) -> List[GoldEvent]:
    df = pd.read_csv(path)

    missing = set(GOLD_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Gold CSV missing columns: {sorted(missing)}"
        )

    # Ignore blank template rows.
    df = df[df["behavior_code"].notna()]
    df = df[df["behavior_code"].astype(str).str.strip() != ""].copy()

    if df.empty:
        raise ValueError(
            "Gold CSV has no completed annotations yet. "
            "Fill behavior_code/start_sec/end_sec first."
        )

    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce")
    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")

    bad = df[df["start_sec"].isna() | df["end_sec"].isna()]
    if not bad.empty:
        raise ValueError(
            "Every completed gold annotation needs numeric start_sec and end_sec. "
            f"Problem cue_ids: {bad['cue_id'].tolist()}"
        )

    events: List[GoldEvent] = []

    for row in df.itertuples(index=False):
        code = str(row.behavior_code).strip()
        if code not in GOLD_CODE_TO_CATEGORY:
            raise ValueError(
                f"Unknown behavior_code {code!r} for cue_id={row.cue_id}. "
                "Use the supplied codebook."
            )

        start = float(row.start_sec)
        end = float(row.end_sec)
        if end < start:
            start, end = end, start

        events.append(
            GoldEvent(
                segment_idx=int(row.segment_idx),
                cue_id=str(row.cue_id),
                category=GOLD_CODE_TO_CATEGORY[code],
                behavior_code=code,
                start_sec=start,
                end_sec=end,
                literal_description=str(row.literal_description or ""),
            )
        )

    return events


def intervals_match(
    gold_start: float,
    gold_end: float,
    det_start: float,
    det_end: float,
    tolerance_sec: float,
) -> bool:
    # Expand gold interval by tolerance.
    return not (
        det_end < gold_start - tolerance_sec
        or det_start > gold_end + tolerance_sec
    )


def match_events(
    gold_events: List[GoldEvent],
    detections: List[Detection],
    tolerance_sec: float,
):
    """
    Greedy one-to-one matching within segment + evaluation category.
    """
    details = []

    gold_groups: Dict[Tuple[int, str], List[GoldEvent]] = {}
    det_groups: Dict[Tuple[int, str], List[Detection]] = {}

    for g in gold_events:
        gold_groups.setdefault((g.segment_idx, g.category), []).append(g)

    for d in detections:
        det_groups.setdefault((d.segment_idx, d.category), []).append(d)

    all_keys = sorted(set(gold_groups) | set(det_groups))

    for key in all_keys:
        gs = sorted(gold_groups.get(key, []), key=lambda x: x.start_sec)
        ds = sorted(det_groups.get(key, []), key=lambda x: x.start_sec)

        unmatched_det = set(range(len(ds)))

        for g in gs:
            candidates = []
            gold_mid = (g.start_sec + g.end_sec) / 2.0

            for di in unmatched_det:
                d = ds[di]
                if intervals_match(
                    g.start_sec,
                    g.end_sec,
                    d.start_sec,
                    d.end_sec,
                    tolerance_sec,
                ):
                    det_mid = (d.start_sec + d.end_sec) / 2.0
                    candidates.append(
                        (abs(det_mid - gold_mid), di)
                    )

            if candidates:
                _, best_di = min(candidates)
                d = ds[best_di]
                unmatched_det.remove(best_di)

                details.append({
                    "segment_idx": g.segment_idx,
                    "category": g.category,
                    "status": "TP",
                    "gold_cue_id": g.cue_id,
                    "gold_behavior_code": g.behavior_code,
                    "gold_start_sec": g.start_sec,
                    "gold_end_sec": g.end_sec,
                    "gold_description": g.literal_description,
                    "model": d.model,
                    "det_start_sec": d.start_sec,
                    "det_end_sec": d.end_sec,
                    "source_behavior": d.source_behavior,
                    "source_note": d.source_note,
                })
            else:
                details.append({
                    "segment_idx": g.segment_idx,
                    "category": g.category,
                    "status": "FN",
                    "gold_cue_id": g.cue_id,
                    "gold_behavior_code": g.behavior_code,
                    "gold_start_sec": g.start_sec,
                    "gold_end_sec": g.end_sec,
                    "gold_description": g.literal_description,
                    "model": detections[0].model if detections else "",
                    "det_start_sec": np.nan,
                    "det_end_sec": np.nan,
                    "source_behavior": "",
                    "source_note": "",
                })

        for di in sorted(unmatched_det):
            d = ds[di]
            details.append({
                "segment_idx": d.segment_idx,
                "category": d.category,
                "status": "FP",
                "gold_cue_id": "",
                "gold_behavior_code": "",
                "gold_start_sec": np.nan,
                "gold_end_sec": np.nan,
                "gold_description": "",
                "model": d.model,
                "det_start_sec": d.start_sec,
                "det_end_sec": d.end_sec,
                "source_behavior": d.source_behavior,
                "source_note": d.source_note,
            })

    return pd.DataFrame(details)


def metrics_from_details(details: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame()

    rows = []
    grouped = details.groupby(group_cols, dropna=False) if group_cols else [((), details)]

    for key, grp in grouped:
        tp = int((grp["status"] == "TP").sum())
        fp = int((grp["status"] == "FP").sum())
        fn = int((grp["status"] == "FN").sum())

        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if pd.notna(precision)
            and pd.notna(recall)
            and (precision + recall) > 0
            else np.nan
        )

        if not isinstance(key, tuple):
            key = (key,)

        row = {col: value for col, value in zip(group_cols, key)}
        row.update({
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": round(precision, 4) if pd.notna(precision) else np.nan,
            "recall": round(recall, 4) if pd.notna(recall) else np.nan,
            "f1": round(f1, 4) if pd.notna(f1) else np.nan,
        })
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------------------

def evaluate(args):
    gold_path = Path(args.gold_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_events = load_gold(gold_path)

    model_specs = []
    if args.qwen_csv:
        model_specs.append(("Qwen3-VL-8B_V5", Path(args.qwen_csv)))
    if args.internvl_csv:
        model_specs.append(("InternVL3.5-38B_V6.1", Path(args.internvl_csv)))

    if not model_specs:
        raise ValueError("Provide at least --qwen-csv or --internvl-csv.")

    all_detections: List[Detection] = []
    all_match_details = []

    for model_name, path in model_specs:
        if not path.exists():
            print(f"WARNING: missing model CSV, skipping: {path}")
            continue

        detections = extract_model_csv(
            path,
            model_name=model_name,
            cluster_gap_sec=args.cluster_gap_sec,
        )

        # Restrict to segments that exist in gold.
        gold_segment_ids = {g.segment_idx for g in gold_events}
        detections = [
            d for d in detections
            if d.segment_idx in gold_segment_ids
        ]

        all_detections.extend(detections)

        details = match_events(
            gold_events=gold_events,
            detections=detections,
            tolerance_sec=args.tolerance_sec,
        )

        if not details.empty:
            details["model"] = model_name
            all_match_details.append(details)

    if not all_match_details:
        raise RuntimeError("No model detections/results were available to evaluate.")

    detection_df = pd.DataFrame([
        {
            "model": d.model,
            "segment_idx": d.segment_idx,
            "category": d.category,
            "start_sec": d.start_sec,
            "end_sec": d.end_sec,
            "source_behavior": d.source_behavior,
            "source_note": d.source_note,
        }
        for d in all_detections
    ])

    details_df = pd.concat(all_match_details, ignore_index=True)

    overall = metrics_from_details(details_df, ["model"])
    by_category = metrics_from_details(details_df, ["model", "category"])
    by_segment = metrics_from_details(details_df, ["model", "segment_idx"])
    by_segment_category = metrics_from_details(
        details_df,
        ["model", "segment_idx", "category"],
    )

    detection_df.to_csv(
        out_dir / "model_detection_events.csv",
        index=False,
        encoding="utf-8-sig",
    )
    details_df.to_csv(
        out_dir / "cue_match_details.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall.to_csv(
        out_dir / "cue_metrics_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_category.to_csv(
        out_dir / "cue_metrics_by_category.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_segment.to_csv(
        out_dir / "cue_metrics_by_segment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_segment_category.to_csv(
        out_dir / "cue_metrics_by_segment_category.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nV6.2 cue-level perception benchmark")
    print(f"Gold events: {len(gold_events)}")
    print(f"Tolerance: ±{args.tolerance_sec:.1f}s")
    print(f"Cluster gap: {args.cluster_gap_sec:.1f}s")
    print()
    print(overall.to_string(index=False))
    print()
    print(f"Outputs: {out_dir}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="V6.2 visual cue gold-standard preparation and benchmark."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Create manual gold-standard template/codebook.")
    p.add_argument(
        "--segments-csv",
        required=True,
    )
    p.add_argument(
        "--output-dir",
        default="./output/visual_cue_benchmark_v6_2",
    )
    p.add_argument(
        "--segment-indices",
        default="4,10,16,63,65",
    )
    p.add_argument(
        "--annotator",
        default="",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing gold-standard template.",
    )

    e = sub.add_parser("evaluate", help="Compare Qwen/InternVL cue detections with manual gold.")
    e.add_argument(
        "--gold-csv",
        default="./output/visual_cue_benchmark_v6_2/visual_cue_gold_standard.csv",
    )
    e.add_argument(
        "--qwen-csv",
        default="./output/qwen3vl_visual_experiment_v5/visual_experiment_v5_predictions.csv",
    )
    e.add_argument(
        "--internvl-csv",
        default="./output/internvl35_38b_visual_experiment_v6_1_1fps/visual_experiment_v6_internvl38_predictions.csv",
    )
    e.add_argument(
        "--output-dir",
        default="./output/visual_cue_benchmark_v6_2/evaluation",
    )
    e.add_argument(
        "--tolerance-sec",
        type=float,
        default=2.0,
        help="Temporal tolerance for matching machine events to manual gold.",
    )
    e.add_argument(
        "--cluster-gap-sec",
        type=float,
        default=1.6,
        help="Merge repeated per-frame machine detections into one interval.",
    )

    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
