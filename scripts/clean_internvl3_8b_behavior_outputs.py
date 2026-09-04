#!/usr/bin/env python
"""
Clean InternVL3-8B behavior detections BEFORE benchmarking.

Reads the SAVED raw window outputs from:
    output/internvl3_8b_behavior_benchmark_v1_3/
        internvl3_8b_behavior_details.jsonl

No VLM inference is rerun.

Cleaning philosophy
-------------------
Only fix representation / parsing problems.
Do NOT "correct" model perception.

We:
1. Parse the original pipe-delimited raw output again.
2. Normalize known behavior-code aliases.
3. Validate timestamps STRICTLY per window.
4. Convert valid local-window times -> global segment times.
5. Reject impossible timestamps instead of clipping them.
6. Merge repeated adjacent detections of the same behavior.
7. Write an audit file showing every rejected / corrected record.

Examples for window 15-30 s:
    18.0 -> valid global time, keep 18.0
    11.0 -> valid local time, convert to 26.0
    116.0 -> impossible, reject
    330.0 -> impossible, reject

This separates SOFTWARE/PARSING errors from actual MODEL errors.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ALIAS_MAP = {
    # Known spelling / singular-plural variants.
    "hand_clasped": "hands_clasped",
    "hands_clasp": "hands_clasped",
    "hand_on_lap": "hands_on_lap",
    "hand_on_table": "hands_on_table",
    "leg_crossed": "legs_cross",
    "ankle_crossed": "ankles_cross",
    "shoulder_elevate": "shoulders_elevate",
    "shoulder_lower": "shoulders_lower",
}

ALLOWED_CODES = {
    "gaze_toward_therapist",
    "gaze_away_from_therapist",
    "gaze_down",
    "gaze_up",
    "gaze_image_left",
    "gaze_image_right",
    "eyes_closed",
    "gaze_uncertain",

    "head_pitch_down",
    "head_pitch_up",
    "head_yaw_left",
    "head_yaw_right",
    "head_turn",
    "negative_head_shake",
    "abrupt_head_turn",
    "lips_pressed",
    "mouth_tension",
    "smile",
    "crying_visible",
    "tear_visible",

    "hand_to_eye_region",
    "hand_to_cheek",
    "hand_to_chin",
    "hand_to_mouth",
    "hand_to_forehead",
    "hand_to_ear",
    "hand_to_face_unspecified",
    "hand_away_from_face",

    "hands_clasped",
    "fingers_interlaced",
    "hands_on_lap",
    "hands_on_table",
    "open_palm",
    "closed_fist",
    "pointing_gesture",
    "palm_up_gesture",
    "palm_down_gesture",
    "hands_apart",
    "hands_together_change",
    "hand_wave",
    "hand_rubbing",
    "finger_fidgeting",
    "hand_grips_object",
    "hand_releases_object",
    "self_touch_arm",
    "self_touch_neck",
    "clothing_adjustment",
    "generic_hand_gesture",

    "arms_cross",
    "arms_uncross",
    "arms_open",
    "arms_close_to_body",
    "push_away_gesture",
    "arm_raise",
    "arm_lower",

    "shoulders_elevate",
    "shoulders_lower",
    "shoulder_lift_drop",
    "torso_backward",
    "torso_forward",
    "slump",
    "body_turn_left",
    "body_turn_right",
    "body_turn",
    "movement_reduction",
    "movement_increase",

    "legs_cross",
    "legs_uncross",
    "ankles_cross",
    "legs_open",
    "legs_close",
    "leg_shift",
    "leg_extend",
    "leg_pull_back",
    "leg_bounce",
    "foot_tap",
    "foot_shift",
    "feet_still",

    "other_literal_visual_event",
}

CERTAINTY_VALUES = {"clear", "possible", "ambiguous"}


def strip_time_suffix(value):
    """
    Accept simple model outputs like 56.8s or 15 sec.
    """
    s = str(value).strip().lower()
    s = re.sub(r"\s*(seconds?|secs?|sec|s)$", "", s)
    return s.strip()


def parse_float(value):
    try:
        return float(strip_time_suffix(value))
    except Exception:
        return None


def normalize_time_pair(
    raw_start,
    raw_end,
    window_start,
    window_end,
):
    """
    Strict timestamp normalization.

    Returns:
        (start, end, time_mode, error_reason)

    time_mode:
        global
        local_to_global
        mixed_recovered
    """
    s = parse_float(raw_start)
    e = parse_float(raw_end)

    if s is None or e is None:
        return None, None, None, "non_numeric_time"

    duration = float(window_end - window_start)
    eps = 0.26

    def is_global(t):
        return (window_start - eps) <= t <= (window_end + eps)

    def is_local(t):
        return (-eps) <= t <= (duration + eps)

    s_global = is_global(s)
    e_global = is_global(e)
    s_local = is_local(s)
    e_local = is_local(e)

    # Prefer both-global when possible.
    if s_global and e_global:
        ns, ne = s, e
        mode = "global"

    # Both local -> shift.
    elif s_local and e_local:
        ns, ne = window_start + s, window_start + e
        mode = "local_to_global"

    # Mixed case can happen if one boundary is exactly 15 in a 15-30 window.
    # Recover only when the result is unambiguous and both normalized values
    # fall inside the window.
    else:
        candidates = []

        s_options = []
        e_options = []

        if s_global:
            s_options.append(("global", s))
        if s_local:
            s_options.append(("local", window_start + s))

        if e_global:
            e_options.append(("global", e))
        if e_local:
            e_options.append(("local", window_start + e))

        for smode, sv in s_options:
            for emode, ev in e_options:
                if (
                    window_start - eps <= sv <= window_end + eps
                    and window_start - eps <= ev <= window_end + eps
                ):
                    candidates.append((sv, ev, smode, emode))

        # Prefer monotonic interval.
        candidates = [
            x for x in candidates
            if x[1] + eps >= x[0]
        ]

        if len(candidates) == 1:
            ns, ne, smode, emode = candidates[0]
            mode = f"mixed_recovered:{smode}+{emode}"
        else:
            return None, None, None, "impossible_or_ambiguous_time"

    # Clamp only tiny floating-point overshoot, never wild timestamps.
    ns = max(window_start, min(ns, window_end))
    ne = max(window_start, min(ne, window_end))

    if ne < ns:
        # Small reversed event boundaries are not silently repaired.
        return None, None, None, "end_before_start"

    return round(ns, 3), round(ne, 3), mode, None


def normalize_code(raw_code):
    code = str(raw_code).strip()
    code = ALIAS_MAP.get(code, code)
    return code


def parse_raw_window(
    raw,
    segment_idx,
    window_start,
    window_end,
):
    accepted = []
    audit = []

    for line_no, raw_line in enumerate(str(raw or "").splitlines(), start=1):
        line = raw_line.strip()

        if not line:
            continue

        line = re.sub(r"^[\-\*\u2022]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)

        parts = [p.strip() for p in line.split("|")]
        if not parts:
            continue

        tag = parts[0].upper()

        if tag in {"VISIBILITY", "UNRESOLVED"}:
            continue

        if tag not in {"STATE", "EVENT"}:
            audit.append(
                {
                    "segment_idx": segment_idx,
                    "window_start": window_start,
                    "window_end": window_end,
                    "line_no": line_no,
                    "raw_line": raw_line,
                    "status": "REJECTED",
                    "reason": "unknown_record_type",
                }
            )
            continue

        if len(parts) < 6:
            audit.append(
                {
                    "segment_idx": segment_idx,
                    "window_start": window_start,
                    "window_end": window_end,
                    "line_no": line_no,
                    "raw_line": raw_line,
                    "status": "REJECTED",
                    "reason": "too_few_fields",
                }
            )
            continue

        raw_code = parts[1]
        code = normalize_code(raw_code)

        if code not in ALLOWED_CODES:
            audit.append(
                {
                    "segment_idx": segment_idx,
                    "window_start": window_start,
                    "window_end": window_end,
                    "line_no": line_no,
                    "raw_line": raw_line,
                    "status": "REJECTED",
                    "reason": f"unknown_behavior_code:{raw_code}",
                }
            )
            continue

        start_sec, end_sec, time_mode, time_error = normalize_time_pair(
            parts[2],
            parts[3],
            float(window_start),
            float(window_end),
        )

        if time_error:
            audit.append(
                {
                    "segment_idx": segment_idx,
                    "window_start": window_start,
                    "window_end": window_end,
                    "line_no": line_no,
                    "raw_line": raw_line,
                    "status": "REJECTED",
                    "reason": time_error,
                }
            )
            continue

        certainty = parts[4].lower()
        if certainty not in CERTAINTY_VALUES:
            certainty = "possible"

        description = "|".join(parts[5:]).strip()

        accepted.append(
            {
                "segment_idx": int(segment_idx),
                "kind": tag.lower(),
                "behavior_code": code,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "certainty": certainty,
                "description": description,
                "source_window_start": float(window_start),
                "source_window_end": float(window_end),
            }
        )

        change_notes = []
        if raw_code != code:
            change_notes.append(f"alias:{raw_code}->{code}")
        if time_mode != "global":
            change_notes.append(time_mode)

        audit.append(
            {
                "segment_idx": segment_idx,
                "window_start": window_start,
                "window_end": window_end,
                "line_no": line_no,
                "raw_line": raw_line,
                "status": "ACCEPTED",
                "reason": ";".join(change_notes) if change_notes else "as_is",
            }
        )

    return accepted, audit


def merge_same_behavior(rows, merge_gap_sec=1.1):
    """
    Merge repeated adjacent detections only when:
    - same segment
    - same behavior code
    - same state/event kind
    - gap <= merge_gap_sec

    We do NOT merge across long gaps.
    """
    if not rows:
        return []

    df = pd.DataFrame(rows)

    merged = []

    for (segment_idx, kind, behavior_code), g in df.groupby(
        ["segment_idx", "kind", "behavior_code"],
        sort=False,
    ):
        g = g.sort_values(["start_sec", "end_sec"])

        current = None

        for row in g.to_dict("records"):
            if current is None:
                current = dict(row)
                continue

            gap = row["start_sec"] - current["end_sec"]

            if gap <= merge_gap_sec:
                current["end_sec"] = max(
                    current["end_sec"],
                    row["end_sec"],
                )

                if (
                    row["description"]
                    and row["description"] not in current["description"]
                ):
                    current["description"] = (
                        current["description"]
                        + " | "
                        + row["description"]
                    ).strip(" |")
            else:
                merged.append(current)
                current = dict(row)

        if current is not None:
            merged.append(current)

    return sorted(
        merged,
        key=lambda x: (
            x["segment_idx"],
            x["start_sec"],
            x["behavior_code"],
        ),
    )


def main(args):
    details_path = Path(args.details_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    audit_rows = []
    segment_meta = {}

    with details_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            segment_idx = int(obj["segment_idx"])

            segment_meta[segment_idx] = {
                "video": Path(obj.get("segment_path", "")).name.replace(
                    f"_eval_{segment_idx:03d}", ""
                ),
                "patient_id": "",
                "session_id": "",
            }

            # Better metadata if merged detections already contain it.
            if obj.get("merged_detections"):
                first = obj["merged_detections"][0]
                segment_meta[segment_idx] = {
                    "video": first.get("video", ""),
                    "patient_id": first.get("patient_id", ""),
                    "session_id": first.get("session_id", ""),
                }

            for window in obj.get("windows", []):
                window_start = float(window["window_start"])
                window_end = float(window["window_end"])
                raw = window.get("raw", "")

                accepted, audit = parse_raw_window(
                    raw,
                    segment_idx,
                    window_start,
                    window_end,
                )

                all_rows.extend(accepted)
                audit_rows.extend(audit)

    merged = merge_same_behavior(
        all_rows,
        merge_gap_sec=args.merge_gap_sec,
    )

    final_rows = []

    for row in merged:
        meta = segment_meta.get(row["segment_idx"], {})
        final_rows.append(
            {
                "segment_idx": row["segment_idx"],
                "video": meta.get("video", ""),
                "patient_id": meta.get("patient_id", ""),
                "session_id": meta.get("session_id", ""),
                "kind": row["kind"],
                "behavior_code": row["behavior_code"],
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "certainty": row["certainty"],
                "description": row["description"],
            }
        )

    cleaned_path = out_dir / "internvl3_8b_behavior_detections_CLEAN.csv"
    audit_path = out_dir / "internvl3_8b_behavior_cleaning_audit.csv"
    rejected_path = out_dir / "internvl3_8b_behavior_rejected_rows.csv"

    pd.DataFrame(final_rows).to_csv(
        cleaned_path,
        index=False,
        encoding="utf-8-sig",
    )

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(
        audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    rejected = audit_df[
        audit_df["status"] == "REJECTED"
    ].copy()

    rejected.to_csv(
        rejected_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("")
    print("INTERNVL3-8B BEHAVIOR CLEANING")
    print("=" * 60)
    print(f"Raw behavior records parsed: {len(all_rows)}")
    print(f"Clean merged detections: {len(final_rows)}")
    print(f"Rejected malformed records: {len(rejected)}")
    print("")
    print("Rejection reasons:")
    if len(rejected):
        print(
            rejected["reason"]
            .value_counts()
            .to_string()
        )
    else:
        print("none")
    print("")
    print("Outputs:")
    print(cleaned_path)
    print(audit_path)
    print(rejected_path)


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--details-jsonl",
        default=(
            "./output/internvl3_8b_behavior_benchmark_v1_3/"
            "internvl3_8b_behavior_details.jsonl"
        ),
    )

    p.add_argument(
        "--output-dir",
        default=(
            "./output/internvl3_8b_behavior_benchmark_v1_3/cleaned"
        ),
    )

    p.add_argument(
        "--merge-gap-sec",
        type=float,
        default=1.1,
    )

    return p


if __name__ == "__main__":
    main(build_parser().parse_args())