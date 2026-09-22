#!/usr/bin/env python
"""
Visual Experiment V6.3
======================

Qwen3-VL-8B expanded literal visual perception benchmark.

PURPOSE
-------
This experiment evaluates VISUAL PERCEPTION ONLY.

It does NOT:
- use human WD_P / CF_P labels as model input
- classify rupture / withdrawal / confrontation
- run a 3RS judge
- infer speech, emotion, intention, motivation, or therapeutic meaning

It DOES:
- reuse the proven V5 YuNet patient localization / crop pipeline
- reuse the V5 patient-side role cache
- sample the patient crop at 2 FPS by default
- analyze four independent 15-second windows
- use an expanded literal body-behavior vocabulary
- explicitly separate SUSTAINED STATES from DISCRETE EVENTS
- normalize all times to GLOBAL 0-60 second segment time
- merge adjacent repeated states deterministically
- deduplicate repeated discrete events deterministically

REQUIREMENT
-----------
Keep this file in the SAME scripts folder as:
    run_qwen3vl_visual_experiment_v5.py

The V5 script is imported only for stable infrastructure:
YuNet, face tracking, patient ROI, Qwen loader/generation, JSON repair,
previews, and CSV resume helpers.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import pandas as pd
import torch

import run_qwen3vl_visual_experiment_v5 as v5


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# ---------------------------------------------------------------------
# Expanded literal vocabulary.
#
# These are OBSERVABLE behavior codes, not clinical interpretations.
# The same code may occur as:
#   STATE: a condition that persists over an interval
#   EVENT: a change/motion that happens over a short interval
# ---------------------------------------------------------------------

BEHAVIOR_GROUPS = {
    "GAZE_EYES": [
        "gaze_toward_therapist",
        "gaze_away_from_therapist",
        "gaze_down",
        "gaze_up",
        "gaze_image_left",
        "gaze_image_right",
        "eyes_closed",
        "gaze_uncertain",
    ],
    "HEAD_FACE": [
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
    ],
    "HAND_FACE": [
        "hand_to_eye_region",
        "hand_to_cheek",
        "hand_to_chin",
        "hand_to_mouth",
        "hand_to_forehead",
        "hand_to_ear",
        "hand_to_face_unspecified",
        "hand_away_from_face",
    ],
    "HANDS_FINGERS": [
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
    ],
    "ARMS": [
        "arms_cross",
        "arms_uncross",
        "arms_open",
        "arms_close_to_body",
        "push_away_gesture",
        "arm_raise",
        "arm_lower",
    ],
    "SHOULDERS_BODY": [
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
    ],
    "LEGS_FEET": [
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
    ],
    "OTHER": [
        "other_literal_visual_event",
    ],
}

ALLOWED_CODES = {
    code
    for codes in BEHAVIOR_GROUPS.values()
    for code in codes
}

CERTAINTY_LEVELS = {"clear", "possible", "ambiguous"}
VISIBILITY_LEVELS = {"good", "partial", "poor"}

# Stable ordering for reporting.
CODE_ORDER = [
    code
    for codes in BEHAVIOR_GROUPS.values()
    for code in codes
]
CODE_RANK = {code: i for i, code in enumerate(CODE_ORDER)}


# =====================================================================
# Prompt
# =====================================================================

def vocabulary_text():
    blocks = []
    for group, codes in BEHAVIOR_GROUPS.items():
        blocks.append(
            group + ":\n  " + "\n  ".join(f"- {x}" for x in codes)
        )
    return "\n\n".join(blocks)


def observer_messages(
    window_frames,
    window_start,
    window_end,
    sample_fps,
    total_pixels,
    therapist_side,
):
    if therapist_side in {"left", "right"}:
        therapist_note = f"""
THERAPIST LOCATION:
The therapist is on IMAGE-{therapist_side.upper()} in the original scene.
For gaze_toward_therapist / gaze_away_from_therapist, use the EYES when
visible, not only head direction. If eye direction is not sufficiently clear,
use gaze_image_left / gaze_image_right / gaze_down / gaze_up or gaze_uncertain.
""".strip()
    else:
        therapist_note = """
THERAPIST LOCATION:
The therapist side cannot be established reliably from the current full scene.
DO NOT use gaze_toward_therapist or gaze_away_from_therapist.
Use gaze_image_left / gaze_image_right / gaze_down / gaze_up or
gaze_uncertain instead.
""".strip()

    prompt = f"""
You are a HIGH-RECALL LITERAL VISUAL BEHAVIOR OBSERVER.

You see ONLY the PATIENT from approximately
{window_start:.1f}-{window_end:.1f} seconds of a psychotherapy video.

There is NO audio and NO transcript.
The timestamp printed on each frame is GLOBAL time within the full
60-second segment.

{therapist_note}

YOUR ONLY JOB:
Record what is visibly observable.

DO NOT decide:
- rupture
- withdrawal
- confrontation
- resistance
- engagement
- emotion unless it is directly visible as a listed facial behavior
- motivation
- intention
- meaning
- speech content
- tone of voice

IMPORTANT STATE-vs-EVENT RULE
-----------------------------
A STATE is something that remains visible for a period:
- gaze stays down
- hand remains on cheek
- hands remain clasped
- ankles remain crossed
- head remains angled down
- patient remains unusually still

An EVENT is a transition or discrete motion:
- head turns
- arm raises
- hand moves to face
- shoulders lift then drop
- pointing movement
- leg changes position

DO NOT create one event for every sampled frame.

BAD:
  head_pitch_down at 15.5, 16.0, 16.5, 17.0, ...

GOOD:
  one STATE:
  head_pitch_down from 15.5 to 21.0

For repeated actions such as hand rubbing, foot tapping, or leg bouncing,
use ONE interval covering the repeated episode.

VISUAL BOUNDARIES
-----------------
- hand_to_eye_region means literal hand contact/passage over eye or
  upper-cheek area. It does NOT automatically mean tear wiping.
- crying_visible requires visually clear crying behavior.
  A hand near the eye alone is NOT crying.
- tear_visible requires a visible tear or tear track.
- mouth_tension / lips_pressed must be visibly observable.
- shoulder_lift_drop requires a visible lift followed by a drop.
- gaze is separate from head pose.
- if gaze is uncertain, say gaze_uncertain.
- use image-left/image-right literally.
- do not infer a behavior that is occluded or too small to see.

ALLOWED BEHAVIOR CODES
----------------------
{vocabulary_text()}

RETURN COMPACT JSON ONLY
------------------------
{{
  "window": "{window_start:.1f}-{window_end:.1f}",
  "visibility": "good|partial|poor",

  "states": [
    {{
      "type": "allowed_behavior_code",
      "start_sec": {window_start:.1f},
      "end_sec": {window_start + 2.0:.1f},
      "certainty": "clear|possible|ambiguous",
      "description": "short literal visual description"
    }}
  ],

  "events": [
    {{
      "type": "allowed_behavior_code",
      "start_sec": {window_start + 3.0:.1f},
      "end_sec": {window_start + 3.5:.1f},
      "certainty": "clear|possible|ambiguous",
      "description": "short literal visible change"
    }}
  ],

  "unresolved": [
    "short note about something visible but too uncertain to code"
  ]
}}

TIME RULES
----------
- Use GLOBAL timestamps printed on the frames.
- State/event boundaries should be approximate to the nearest visible
  sampled frame.
- start_sec <= end_sec.
- Keep separate intervals separate when the behavior clearly stops.
- Empty lists are allowed.
- Prefer a small number of accurate intervals over repetitive per-frame rows.
""".strip()

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": window_frames,
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


# =====================================================================
# Normalization / validation
# =====================================================================

def safe_float(value):
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return None


def normalize_global_time(value, window_start, window_end):
    """
    The prompt requests GLOBAL time, but models sometimes return local
    window time. This converts either form deterministically.
    """
    t = safe_float(value)
    if t is None:
        return None

    duration = float(window_end - window_start)

    # Clearly inside requested global window.
    if window_start - 0.75 <= t <= window_end + 0.75:
        return round(max(window_start, min(t, window_end)), 3)

    # Plausible local time within current window.
    if -0.25 <= t <= duration + 0.75:
        t = window_start + t
        return round(max(window_start, min(t, window_end)), 3)

    # Out-of-range time: reject rather than silently invent.
    return None


def normalize_certainty(value):
    x = str(value or "").strip().lower()
    return x if x in CERTAINTY_LEVELS else "possible"


def normalize_visibility(value):
    x = str(value or "").strip().lower()
    return x if x in VISIBILITY_LEVELS else "partial"


def normalize_item(item, kind, window_start, window_end):
    if not isinstance(item, dict):
        return None

    code = str(
        item.get("type", item.get("behavior_code", ""))
        or ""
    ).strip()

    if code not in ALLOWED_CODES:
        return None

    start = normalize_global_time(
        item.get("start_sec", item.get("time_sec")),
        window_start,
        window_end,
    )
    end = normalize_global_time(
        item.get("end_sec", item.get("time_sec")),
        window_start,
        window_end,
    )

    if start is None and end is None:
        return None
    if start is None:
        start = end
    if end is None:
        end = start
    if end < start:
        start, end = end, start

    return {
        "kind": kind,
        "type": code,
        "start_sec": round(float(start), 3),
        "end_sec": round(float(end), 3),
        "certainty": normalize_certainty(item.get("certainty")),
        "description": str(item.get("description", "") or "").strip(),
    }


def normalize_observation(parsed, window_start, window_end):
    if not isinstance(parsed, dict):
        parsed = {}

    states = []
    events = []

    for item in parsed.get("states", []) or []:
        x = normalize_item(
            item,
            "state",
            window_start,
            window_end,
        )
        if x:
            states.append(x)

    for item in parsed.get("events", []) or []:
        x = normalize_item(
            item,
            "event",
            window_start,
            window_end,
        )
        if x:
            events.append(x)

    unresolved = parsed.get("unresolved", []) or []
    if not isinstance(unresolved, list):
        unresolved = [str(unresolved)]

    return {
        "window": f"{window_start:.1f}-{window_end:.1f}",
        "visibility": normalize_visibility(
            parsed.get("visibility")
        ),
        "states": states,
        "events": events,
        "unresolved": [
            str(x).strip()
            for x in unresolved
            if str(x).strip()
        ],
        "window_start_global": round(window_start, 3),
        "window_end_global": round(window_end, 3),
    }


CERTAINTY_PRIORITY = {
    "ambiguous": 0,
    "possible": 1,
    "clear": 2,
}


def weaker_certainty(a, b):
    # Conservative: merged interval gets the weaker of the two.
    return min(
        [a, b],
        key=lambda x: CERTAINTY_PRIORITY.get(x, 1),
    )


def merge_state_intervals(states, max_gap_sec=1.1):
    """
    Merge adjacent/overlapping intervals of the SAME behavior code.

    This directly addresses the V6/V6.1 problem where persistent states
    were emitted as many repeated frame-level detections.
    """
    if not states:
        return []

    by_type = {}
    for x in states:
        by_type.setdefault(x["type"], []).append(dict(x))

    merged = []

    for code, items in by_type.items():
        items.sort(key=lambda x: (x["start_sec"], x["end_sec"]))
        current = dict(items[0])
        descriptions = [
            current["description"]
        ] if current.get("description") else []

        for nxt in items[1:]:
            gap = nxt["start_sec"] - current["end_sec"]

            if gap <= max_gap_sec:
                current["end_sec"] = max(
                    current["end_sec"],
                    nxt["end_sec"],
                )
                current["certainty"] = weaker_certainty(
                    current["certainty"],
                    nxt["certainty"],
                )
                if nxt.get("description"):
                    descriptions.append(nxt["description"])
            else:
                current["description"] = " | ".join(
                    dict.fromkeys(descriptions)
                )
                merged.append(current)

                current = dict(nxt)
                descriptions = [
                    current["description"]
                ] if current.get("description") else []

        current["description"] = " | ".join(
            dict.fromkeys(descriptions)
        )
        merged.append(current)

    return sorted(
        merged,
        key=lambda x: (
            x["start_sec"],
            CODE_RANK.get(x["type"], 9999),
        ),
    )


def deduplicate_events(events, dedup_sec=0.75):
    """
    Merge duplicate model outputs for the same short event when their
    intervals overlap or are nearly adjacent.
    """
    if not events:
        return []

    by_type = {}
    for x in events:
        by_type.setdefault(x["type"], []).append(dict(x))

    out = []

    for code, items in by_type.items():
        items.sort(key=lambda x: (x["start_sec"], x["end_sec"]))
        current = dict(items[0])
        descriptions = [
            current["description"]
        ] if current.get("description") else []

        for nxt in items[1:]:
            if nxt["start_sec"] - current["end_sec"] <= dedup_sec:
                current["end_sec"] = max(
                    current["end_sec"],
                    nxt["end_sec"],
                )
                current["certainty"] = weaker_certainty(
                    current["certainty"],
                    nxt["certainty"],
                )
                if nxt.get("description"):
                    descriptions.append(nxt["description"])
            else:
                current["description"] = " | ".join(
                    dict.fromkeys(descriptions)
                )
                out.append(current)

                current = dict(nxt)
                descriptions = [
                    current["description"]
                ] if current.get("description") else []

        current["description"] = " | ".join(
            dict.fromkeys(descriptions)
        )
        out.append(current)

    return sorted(
        out,
        key=lambda x: (
            x["start_sec"],
            CODE_RANK.get(x["type"], 9999),
        ),
    )


def build_type_summary(states, events):
    summary = {}

    for code in CODE_ORDER:
        s = [x for x in states if x["type"] == code]
        e = [x for x in events if x["type"] == code]

        if not s and not e:
            continue

        state_duration = sum(
            max(0.0, x["end_sec"] - x["start_sec"])
            for x in s
        )

        summary[code] = {
            "present": 1,
            "state_interval_count": len(s),
            "event_count": len(e),
            "state_duration_sec": round(state_duration, 3),
            "state_intervals": [
                [x["start_sec"], x["end_sec"]]
                for x in s
            ],
            "event_intervals": [
                [x["start_sec"], x["end_sec"]]
                for x in e
            ],
        }

    return summary


def therapist_side_from_tracks(tracks, selected_track):
    """
    Only call therapist side 'known' when a second persistent face exists.
    If only the patient is visible, do not invent therapist location.
    """
    others = [
        t for t in tracks
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


# =====================================================================
# Output helpers
# =====================================================================

def load_detection_rows(path):
    path = Path(path)
    if not path.exists():
        return []

    try:
        df = pd.read_csv(path)
        return df.to_dict("records")
    except Exception:
        return []


def save_detection_rows(rows, path):
    columns = [
        "segment_idx",
        "video",
        "patient_id",
        "session_id",
        "kind",
        "behavior_code",
        "start_sec",
        "end_sec",
        "certainty",
        "description",
    ]

    df = pd.DataFrame(rows)

    if df.empty:
        df = pd.DataFrame(columns=columns)
    else:
        for c in columns:
            if c not in df.columns:
                df[c] = ""
        df = df[columns].sort_values(
            ["segment_idx", "start_sec", "kind", "behavior_code"]
        )

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def detection_rows_for_segment(
    segment_idx,
    row,
    states,
    events,
):
    rows = []

    for x in states + events:
        rows.append(
            {
                "segment_idx": segment_idx,
                "video": getattr(row, "video", ""),
                "patient_id": getattr(row, "patient_id", ""),
                "session_id": getattr(row, "session_id", ""),
                "kind": x["kind"],
                "behavior_code": x["type"],
                "start_sec": x["start_sec"],
                "end_sec": x["end_sec"],
                "certainty": x["certainty"],
                "description": x["description"],
            }
        )

    return rows


# =====================================================================
# Main
# =====================================================================

def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    track_preview_dir = output_dir / "track_previews"
    patient_preview_dir = output_dir / "patient_crop_previews"

    csv_path = (
        output_dir
        / "visual_experiment_v6_3_predictions.csv"
    )
    jsonl_path = (
        output_dir
        / "visual_experiment_v6_3_details.jsonl"
    )
    detections_path = (
        output_dir
        / "visual_experiment_v6_3_detections.csv"
    )

    role_cache_path = Path(args.role_cache)
    role_cache = v5.load_role_cache(role_cache_path)

    segments = pd.read_csv(args.segments_csv)

    required = {"segment_idx", "segment_path"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(
            f"segments CSV missing columns: {sorted(missing)}"
        )

    segments["segment_idx"] = pd.to_numeric(
        segments["segment_idx"],
        errors="raise",
    ).astype(int)

    requested = v5.parse_segment_indices(args.segment_indices)

    if requested is not None:
        segments = segments[
            segments["segment_idx"].isin(requested)
        ].copy()

    if args.max_segments is not None:
        segments = segments.head(args.max_segments)

    rows, completed = v5.load_previous(csv_path)
    detection_rows = load_detection_rows(detections_path)

    window_total_pixels = int(
        args.window_video_token_budget * 32 * 32
    )

    face_detector = v5.get_face_detector(
        model_path=args.yunet_model,
        score_threshold=args.yunet_score_threshold,
        nms_threshold=args.yunet_nms_threshold,
        top_k=args.yunet_top_k,
    )

    print("Visual Experiment V6.3", flush=True)
    print("Purpose: expanded literal perception ONLY", flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Segments: {len(segments)}", flush=True)
    print(f"Sampling: {args.sample_fps} FPS", flush=True)
    print(
        f"Windowing: {args.window_seconds:.0f}s "
        f"(~{int(args.sample_fps * args.window_seconds)} frames/call)",
        flush=True,
    )
    print(
        f"Patient crop width: <= {args.frame_width}px",
        flush=True,
    )
    print(
        f"Role cache: {role_cache_path}",
        flush=True,
    )
    print(
        "Output schema: sustained STATES + discrete EVENTS",
        flush=True,
    )
    print(
        "NO WD_P/CF_P/rupture judge in this experiment.",
        flush=True,
    )

    qwen = v5.QwenRunner(args.model_id)

    for pos, row in enumerate(
        segments.itertuples(index=False),
        start=1,
    ):
        segment_idx = int(row.segment_idx)
        segment_path = Path(row.segment_path)

        if segment_idx in completed:
            print(
                f"[{pos}/{len(segments)}] segment "
                f"{segment_idx}: already done",
                flush=True,
            )
            continue

        print(
            f"\n[{pos}/{len(segments)}] segment {segment_idx}: "
            f"{segment_path.name}",
            flush=True,
        )

        full_started = time.time()

        try:
            # ---------------------------------------------------------
            # 1. Sample full scene.
            # ---------------------------------------------------------
            full_frames, timestamps, duration = v5.sample_full_frames(
                segment_path,
                sample_fps=args.sample_fps,
                max_duration=args.max_duration,
            )

            fw, fh = full_frames[0].size

            # ---------------------------------------------------------
            # 2. Persistent face tracks.
            # ---------------------------------------------------------
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

            print(
                f"Persistent face candidates: {len(tracks)}",
                flush=True,
            )

            if not tracks:
                raise RuntimeError(
                    "No persistent face candidates detected."
                )

            for track in tracks:
                print(
                    f"  TRACK {track['track_id']}: "
                    f"x={track['median_center_x_norm']:.3f}, "
                    f"detections={track['detection_count']}, "
                    f"face={track['median_face_size_px']:.1f}px",
                    flush=True,
                )

            # ---------------------------------------------------------
            # 3. Reuse V5 patient role cache.
            # ---------------------------------------------------------
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

            selected_track, selection_method = (
                v5.select_patient_track(
                    tracks=tracks,
                    row=row,
                    preview_path=track_preview_path,
                    cache=role_cache,
                    cache_path=role_cache_path,
                    mode=args.patient_selection,
                    forced_side=args.patient_side,
                )
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

            print(
                f"Patient: TRACK {selected_track['track_id']} "
                f"via {selection_method}; side={patient_side}",
                flush=True,
            )
            print(
                f"Therapist side for gaze coding: {therapist_side}",
                flush=True,
            )
            print(
                f"ROI x={patient_roi['x1']}:{patient_roi['x2']} "
                f"y={patient_roi['y1']}:{patient_roi['y2']}",
                flush=True,
            )

            # ---------------------------------------------------------
            # 4. Patient crop with global timestamps.
            # ---------------------------------------------------------
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

            # ---------------------------------------------------------
            # 5. Independent short-window perception calls.
            # ---------------------------------------------------------
            windows = v5.split_windows(
                patient_frames,
                timestamps,
                window_seconds=args.window_seconds,
                max_duration=min(args.max_duration, duration),
            )

            window_results = []
            window_details = []

            peak_alloc_values = []
            peak_reserved_values = []
            total_observer_sec = 0.0

            for window_no, window in enumerate(
                windows,
                start=1,
            ):
                print(
                    f"  Window {window_no}/{len(windows)} "
                    f"{window['start']:.0f}-{window['end']:.0f}s | "
                    f"{len(window['frames'])} frames",
                    flush=True,
                )

                raw, sec, peak_alloc, peak_reserved = (
                    qwen.generate(
                        observer_messages(
                            window_frames=window["frames"],
                            window_start=window["start"],
                            window_end=window["end"],
                            sample_fps=args.sample_fps,
                            total_pixels=window_total_pixels,
                            therapist_side=therapist_side,
                        ),
                        max_new_tokens=args.observer_max_new_tokens,
                    )
                )

                parsed, repair_method = v5.parse_model_json(
                    raw,
                    qwen=qwen,
                )

                normalized = normalize_observation(
                    parsed,
                    window_start=window["start"],
                    window_end=window["end"],
                )

                total_observer_sec += sec

                if peak_alloc is not None:
                    peak_alloc_values.append(peak_alloc)
                if peak_reserved is not None:
                    peak_reserved_values.append(peak_reserved)

                window_results.append(normalized)
                window_details.append(
                    {
                        "window_start": window["start"],
                        "window_end": window["end"],
                        "timestamps": window["timestamps"],
                        "normalized": normalized,
                        "raw_parsed": parsed,
                        "raw_text": raw,
                        "json_parse_method": repair_method,
                        "inference_sec": sec,
                        "peak_allocated_gb": peak_alloc,
                        "peak_reserved_gb": peak_reserved,
                    }
                )

                state_types = [
                    x["type"]
                    for x in normalized["states"]
                ]
                event_types = [
                    x["type"]
                    for x in normalized["events"]
                ]

                detected = list(
                    dict.fromkeys(state_types + event_types)
                )

                print(
                    "    detected: "
                    + (
                        ", ".join(detected)
                        if detected
                        else "none"
                    )
                    + f" | states={len(state_types)} "
                    + f"events={len(event_types)} "
                    + f"| {sec:.1f}s "
                    + f"| JSON={repair_method}",
                    flush=True,
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ---------------------------------------------------------
            # 6. Deterministic cross-window normalization.
            # ---------------------------------------------------------
            raw_states = [
                x
                for window in window_results
                for x in window["states"]
            ]
            raw_events = [
                x
                for window in window_results
                for x in window["events"]
            ]

            merged_states = merge_state_intervals(
                raw_states,
                max_gap_sec=args.state_merge_gap_sec,
            )
            merged_events = deduplicate_events(
                raw_events,
                dedup_sec=args.event_dedup_sec,
            )

            type_summary = build_type_summary(
                merged_states,
                merged_events,
            )

            positive_types = list(type_summary.keys())

            print(
                "Aggregate literal behavior: "
                + (
                    ", ".join(positive_types)
                    if positive_types
                    else "none"
                ),
                flush=True,
            )

            elapsed = time.time() - full_started

            peak_allocated = (
                max(peak_alloc_values)
                if peak_alloc_values
                else None
            )
            peak_reserved = (
                max(peak_reserved_values)
                if peak_reserved_values
                else None
            )

            # ---------------------------------------------------------
            # 7. Save segment-level output. NO clinical judge.
            # ---------------------------------------------------------
            output_row = {
                "segment_idx": segment_idx,
                "segment_path": str(segment_path),
                "video": getattr(row, "video", ""),
                "patient_id": getattr(row, "patient_id", ""),
                "session_id": getattr(row, "session_id", ""),
                "segment_id": getattr(row, "segment_id", ""),
                "segment_start_sec": getattr(
                    row,
                    "segment_start_sec",
                    "",
                ),
                "segment_duration_sec": getattr(
                    row,
                    "segment_duration_sec",
                    "",
                ),
                "status": "ok",
                "error": "",
                "model": args.model_id,
                "purpose": "expanded_literal_visual_perception_only",
                "patient_track_id": selected_track["track_id"],
                "patient_selection_method": selection_method,
                "patient_side": patient_side,
                "therapist_side_for_gaze": therapist_side,
                "face_candidate_count": len(tracks),
                "patient_roi": json.dumps(patient_roi),
                "state_intervals": json.dumps(
                    merged_states,
                    ensure_ascii=False,
                ),
                "events": json.dumps(
                    merged_events,
                    ensure_ascii=False,
                ),
                "type_summary": json.dumps(
                    type_summary,
                    ensure_ascii=False,
                ),
                "detected_behavior_count": len(positive_types),
                "detected_behavior_types": ", ".join(positive_types),
                "window_observations": json.dumps(
                    window_results,
                    ensure_ascii=False,
                ),
                "duration_sec": round(duration, 3),
                "frames_sampled": len(patient_frames),
                "sample_fps": args.sample_fps,
                "window_seconds": args.window_seconds,
                "frame_width": args.frame_width,
                "window_video_token_budget": (
                    args.window_video_token_budget
                ),
                "peak_allocated_gb": (
                    round(peak_allocated, 3)
                    if peak_allocated is not None
                    else None
                ),
                "peak_reserved_gb": (
                    round(peak_reserved, 3)
                    if peak_reserved is not None
                    else None
                ),
                "observer_total_sec": round(
                    total_observer_sec,
                    3,
                ),
                "elapsed_sec": round(elapsed, 3),
            }

            rows = [
                old
                for old in rows
                if int(float(old["segment_idx"])) != segment_idx
            ]
            rows.append(output_row)
            v5.save_rows(rows, csv_path)

            # Flat detection table: convenient for benchmarking.
            detection_rows = [
                old
                for old in detection_rows
                if int(float(old["segment_idx"])) != segment_idx
            ]
            detection_rows.extend(
                detection_rows_for_segment(
                    segment_idx,
                    row,
                    merged_states,
                    merged_events,
                )
            )
            save_detection_rows(
                detection_rows,
                detections_path,
            )

            detail = {
                "experiment": (
                    "Visual Experiment V6.3 - "
                    "Qwen8 expanded literal perception"
                ),
                "segment_idx": segment_idx,
                "segment_path": str(segment_path),
                "patient_tracking": {
                    "candidates": tracks,
                    "selected_track_id": (
                        selected_track["track_id"]
                    ),
                    "selection_method": selection_method,
                    "patient_side": patient_side,
                    "therapist_side_for_gaze": therapist_side,
                    "patient_roi": patient_roi,
                    "track_preview": str(track_preview_path),
                    "patient_preview": str(patient_preview_path),
                },
                "settings": {
                    "model_id": args.model_id,
                    "sample_fps": args.sample_fps,
                    "window_seconds": args.window_seconds,
                    "frame_width": args.frame_width,
                    "window_video_token_budget": (
                        args.window_video_token_budget
                    ),
                    "window_total_pixels": window_total_pixels,
                    "patient_selection": args.patient_selection,
                    "state_merge_gap_sec": (
                        args.state_merge_gap_sec
                    ),
                    "event_dedup_sec": args.event_dedup_sec,
                },
                "window_observations": window_details,
                "merged_states": merged_states,
                "merged_events": merged_events,
                "type_summary": type_summary,
                "peak_allocated_gb": peak_allocated,
                "peak_reserved_gb": peak_reserved,
                "observer_total_sec": total_observer_sec,
                "elapsed_sec": elapsed,
            }

            with jsonl_path.open(
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    json.dumps(
                        detail,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            completed.add(segment_idx)

            print(
                f"Perception complete | "
                f"states={len(merged_states)} | "
                f"events={len(merged_events)} | "
                f"types={len(positive_types)} | "
                f"observer={total_observer_sec:.1f}s | "
                f"total={elapsed:.1f}s",
                flush=True,
            )

            if peak_allocated is not None:
                print(
                    f"Peak allocated VRAM: "
                    f"{peak_allocated:.2f} GiB",
                    flush=True,
                )

            del full_frames
            del patient_frames

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            elapsed = time.time() - full_started

            print(
                f"ERROR segment {segment_idx}: {exc}",
                flush=True,
            )
            traceback.print_exc()

            output_row = {
                "segment_idx": segment_idx,
                "segment_path": str(segment_path),
                "video": getattr(row, "video", ""),
                "patient_id": getattr(row, "patient_id", ""),
                "session_id": getattr(row, "session_id", ""),
                "segment_id": getattr(row, "segment_id", ""),
                "status": "error",
                "error": repr(exc),
                "model": args.model_id,
                "purpose": "expanded_literal_visual_perception_only",
                "elapsed_sec": round(elapsed, 3),
            }

            rows = [
                old
                for old in rows
                if int(float(old["segment_idx"])) != segment_idx
            ]
            rows.append(output_row)
            v5.save_rows(rows, csv_path)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nFinished V6.3", flush=True)
    print(f"CSV: {csv_path}", flush=True)
    print(f"Flat detections: {detections_path}", flush=True)
    print(f"JSONL: {jsonl_path}", flush=True)


# =====================================================================
# CLI
# =====================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Visual Experiment V6.3: Qwen3-VL-8B expanded literal "
            "patient behavior perception with explicit states/events."
        )
    )

    parser.add_argument(
        "--segments-csv",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="./output/qwen3vl_visual_experiment_v6_3",
    )

    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--segment-indices",
        default="4,10,16,63,65",
        help="Comma-separated diagnostic segment indices.",
    )

    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
    )

    # Keep V5 temporal settings for controlled comparison.
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--window-seconds",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--frame-width",
        type=int,
        default=320,
    )

    parser.add_argument(
        "--window-video-token-budget",
        type=int,
        default=4096,
    )

    # Reuse V5 patient role cache by default.
    parser.add_argument(
        "--role-cache",
        default=(
            "./output/qwen3vl_visual_experiment_v5/"
            "patient_role_cache.json"
        ),
    )

    # YuNet.
    parser.add_argument(
        "--yunet-model",
        default="./models/face_detection_yunet_2026may.onnx",
    )

    parser.add_argument(
        "--yunet-score-threshold",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--yunet-nms-threshold",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--yunet-top-k",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--min-face-px",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--track-center-threshold",
        type=float,
        default=0.14,
    )

    parser.add_argument(
        "--min-track-detections",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--min-track-fraction",
        type=float,
        default=0.08,
    )

    # Patient ROI: same V5 settings.
    parser.add_argument(
        "--roi-width-face-mult",
        type=float,
        default=4.5,
    )

    parser.add_argument(
        "--roi-top-face-mult",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--roi-bottom-face-mult",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--patient-selection",
        choices=[
            "interactive",
            "leftmost",
            "rightmost",
            "largest",
        ],
        default="interactive",
    )

    parser.add_argument(
        "--patient-side",
        choices=["left", "right"],
        default=None,
    )

    # Generation.
    parser.add_argument(
        "--observer-max-new-tokens",
        type=int,
        default=1100,
        help=(
            "Expanded schema needs more room than V5. "
            "This is only a maximum, not a required output length."
        ),
    )

    # Deterministic post-processing.
    parser.add_argument(
        "--state-merge-gap-sec",
        type=float,
        default=1.1,
        help=(
            "Merge same-type state intervals separated by <= this gap."
        ),
    )

    parser.add_argument(
        "--event-dedup-sec",
        type=float,
        default=0.75,
        help=(
            "Merge duplicate same-type events separated by <= this gap."
        ),
    )

    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())