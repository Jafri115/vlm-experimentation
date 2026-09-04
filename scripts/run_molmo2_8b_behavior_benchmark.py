#!/usr/bin/env python
"""
Molmo2-8B — literal behavior detection benchmark
================================================

Model:
    allenai/Molmo2-8B

Goal:
Run the SAME 5-clip behavior benchmark used for Qwen3-VL-8B,
InternVL3-8B, and MiniCPM-V 4.5.

Fairness:
- same clips: 4,10,16,63,65
- same patient-focused crop
- same 2 FPS
- same four 15-second windows
- same <=30 frames/window
- same literal behavior vocabulary
- same pipe-delimited output format
- no human labels in prompt
- no rupture classification
- no AU/OpenFace/pose behavior features

Implementation note
-------------------
Molmo2's official Transformers interface accepts a video object/path through
AutoProcessor.apply_chat_template(). To preserve the exact patient crop and
sampling used in the other runs, each 15-second cropped window is written to a
small temporary local MP4 at 2 FPS, then passed to the Molmo2 processor.

No internet video is used during inference.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoProcessor, AutoModelForImageTextToText

# Reuse only stable video/face-track/crop utilities.
if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
    class _UnavailableQwen3VLForConditionalGeneration:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError(
                "Qwen3-VL is unavailable in this environment. "
                "This runner imports V5 only for video/crop utilities."
            )

    transformers.Qwen3VLForConditionalGeneration = (
        _UnavailableQwen3VLForConditionalGeneration
    )

import run_qwen3vl_visual_experiment_v5 as v5


DEFAULT_MODEL = "allenai/Molmo2-8B"

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

ALLOWED_CODES = {x for xs in BEHAVIOR_GROUPS.values() for x in xs}
CERTAINTY = {"clear", "possible", "ambiguous"}
VISIBILITY = {"good", "partial", "poor"}

ALIASES = {
    "hand_clasped": "hands_clasped",
    "hands_clasp": "hands_clasped",
    "hand_on_lap": "hands_on_lap",
    "hand_on_table": "hands_on_table",
    "shoulder_elevate": "shoulders_elevate",
    "shoulder_lower": "shoulders_lower",
}


def limit_frames_and_times(frames, timestamps, max_frames):
    frames = list(frames)
    timestamps = list(timestamps)

    if max_frames is None or len(frames) <= max_frames:
        return frames, timestamps

    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")

    if max_frames == 1:
        i = len(frames) // 2
        return [frames[i]], [timestamps[i]]

    last = len(frames) - 1
    ids = [round(i * last / (max_frames - 1)) for i in range(max_frames)]

    return [frames[i] for i in ids], [timestamps[i] for i in ids]


def vocabulary_text():
    blocks = []
    for group, codes in BEHAVIOR_GROUPS.items():
        blocks.append(group + ":\n" + "\n".join(f"- {x}" for x in codes))
    return "\n\n".join(blocks)


def therapist_context(side):
    if side in {"left", "right"}:
        return (
            f"The therapist is on IMAGE-{side.upper()} in the original scene. "
            "Use gaze_toward_therapist / gaze_away_from_therapist only when "
            "the patient's eyes are sufficiently visible. Head direction "
            "alone is not exact eye gaze."
        )

    return (
        "The therapist's image-side is UNKNOWN. "
        "Do NOT use gaze_toward_therapist or gaze_away_from_therapist. "
        "Use image-relative / vertical gaze codes or gaze_uncertain."
    )


def observer_prompt(window_start, window_end, therapist_side):
    return f"""
You are a HIGH-RECALL LITERAL VISUAL BEHAVIOR OBSERVER.

You see ONLY the PATIENT from approximately
{window_start:.1f}-{window_end:.1f} seconds of psychotherapy video.

There is NO audio and NO transcript.

{therapist_context(therapist_side)}

Your task is ONLY literal visible behavior detection.

DO NOT classify or discuss:
- rupture
- withdrawal
- confrontation
- resistance
- alliance quality
- emotion
- intention
- motivation
- speech content

STATE VS EVENT
--------------
STATE = behavior persists for an interval.
Examples:
- gaze remains down
- head remains angled down
- hand remains on cheek
- hands remain clasped
- body remains unusually still

EVENT = a discrete visible movement/change.
Examples:
- arm raises
- head turns
- hand moves to eye
- shoulder lifts/drops
- pointing gesture

Do NOT create one event per sampled frame.
Repeated actions such as rubbing/tapping should be one interval.

IMPORTANT LITERAL RULES
-----------------------
- hand_to_eye_region = visible hand contact/passage over eye/upper cheek.
  It does NOT automatically mean crying or tear wiping.
- crying_visible requires visually clear crying behavior.
- tear_visible requires a visible tear/tear track.
- gaze is separate from head orientation.
- if gaze cannot be determined, use gaze_uncertain.
- shoulder_lift_drop requires a visible lift followed by a drop.
- do not infer hidden/occluded behavior.

ALLOWED CODES
-------------
{vocabulary_text()}

OUTPUT FORMAT — IMPORTANT
-------------------------
DO NOT RETURN JSON.

Return plain text, exactly ONE record per line, using | as separator.

First line:
VISIBILITY|good
or:
VISIBILITY|partial
or:
VISIBILITY|poor

Then zero or more behavior lines:

STATE|behavior_code|start_sec|end_sec|certainty|short literal description

EVENT|behavior_code|start_sec|end_sec|certainty|short literal description

Optional unresolved line:
UNRESOLVED|short note

Examples:
STATE|gaze_down|15.0|20.0|clear|eyes remain directed downward
EVENT|arm_raise|21.5|22.0|clear|right arm rises
STATE|hand_to_chin|24.0|29.0|possible|hand appears to remain at chin

Rules:
- use GLOBAL times within the 60-second clip
- certainty must be clear, possible, or ambiguous
- use only allowed behavior codes
- do not use commas or JSON syntax
- do not add bullets, numbering, markdown, or explanation
- if no behavior is detected, return only the VISIBILITY line
""".strip()


def write_window_video(frames, path, fps):
    frames = list(frames)
    if not frames:
        raise ValueError("No frames to write.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    first = np.array(frames[0].convert("RGB"))
    h, w = first.shape[:2]

    # H.264 availability is variable on Windows OpenCV, so use mp4v.
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create temporary MP4: {path}")

    try:
        for image in frames:
            arr = np.array(image.convert("RGB"))
            if arr.shape[1] != w or arr.shape[0] != h:
                arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


class Molmo2Runner:
    def __init__(
        self,
        model_id,
        dtype="bfloat16",
        device="cuda:0",
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Molmo2 benchmark requires CUDA.")

        self.device = torch.device(device)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        self.dtype = dtype_map[dtype]

        print(
            f"Loading {model_id} | transformers={transformers.__version__} "
            f"| dtype={dtype}",
            flush=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).eval().to(self.device)

    def generate(self, video_path, prompt, max_new_tokens=1200):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video", "video": str(video_path)},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        )

        inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        started = time.time()

        with torch.inference_mode(), torch.autocast(
            "cuda",
            dtype=self.dtype,
        ):
            output = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
            )

        torch.cuda.synchronize()

        elapsed = time.time() - started
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

        generated_tokens = output[0, inputs["input_ids"].shape[1]:]
        raw = self.processor.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        del inputs, output, generated_tokens

        return str(raw), elapsed, peak_alloc, peak_reserved


def parse_time(value):
    s = str(value).strip().lower()
    s = re.sub(r"\s*(seconds?|secs?|sec|s)$", "", s)
    try:
        x = float(s)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def normalize_time_pair(raw_start, raw_end, window_start, window_end):
    s = parse_time(raw_start)
    e = parse_time(raw_end)

    if s is None or e is None:
        return None, None, "non_numeric_time"

    duration = float(window_end - window_start)
    eps = 0.26

    def in_global(t):
        return window_start - eps <= t <= window_end + eps

    def in_local(t):
        return -eps <= t <= duration + eps

    if in_global(s) and in_global(e):
        ns, ne = s, e
    elif in_local(s) and in_local(e):
        ns, ne = window_start + s, window_start + e
    else:
        return None, None, "impossible_or_ambiguous_time"

    ns = max(window_start, min(ns, window_end))
    ne = max(window_start, min(ne, window_end))

    if ne < ns:
        return None, None, "end_before_start"

    return round(ns, 3), round(ne, 3), None


def parse_pipe_output(raw_text, window_start, window_end):
    raw = str(raw_text or "").strip()
    if not raw:
        raise ValueError("Molmo2 returned an empty response.")

    raw = re.sub(r"^```(?:text)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)

    visibility = "partial"
    rows = []
    rejected = []
    unresolved = []

    for line_no, original_line in enumerate(raw.splitlines(), start=1):
        line = original_line.strip()
        if not line:
            continue

        line = re.sub(r"^[\-\*\u2022]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)

        parts = [p.strip() for p in line.split("|")]
        if not parts:
            continue

        tag = parts[0].upper()

        if tag == "VISIBILITY":
            if len(parts) >= 2 and parts[1].lower() in VISIBILITY:
                visibility = parts[1].lower()
            continue

        if tag == "UNRESOLVED":
            unresolved.append("|".join(parts[1:]).strip())
            continue

        if tag not in {"STATE", "EVENT"}:
            rejected.append({
                "line_no": line_no,
                "raw_line": original_line,
                "reason": "unknown_record_type",
            })
            continue

        if len(parts) < 6:
            rejected.append({
                "line_no": line_no,
                "raw_line": original_line,
                "reason": "too_few_fields",
            })
            continue

        raw_code = parts[1]
        code = ALIASES.get(raw_code, raw_code)

        if code not in ALLOWED_CODES:
            rejected.append({
                "line_no": line_no,
                "raw_line": original_line,
                "reason": f"unknown_behavior_code:{raw_code}",
            })
            continue

        start_sec, end_sec, error = normalize_time_pair(
            parts[2],
            parts[3],
            float(window_start),
            float(window_end),
        )

        if error:
            rejected.append({
                "line_no": line_no,
                "raw_line": original_line,
                "reason": error,
            })
            continue

        certainty = parts[4].lower()
        if certainty not in CERTAINTY:
            certainty = "possible"

        rows.append({
            "kind": tag.lower(),
            "behavior_code": code,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "certainty": certainty,
            "description": "|".join(parts[5:]).strip(),
        })

    return {
        "visibility": visibility,
        "detections": rows,
        "rejected": rejected,
        "unresolved": unresolved,
    }


def merge_intervals(rows, gap=1.1):
    if not rows:
        return []

    groups = {}
    for row in rows:
        groups.setdefault(
            (row["kind"], row["behavior_code"]),
            [],
        ).append(dict(row))

    merged = []

    for _, items in groups.items():
        items.sort(key=lambda x: (x["start_sec"], x["end_sec"]))
        current = dict(items[0])

        for nxt in items[1:]:
            if nxt["start_sec"] - current["end_sec"] <= gap:
                current["end_sec"] = max(
                    current["end_sec"],
                    nxt["end_sec"],
                )
                if (
                    nxt["description"]
                    and nxt["description"] not in current["description"]
                ):
                    current["description"] = (
                        current["description"]
                        + " | "
                        + nxt["description"]
                    ).strip(" |")
            else:
                merged.append(current)
                current = dict(nxt)

        merged.append(current)

    return sorted(
        merged,
        key=lambda x: (
            x["start_sec"],
            x["behavior_code"],
            x["kind"],
        ),
    )


def therapist_side_from_tracks(tracks, selected_track):
    others = [
        x for x in tracks
        if x["track_id"] != selected_track["track_id"]
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


def load_completed(details_jsonl):
    path = Path(details_jsonl)
    if not path.exists():
        return set()

    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if obj.get("status") == "ok":
                    done.add(int(obj["segment_idx"]))
            except Exception:
                pass
    return done


def append_jsonl(path, obj):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_detections_csv(details_jsonl, detections_csv):
    all_rows = []
    path = Path(details_jsonl)

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("status") == "ok":
                    all_rows.extend(obj.get("merged_detections", []))

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(
            ["segment_idx", "start_sec", "behavior_code"]
        )

    df.to_csv(
        detections_csv,
        index=False,
        encoding="utf-8-sig",
    )


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = out_dir / "_window_videos"
    temp_dir.mkdir(parents=True, exist_ok=True)

    details_jsonl = out_dir / "molmo2_8b_behavior_details.jsonl"
    detections_csv = out_dir / "molmo2_8b_behavior_detections.csv"
    rejected_csv = out_dir / "molmo2_8b_rejected_rows.csv"

    role_cache_path = Path(args.role_cache)
    role_cache = v5.load_role_cache(role_cache_path)

    segments = pd.read_csv(args.segments_csv)
    segments["segment_idx"] = pd.to_numeric(
        segments["segment_idx"],
        errors="raise",
    ).astype(int)

    requested = v5.parse_segment_indices(args.segment_indices)
    if requested is not None:
        segments = segments[
            segments["segment_idx"].isin(requested)
        ].copy()

    done = load_completed(details_jsonl)

    face_detector = v5.get_face_detector(
        model_path=args.yunet_model,
        score_threshold=args.yunet_score_threshold,
        nms_threshold=args.yunet_nms_threshold,
        top_k=args.yunet_top_k,
    )

    print("")
    print("MOLMO2-8B BEHAVIOR DETECTION BENCHMARK")
    print("=" * 60)
    print(f"Model: {args.model_id}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Segments: {len(segments)}")
    print(f"Sampling: {args.sample_fps} FPS")
    print(f"Window: {args.window_seconds}s")
    print(f"Max frames/window: {args.window_max_frames}")
    print("NO rupture classification.")
    print("NO AU/OpenFace/pose behavior features.")
    print("")

    model = Molmo2Runner(
        model_id=args.model_id,
        dtype=args.dtype,
    )

    all_rejected = []

    for pos, row in enumerate(segments.itertuples(index=False), start=1):
        idx = int(row.segment_idx)

        if idx in done:
            print(
                f"[{pos}/{len(segments)}] segment {idx}: already done",
                flush=True,
            )
            continue

        segment_path = Path(row.segment_path)

        print(
            f"\n[{pos}/{len(segments)}] segment {idx}: "
            f"{segment_path.name}",
            flush=True,
        )

        segment_started = time.time()

        try:
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
                    detections.append({
                        "frame_idx": frame_idx,
                        "timestamp": float(ts),
                        "bbox": face,
                    })

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
                raise RuntimeError("No persistent face candidate.")

            preview_path = (
                out_dir
                / "track_previews"
                / f"segment_{idx:03d}_tracks.jpg"
            )

            v5.annotate_tracks(
                full_frames,
                timestamps,
                tracks,
                preview_path,
                max_frames=4,
            )

            selected, selection_method = v5.select_patient_track(
                tracks=tracks,
                row=row,
                preview_path=preview_path,
                cache=role_cache,
                cache_path=role_cache_path,
                mode=args.patient_selection,
                forced_side=args.patient_side,
            )

            therapist_side = therapist_side_from_tracks(
                tracks,
                selected,
            )

            patient_roi = v5.face_to_person_roi(
                selected["median_face_bbox"],
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

            windows = v5.split_windows(
                patient_frames,
                timestamps,
                window_seconds=args.window_seconds,
                max_duration=min(args.max_duration, duration),
            )

            segment_rows = []

            detail = {
                "segment_idx": idx,
                "status": "ok",
                "segment_path": str(segment_path),
                "model": args.model_id,
                "patient_track_id": selected["track_id"],
                "patient_selection_method": selection_method,
                "therapist_side": therapist_side,
                "windows": [],
            }

            for w_no, w in enumerate(windows, start=1):
                frames, times = limit_frames_and_times(
                    w["frames"],
                    w["timestamps"],
                    args.window_max_frames,
                )

                print(
                    f"  window {w_no}/{len(windows)} "
                    f"{w['start']:.0f}-{w['end']:.0f}s | "
                    f"{len(frames)} frames",
                    flush=True,
                )

                temp_video = (
                    temp_dir
                    / f"segment_{idx:03d}_window_{w_no:02d}.mp4"
                )

                write_window_video(
                    frames,
                    temp_video,
                    fps=args.sample_fps,
                )

                prompt = observer_prompt(
                    w["start"],
                    w["end"],
                    therapist_side,
                )

                raw, sec, peak_alloc, peak_reserved = model.generate(
                    temp_video,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                )

                parsed = parse_pipe_output(
                    raw,
                    w["start"],
                    w["end"],
                )

                detected_codes = [
                    x["behavior_code"]
                    for x in parsed["detections"]
                ]

                print(
                    "    "
                    + (
                        ", ".join(dict.fromkeys(detected_codes))
                        if detected_codes
                        else "none"
                    )
                    + f" | {sec:.1f}s"
                    + f" | peak {peak_alloc:.1f} GB",
                    flush=True,
                )

                segment_rows.extend(parsed["detections"])

                for reject in parsed["rejected"]:
                    all_rejected.append({
                        "segment_idx": idx,
                        "window_start": w["start"],
                        "window_end": w["end"],
                        **reject,
                    })

                detail["windows"].append({
                    "window_start": w["start"],
                    "window_end": w["end"],
                    "frame_timestamps": times,
                    "temp_video": str(temp_video),
                    "raw": raw,
                    "parsed": parsed,
                    "inference_sec": sec,
                    "peak_allocated_gb": peak_alloc,
                    "peak_reserved_gb": peak_reserved,
                })

            segment_rows = merge_intervals(
                segment_rows,
                gap=args.merge_gap_sec,
            )

            flat = []
            for x in segment_rows:
                flat.append({
                    "segment_idx": idx,
                    "video": getattr(row, "video", ""),
                    "patient_id": getattr(row, "patient_id", ""),
                    "session_id": getattr(row, "session_id", ""),
                    "kind": x["kind"],
                    "behavior_code": x["behavior_code"],
                    "start_sec": x["start_sec"],
                    "end_sec": x["end_sec"],
                    "certainty": x["certainty"],
                    "description": x["description"],
                })

            detail["merged_detections"] = flat
            detail["elapsed_sec"] = time.time() - segment_started

            append_jsonl(details_jsonl, detail)
            done.add(idx)

            write_detections_csv(
                details_jsonl,
                detections_csv,
            )

            if all_rejected:
                pd.DataFrame(all_rejected).to_csv(
                    rejected_csv,
                    index=False,
                    encoding="utf-8-sig",
                )

            print(
                f"  complete: {len(flat)} detections | "
                f"{time.time() - segment_started:.1f}s",
                flush=True,
            )

            torch.cuda.empty_cache()

        except Exception as exc:
            traceback.print_exc()

            append_jsonl(
                details_jsonl,
                {
                    "segment_idx": idx,
                    "status": "error",
                    "segment_path": str(segment_path),
                    "model": args.model_id,
                    "error": repr(exc),
                },
            )

            print(
                f"ERROR segment {idx}: {exc}",
                flush=True,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_detections_csv(
        details_jsonl,
        detections_csv,
    )

    if all_rejected:
        pd.DataFrame(all_rejected).to_csv(
            rejected_csv,
            index=False,
            encoding="utf-8-sig",
        )

    if args.delete_temp_videos:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("")
    print("Finished.")
    print(f"Detections: {detections_csv}")
    print(f"Details: {details_jsonl}")
    print(f"Rejected rows: {rejected_csv}")


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--segments-csv", required=True)

    p.add_argument(
        "--output-dir",
        default="./output/molmo2_8b_behavior_benchmark",
    )

    p.add_argument(
        "--model-id",
        default=DEFAULT_MODEL,
    )

    p.add_argument(
        "--segment-indices",
        default="4,10,16,63,65",
    )

    p.add_argument("--sample-fps", type=float, default=2.0)
    p.add_argument("--window-seconds", type=float, default=15.0)
    p.add_argument("--window-max-frames", type=int, default=30)
    p.add_argument("--max-duration", type=float, default=60.0)
    p.add_argument("--frame-width", type=int, default=320)
    p.add_argument("--max-new-tokens", type=int, default=1200)
    p.add_argument("--merge-gap-sec", type=float, default=1.1)

    p.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
    )

    p.add_argument(
        "--role-cache",
        default=(
            "./output/qwen3vl_visual_experiment_v5/"
            "patient_role_cache.json"
        ),
    )

    p.add_argument(
        "--yunet-model",
        default="./models/face_detection_yunet_2026may.onnx",
    )

    p.add_argument("--yunet-score-threshold", type=float, default=0.60)
    p.add_argument("--yunet-nms-threshold", type=float, default=0.30)
    p.add_argument("--yunet-top-k", type=int, default=5000)
    p.add_argument("--min-face-px", type=int, default=24)
    p.add_argument("--track-center-threshold", type=float, default=0.14)
    p.add_argument("--min-track-detections", type=int, default=6)
    p.add_argument("--min-track-fraction", type=float, default=0.08)
    p.add_argument("--roi-width-face-mult", type=float, default=4.5)
    p.add_argument("--roi-top-face-mult", type=float, default=1.0)
    p.add_argument("--roi-bottom-face-mult", type=float, default=5.0)

    p.add_argument(
        "--patient-selection",
        choices=["interactive", "leftmost", "rightmost", "largest"],
        default="interactive",
    )

    p.add_argument(
        "--patient-side",
        choices=["left", "right"],
        default=None,
    )

    p.add_argument(
        "--delete-temp-videos",
        action="store_true",
    )

    return p


if __name__ == "__main__":
    run(build_parser().parse_args())