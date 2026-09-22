#!/usr/bin/env python
"""
InternVL3-8B literal behavior detection benchmark
==================================================

PURPOSE
-------
Compare VLM visual-behavior detection against the 5-segment manual gold
standard. This script does NOT classify rupture.

Model:
    OpenGVLab/InternVL3-8B

Input:
    patient-focused crop
    2 FPS
    four 15-second windows
    up to 30 frames/window by default

Output:
    one flat row per detected literal behavior interval/event

No human labels are used in the prompt.
No AU/OpenFace/pose features are used.
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
import torchvision.transforms as T
import transformers
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------
# V5 contains only shared video / face-track infrastructure here.
# ---------------------------------------------------------------------
if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
    class _UnavailableQwen3VLForConditionalGeneration:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError(
                "Qwen3-VL is unavailable in this environment. "
                "This script imports V5 only for video/tracking utilities."
            )
    transformers.Qwen3VLForConditionalGeneration = (
        _UnavailableQwen3VLForConditionalGeneration
    )

import run_qwen3vl_visual_experiment_v5 as v5


DEFAULT_MODEL = "OpenGVLab/InternVL3-8B"
SUPPORTED_TRANSFORMERS = "4.37.2"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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
    "OTHER": ["other_literal_visual_event"],
}

ALLOWED_CODES = {
    code for group in BEHAVIOR_GROUPS.values() for code in group
}

CERTAINTY = {"clear", "possible", "ambiguous"}
VISIBILITY = {"good", "partial", "poor"}


def build_transform(input_size=448):
    return T.Compose(
        [
            T.Lambda(
                lambda image: image.convert("RGB")
                if image.mode != "RGB"
                else image
            ),
            T.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def limit_frames_evenly(frames, max_frames):
    frames = list(frames)
    if max_frames is None or len(frames) <= max_frames:
        return frames

    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")

    if max_frames == 1:
        return [frames[len(frames) // 2]]

    last = len(frames) - 1
    ids = [
        round(i * last / (max_frames - 1))
        for i in range(max_frames)
    ]
    return [frames[i] for i in ids]


class InternVLRunner:
    def __init__(self, model_id, load_in_8bit=False, use_flash_attn=False):
        if transformers.__version__ != SUPPORTED_TRANSFORMERS:
            raise RuntimeError(
                f"InternVL3-8B is pinned to transformers=="
                f"{SUPPORTED_TRANSFORMERS}, but installed version is "
                f"{transformers.__version__}."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required.")

        self.model_id = model_id
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda:0")
        self.transform = build_transform(448)

        kwargs = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "use_flash_attn": bool(use_flash_attn),
        }

        if load_in_8bit:
            kwargs["load_in_8bit"] = True
            kwargs["device_map"] = "auto"

        print(f"Loading {model_id}", flush=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            **kwargs,
        ).eval()

        if not load_in_8bit:
            self.model = self.model.cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )

    def generate(self, prompt, frames=None, max_new_tokens=1100):
        """
        Supports both video+text inference and text-only JSON repair.
        """
        frames = list(frames or [])

        pixel_values = None
        num_patches_list = None
        question = prompt

        if frames:
            tensors = [self.transform(frame) for frame in frames]
            pixel_values = torch.stack(tensors).to(
                device=self.device,
                dtype=self.dtype,
            )

            num_patches_list = [1] * len(frames)
            video_prefix = "".join(
                f"Frame{i + 1}: <image>\\n"
                for i in range(len(frames))
            )
            question = video_prefix + prompt

        generation_config = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
        }

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.time()

        chat_kwargs = {
            "tokenizer": self.tokenizer,
            "pixel_values": pixel_values,
            "question": question,
            "generation_config": generation_config,
            "history": None,
            "return_history": False,
        }

        if num_patches_list is not None:
            chat_kwargs["num_patches_list"] = num_patches_list

        with torch.inference_mode():
            response = self.model.chat(**chat_kwargs)

        torch.cuda.synchronize()
        elapsed = time.time() - started
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)

        if pixel_values is not None:
            del pixel_values

        return str(response), elapsed, peak_alloc, peak_reserved



def vocabulary_text():
    blocks = []
    for group, codes in BEHAVIOR_GROUPS.items():
        blocks.append(
            group + ":\n" + "\n".join(f"- {x}" for x in codes)
        )
    return "\n\n".join(blocks)


def therapist_context(side):
    if side in {"left", "right"}:
        return (
            f"The therapist is on IMAGE-{side.upper()} in the original "
            "scene. Judge gaze toward/away from therapist only when the "
            "eyes are sufficiently visible. Head direction alone is not "
            "exact eye gaze."
        )
    return (
        "The therapist side is unknown. Do NOT use "
        "gaze_toward_therapist or gaze_away_from_therapist. "
        "Use image-relative or vertical gaze codes instead."
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

def _extra_json_cleanup(candidate):
    candidate = str(candidate or "").strip()

    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
    candidate = re.sub(r"\s*```$", "", candidate)

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        candidate = candidate[start:end + 1]

    candidate = re.sub(r"}\s*{", "},{", candidate)

    candidate = re.sub(
        r'([}\]])\s*(?="[^"\n]+"\s*:)',
        r"\1,",
        candidate,
    )

    candidate = re.sub(
        r'([-+]?\d+(?:\.\d+)?)\s+(?="[^"\n]+"\s*:)',
        r"\1, ",
        candidate,
    )

    return candidate


def extract_json(text, model=None, max_repair_tokens=1600):
    """
    Robust JSON recovery for InternVL behavior outputs.

    1. strict JSON
    2. deterministic cleanup
    3. formatting-only InternVL repair
    4. compact InternVL reconstruction
    """
    raw = str(text or "").strip()

    if not raw:
        raise ValueError("Model returned an empty response.")

    try:
        return json.loads(raw), "strict"
    except Exception:
        pass

    candidate = _extra_json_cleanup(raw)

    try:
        return json.loads(candidate), "deterministic_repair"
    except Exception as deterministic_error:
        if model is None:
            raise deterministic_error

    repair_prompt = f"""
Repair the malformed JSON below.

STRICT RULES:
- Return valid JSON only.
- Preserve the exact same visual observations and timestamps.
- Do NOT add any new visual behavior.
- Do NOT reinterpret any observation.
- Do NOT add rupture, withdrawal, confrontation, emotion, or intention.
- Keep the same top-level keys:
  visibility, states, events, unresolved
- Each state/event may use only:
  type, start_sec, end_sec, certainty, description
- Remove markdown/code fences.
- If an entry is duplicated, keep only one copy.

MALFORMED JSON:
{candidate}
""".strip()

    repaired_raw, _, _, _ = model.generate(
        repair_prompt,
        frames=[],
        max_new_tokens=max_repair_tokens,
    )

    repaired = _extra_json_cleanup(repaired_raw)

    try:
        return json.loads(repaired), "internvl_json_repair"
    except Exception:
        pass

    compact_prompt = f"""
Convert the malformed JSON below into compact VALID JSON.

RULES:
- JSON only.
- Do not invent observations.
- Preserve timestamps when possible.
- Keep only valid entries.
- Use:
  {{
    "visibility": "good|partial|poor",
    "states": [],
    "events": [],
    "unresolved": []
  }}
- Each state/event may contain only:
  type, start_sec, end_sec, certainty, description
- If one broken item cannot be recovered, omit only that item and add
  "one malformed item omitted during JSON recovery" to unresolved.

MALFORMED JSON:
{candidate}
""".strip()

    compact_raw, _, _, _ = model.generate(
        compact_prompt,
        frames=[],
        max_new_tokens=1200,
    )

    compact = _extra_json_cleanup(compact_raw)
    return json.loads(compact), "internvl_compact_repair"


def parse_delimited_output(raw_text, window_start, window_end):
    """
    Parse the pipe-delimited output format.

    Returns:
        parsed_dict, parse_method
    """
    raw = str(raw_text or "").strip()
    if not raw:
        raise ValueError("Model returned an empty response.")

    visibility = "partial"
    states = []
    events = []
    unresolved = []
    parsed_any_record = False

    # Strip common accidental markdown fences.
    raw = re.sub(r"^```(?:text)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)

    for original_line in raw.splitlines():
        line = original_line.strip()

        if not line:
            continue

        # Remove accidental bullets/numbering.
        line = re.sub(r"^[\-\*\u2022]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)

        parts = [p.strip() for p in line.split("|")]
        if not parts:
            continue

        tag = parts[0].upper()

        if tag == "VISIBILITY" and len(parts) >= 2:
            candidate = parts[1].lower()
            if candidate in VISIBILITY:
                visibility = candidate
                parsed_any_record = True
            continue

        if tag == "UNRESOLVED" and len(parts) >= 2:
            unresolved.append("|".join(parts[1:]).strip())
            parsed_any_record = True
            continue

        if tag not in {"STATE", "EVENT"}:
            continue

        if len(parts) < 6:
            continue

        code = parts[1]
        if code not in ALLOWED_CODES:
            continue

        try:
            start_sec = float(parts[2])
            end_sec = float(parts[3])
        except Exception:
            continue

        certainty = parts[4].lower()
        if certainty not in CERTAINTY:
            certainty = "possible"

        description = "|".join(parts[5:]).strip()

        item = {
            "type": code,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "certainty": certainty,
            "description": description,
        }

        if tag == "STATE":
            states.append(item)
        else:
            events.append(item)

        parsed_any_record = True

    if not parsed_any_record:
        raise ValueError(
            "No valid pipe-delimited behavior records could be parsed."
        )

    return {
        "visibility": visibility,
        "states": states,
        "events": events,
        "unresolved": unresolved,
    }, "pipe_delimited"


def normalize_time(x, start, end):
    try:
        t = float(x)
    except Exception:
        return None

    if not math.isfinite(t):
        return None

    duration = end - start

    # global
    if start - 0.75 <= t <= end + 0.75:
        return round(max(start, min(t, end)), 3)

    # local window time
    if -0.25 <= t <= duration + 0.75:
        return round(max(start, min(start + t, end)), 3)

    return None


def normalize_item(item, kind, start, end):
    if not isinstance(item, dict):
        return None

    code = str(item.get("type", "")).strip()
    if code not in ALLOWED_CODES:
        return None

    s = normalize_time(
        item.get("start_sec", item.get("time_sec")),
        start,
        end,
    )
    e = normalize_time(
        item.get("end_sec", item.get("time_sec")),
        start,
        end,
    )

    if s is None and e is None:
        return None
    if s is None:
        s = e
    if e is None:
        e = s
    if e < s:
        s, e = e, s

    cert = str(item.get("certainty", "possible")).lower().strip()
    if cert not in CERTAINTY:
        cert = "possible"

    return {
        "kind": kind,
        "behavior_code": code,
        "start_sec": float(s),
        "end_sec": float(e),
        "certainty": cert,
        "description": str(item.get("description", "")).strip(),
    }


def normalize_window(parsed, start, end):
    states = []
    events = []

    for x in parsed.get("states", []) or []:
        y = normalize_item(x, "state", start, end)
        if y:
            states.append(y)

    for x in parsed.get("events", []) or []:
        y = normalize_item(x, "event", start, end)
        if y:
            events.append(y)

    return states, events


def merge_intervals(rows, gap=1.1):
    if not rows:
        return []

    # Merge only same kind + behavior.
    groups = {}
    for row in rows:
        key = (row["kind"], row["behavior_code"])
        groups.setdefault(key, []).append(dict(row))

    merged = []

    for key, items in groups.items():
        items.sort(key=lambda x: (x["start_sec"], x["end_sec"]))
        cur = dict(items[0])

        for nxt in items[1:]:
            if nxt["start_sec"] - cur["end_sec"] <= gap:
                cur["end_sec"] = max(cur["end_sec"], nxt["end_sec"])
                if nxt["description"] and nxt["description"] not in cur["description"]:
                    cur["description"] = (
                        cur["description"] + " | " + nxt["description"]
                    ).strip(" |")
            else:
                merged.append(cur)
                cur = dict(nxt)

        merged.append(cur)

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


def load_done(path):
    if not Path(path).exists():
        return pd.DataFrame(), set()

    df = pd.read_csv(path)
    if df.empty or "segment_idx" not in df.columns:
        return df, set()

    return df, set(
        pd.to_numeric(df["segment_idx"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )


def save_rows(df, rows, path):
    if rows:
        new = pd.DataFrame(rows)
        out = pd.concat([df, new], ignore_index=True)
    else:
        out = df

    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detections_csv = out_dir / "internvl3_8b_behavior_detections.csv"
    details_jsonl = out_dir / "internvl3_8b_behavior_details.jsonl"

    role_cache = v5.load_role_cache(Path(args.role_cache))

    segments = pd.read_csv(args.segments_csv)
    segments["segment_idx"] = pd.to_numeric(
        segments["segment_idx"], errors="raise"
    ).astype(int)

    requested = v5.parse_segment_indices(args.segment_indices)
    if requested is not None:
        segments = segments[
            segments["segment_idx"].isin(requested)
        ].copy()

    face_detector = v5.get_face_detector(
        model_path=args.yunet_model,
        score_threshold=args.yunet_score_threshold,
        nms_threshold=args.yunet_nms_threshold,
        top_k=args.yunet_top_k,
    )

    existing, done = load_done(detections_csv)

    print("")
    print("INTERNVL3-8B BEHAVIOR DETECTION BENCHMARK")
    print("=" * 48)
    print(f"Model: {args.model_id}")
    print(f"Segments: {len(segments)}")
    print(f"Sampling: {args.sample_fps} FPS")
    print(f"Window: {args.window_seconds}s")
    print(f"Max frames/window: {args.window_max_frames}")
    print("NO rupture classification.")
    print("NO AU/OpenFace/pose features.")
    print("")

    model = InternVLRunner(
        args.model_id,
        load_in_8bit=args.load_in_8bit,
        use_flash_attn=args.use_flash_attn,
    )

    for pos, row in enumerate(
        segments.itertuples(index=False),
        start=1,
    ):
        idx = int(row.segment_idx)

        if idx in done:
            print(
                f"[{pos}/{len(segments)}] segment {idx}: already done"
            )
            continue

        path = Path(row.segment_path)
        print(
            f"\n[{pos}/{len(segments)}] segment {idx}: {path.name}"
        )

        started = time.time()

        try:
            full_frames, timestamps, duration = v5.sample_full_frames(
                path,
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
                raise RuntimeError("No persistent face candidate.")

            preview = (
                out_dir
                / "track_previews"
                / f"segment_{idx:03d}_tracks.jpg"
            )
            v5.annotate_tracks(
                full_frames,
                timestamps,
                tracks,
                preview,
                max_frames=4,
            )

            selected, method = v5.select_patient_track(
                tracks=tracks,
                row=row,
                preview_path=preview,
                cache=role_cache,
                cache_path=Path(args.role_cache),
                mode=args.patient_selection,
                forced_side=args.patient_side,
            )

            therapist_side = therapist_side_from_tracks(
                tracks,
                selected,
            )

            roi = v5.face_to_person_roi(
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
                roi,
                frame_width=args.frame_width,
            )

            windows = v5.split_windows(
                patient_frames,
                timestamps,
                window_seconds=args.window_seconds,
                max_duration=min(args.max_duration, duration),
            )

            segment_rows = []
            segment_detail = {
                "segment_idx": idx,
                "segment_path": str(path),
                "patient_track_id": selected["track_id"],
                "patient_selection_method": method,
                "therapist_side": therapist_side,
                "windows": [],
            }

            for w_no, w in enumerate(windows, start=1):
                frames = limit_frames_evenly(
                    w["frames"],
                    args.window_max_frames,
                )

                print(
                    f"  window {w_no}/{len(windows)} "
                    f"{w['start']:.0f}-{w['end']:.0f}s | "
                    f"{len(frames)} frames",
                    flush=True,
                )

                raw, sec, pa, pr = model.generate(
                    observer_prompt(
                        w["start"],
                        w["end"],
                        therapist_side,
                    ),
                    frames,
                    max_new_tokens=args.max_new_tokens,
                )

                try:
                    parsed, parse_method = parse_delimited_output(
                        raw,
                        w["start"],
                        w["end"],
                    )
                except Exception as pipe_exc:
                    # Backward-compatible fallback if the model unexpectedly
                    # emits JSON despite the plain-text instruction.
                    try:
                        parsed, parse_method = extract_json(
                            raw,
                            model=model,
                        )
                        parse_method = "json_fallback_" + parse_method
                    except Exception as json_exc:
                        failure_dir = out_dir / "parse_failures"
                        failure_dir.mkdir(parents=True, exist_ok=True)
                        failure_path = (
                            failure_dir
                            / f"segment_{idx:03d}_window_{w_no:02d}_raw.txt"
                        )
                        failure_path.write_text(raw, encoding="utf-8")
                        raise RuntimeError(
                            f"Could not parse behavior output for segment {idx}, "
                            f"window {w_no}. Raw output saved to {failure_path}"
                        ) from json_exc

                states, events = normalize_window(
                    parsed,
                    w["start"],
                    w["end"],
                )

                detected = [
                    x["behavior_code"]
                    for x in states + events
                ]
                print(
                    "    "
                    + (
                        ", ".join(dict.fromkeys(detected))
                        if detected
                        else "none"
                    )
                    + f" | {sec:.1f}s",
                    flush=True,
                )

                segment_rows.extend(states)
                segment_rows.extend(events)

                segment_detail["windows"].append(
                    {
                        "window_start": w["start"],
                        "window_end": w["end"],
                        "parse_method": parse_method,
                        "parsed": parsed,
                        "raw": raw,
                        "inference_sec": sec,
                        "peak_allocated_gb": pa,
                        "peak_reserved_gb": pr,
                    }
                )

            segment_rows = merge_intervals(
                segment_rows,
                gap=args.merge_gap_sec,
            )

            flat = []
            for x in segment_rows:
                flat.append(
                    {
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
                    }
                )

            existing = save_rows(
                existing,
                flat,
                detections_csv,
            )

            segment_detail["merged_detections"] = flat
            segment_detail["elapsed_sec"] = time.time() - started

            with details_jsonl.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        segment_detail,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            done.add(idx)

            print(
                f"  complete: {len(flat)} detections | "
                f"{time.time() - started:.1f}s"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            print(f"ERROR segment {idx}: {exc}")
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("")
    print("Finished.")
    print(f"Detections: {detections_csv}")
    print(f"Details: {details_jsonl}")


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--segments-csv",
        required=True,
    )
    p.add_argument(
        "--output-dir",
        default="./output/internvl3_8b_behavior_benchmark_v1_3",
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
    p.add_argument("--max-new-tokens", type=int, default=1100)
    p.add_argument("--merge-gap-sec", type=float, default=1.1)

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

    p.add_argument(
        "--load-in-8bit",
        action="store_true",
    )
    p.add_argument(
        "--use-flash-attn",
        action="store_true",
    )

    return p


if __name__ == "__main__":
    run(build_parser().parse_args())