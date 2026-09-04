#!/usr/bin/env python
"""
Open-ended visual description benchmark for psychotherapy clips.

Purpose
-------
Ask each VLM what it literally sees WITHOUT giving it a behavior vocabulary,
3RS definitions, rupture labels, or human reference descriptions.

Fair common configuration
-------------------------
- same segment manifest
- same five diagnostic clips by default: 4,10,16,63,65
- same patient role cache
- same YuNet patient localization / upper-body crop
- sample source at 2 FPS
- four independent 15-second windows
- evenly retain 16 frames/window for ALL models by default
- width 320
- no audio / transcript / rupture labels
- global 0-60 second timestamps
- deterministic generation

This script reuses your already-working model runners:
  run_qwen3vl_visual_experiment_v5.py
  run_minicpm_v45_behavior_benchmark.py
  run_molmo2_8b_behavior_benchmark.py
  run_internvl35_14b_behavior_benchmark.py

Run the SAME file under the environment appropriate for each backend.
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
import transformers

# MiniCPM environments may not expose Qwen3VL, but V5 is imported only for
# stable face-track / crop utilities in those runs.
if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
    class _UnavailableQwen3VLForConditionalGeneration:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError(
                "Qwen3-VL is unavailable in this environment. "
                "Use the Qwen environment for --backend qwen."
            )
    transformers.Qwen3VLForConditionalGeneration = (
        _UnavailableQwen3VLForConditionalGeneration
    )

import run_qwen3vl_visual_experiment_v5 as v5


DEFAULT_MODELS = {
    "qwen": "Qwen/Qwen3-VL-8B-Instruct",
    "minicpm": "openbmb/MiniCPM-V-4_5",
    "molmo": "allenai/Molmo2-8B",
    "internvl35": "OpenGVLab/InternVL3_5-14B",
}

FORBIDDEN_MANIFEST_COLUMNS = {
    "human_label", "human_binary",
    "WD_P", "WD_T", "CF_P", "CF_T",
    "WD_P_mean", "WD_T_mean", "CF_P_mean", "CF_T_mean",
}


def parse_segment_indices(value):
    if not value:
        return None
    return {int(x.strip()) for x in value.split(",") if x.strip()}


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

def open_description_prompt(
    window_start,
    window_end,
    num_frames,
):
    return f"""
Watch these {num_frames} sampled images carefully.

They are ordered chronologically as:

Frame 1, Frame 2, ..., Frame {num_frames}

They represent the patient/person being observed during one 15-second portion
of a psychotherapy video.

Your task is to report ONLY what is directly and visibly happening to the
person.

IMPORTANT:
Each observation must describe ONE visible fact only.

Do NOT combine several different observations into one sentence.

For example, do NOT write:

"The person looks downward, keeps their hands in their lap, sits upright,
and moves their head slightly."

Instead report these as separate observations.

Report two kinds of visible information:

1. STABLE VISIBLE STATES
   A posture, position, orientation, or visible behavior that remains present
   across several sampled frames.

2. BRIEF VISIBLE EVENTS
   A movement or visible change that occurs over one or a few sampled frames.

Look carefully for visible information involving:

- head orientation and head movement
- visible eye or gaze direction when reasonably observable
- facial movement
- mouth or lip movement
- hands and fingers
- hand position
- hand-to-face movements
- arms
- torso orientation or movement
- legs and feet when visible
- posture
- gestures
- overall amount of visible body movement
- transitions from one visible state to another

Pay particular attention to small or brief visible changes that could easily
be missed.

A mostly still person is NOT an empty observation window.

If there are few movements, still report meaningful stable visible states such
as head orientation, gaze direction, hand position, posture, or clearly reduced
visible movement.

However, do NOT report generic scene facts that are not useful for describing
the person's visible behavior.

Do NOT report things such as:

- "the person is seated in a chair"
- clothing descriptions
- furniture
- room contents
- background objects
- physical appearance

unless they are necessary to explain an occlusion or why something cannot be
seen.

Do NOT infer:

- emotions
- intentions
- thoughts
- psychological state
- engagement
- attention
- therapeutic alliance
- rupture
- withdrawal
- confrontation
- meaning of a gesture
- meaning of facial movement
- speech content

There is no audio and no transcript.

Do NOT use a predefined behavior vocabulary or clinical terminology.

Use ordinary natural language to describe exactly what you see.

TIMING RULES:

Do NOT estimate seconds or timestamps.

Use ONLY frame numbers.

For every observation:

- start_frame = first sampled frame where the observation is visible
- end_frame = last sampled frame where the observation is visible

Frame numbers must be integers from 1 to {num_frames}.

For a stable state visible throughout the sampled sequence:

start_frame = 1
end_frame = {num_frames}

For a brief event visible only in Frames 13 and 14:

start_frame = 13
end_frame = 14

For a movement or change, do NOT automatically use the full frame range.

Use the smallest reasonable frame range in which the movement or change is
actually visible.

Use the full frame range only for a genuinely stable visible state.

If the exact first or last frame is uncertain, choose the closest reasonable
frame and set certainty to "uncertain".

OUTPUT RULES:

- One observation per JSON item.
- Do not repeat the same unchanged observation several times.
- Do not invent observations merely to increase the number of outputs.
- When several distinct visible observations are present, report them
  separately.
- Approximately 2-8 observations may be appropriate in a 15-second window,
  but there is NO required number.
- Accuracy is more important than producing many observations.

Return JSON only in exactly this structure:

{{
  "observations": [
    {{
      "start_frame": 1,
      "end_frame": {num_frames},
      "description": "one literal visible observation only",
      "certainty": "clear"
    }}
  ]
}}

certainty must be exactly one of:

- "clear"
- "uncertain"

Return:

{{"observations":[]}}

ONLY if the person is not sufficiently visible to make any reliable visual
observation.

Do not include explanations outside the JSON.
""".strip()

def extract_json_candidate(raw):
    text = str(raw).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < a:
        raise ValueError("No JSON object found.")
    candidate = text[a:b + 1]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    return candidate


def normalize_times(start, end, window_start, window_end):
    """
    Preserve valid global times.
    Convert unambiguous local 0-window_length times to global times.
    Reject impossible/ambiguous times; do not clip.
    """
    s = float(start)
    e = float(end)

    if not (math.isfinite(s) and math.isfinite(e)) or e < s:
        return None, "invalid_order_or_nonfinite"

    eps = 0.25
    if window_start - eps <= s <= window_end + eps and window_start - eps <= e <= window_end + eps:
        return (max(0.0, s), min(60.0, e)), "global"

    length = float(window_end) - float(window_start)
    if window_start > 0 and -eps <= s <= length + eps and -eps <= e <= length + eps:
        return (window_start + max(0.0, s), window_start + min(length, e)), "local_to_global"

    return None, "outside_window"
def salvage_observation_objects(raw):
    """
    Recover individually valid observation objects when the outer JSON
    is malformed (for example, a missing comma between objects).

    This does NOT modify model content. It only parses valid objects
    that the model actually generated.
    """
    text = str(raw).strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    match = re.search(
        r'"observations"\s*:\s*\[',
        text,
    )

    if not match:
        return [], [{
            "item_no": -1,
            "reason": "salvage_no_observations_array",
            "raw_item": "",
        }]

    decoder = json.JSONDecoder()
    pos = match.end()

    items = []
    rejected = []

    while pos < len(text):

        array_end = text.find("]", pos)
        obj_start = text.find("{", pos)

        if obj_start == -1:
            break

        # Do not move outside the observations array.
        if (
            array_end != -1
            and array_end < obj_start
        ):
            break

        try:
            obj, obj_end = decoder.raw_decode(
                text,
                obj_start,
            )

            if isinstance(obj, dict):
                items.append(obj)

            pos = obj_end

        except json.JSONDecodeError as exc:

            next_obj = text.find(
                "{",
                obj_start + 1,
            )

            if next_obj == -1:
                snippet = text[
                    obj_start:
                    min(
                        len(text),
                        obj_start + 500,
                    )
                ]

                rejected.append({
                    "item_no": -1,
                    "reason": (
                        "malformed_object:"
                        f"{exc}"
                    ),
                    "raw_item": snippet,
                })
                break

            snippet = text[
                obj_start:next_obj
            ]

            rejected.append({
                "item_no": -1,
                "reason": (
                    "malformed_object:"
                    f"{exc}"
                ),
                "raw_item": snippet,
            })

            pos = next_obj

    return items, rejected
def frame_range_to_seconds(
    start_frame,
    end_frame,
    frame_timestamps,
    window_start,
    window_end,
):
    n = len(frame_timestamps)

    if n == 0:
        raise ValueError(
            "No frame timestamps available."
        )

    start_frame = int(start_frame)
    end_frame = int(end_frame)

    if not 1 <= start_frame <= n:
        raise ValueError(
            f"start_frame {start_frame} "
            f"outside valid range 1-{n}"
        )

    if not 1 <= end_frame <= n:
        raise ValueError(
            f"end_frame {end_frame} "
            f"outside valid range 1-{n}"
        )

    if end_frame < start_frame:
        raise ValueError(
            f"end_frame {end_frame} "
            f"< start_frame {start_frame}"
        )

    start_idx = (
        start_frame - 1
    )

    end_idx = (
        end_frame - 1
    )

    start_sec = float(
        frame_timestamps[start_idx]
    )

    if end_idx < n - 1:
        end_sec = (
            float(
                frame_timestamps[end_idx]
            )
            + float(
                frame_timestamps[
                    end_idx + 1
                ]
            )
        ) / 2.0
    else:
        end_sec = float(
            window_end
        )

    start_sec = max(
        float(window_start),
        start_sec,
    )

    end_sec = min(
        float(window_end),
        end_sec,
    )

    return (
        round(start_sec, 3),
        round(end_sec, 3),
    )
def parse_output(
    raw,
    window_start,
    window_end,
    frame_timestamps,
):
    rejected = []
    rows = []

    parse_method = "frame_json"

    try:
        obj = json.loads(
            extract_json_candidate(raw)
        )

        observations = obj.get(
            "observations",
            [],
        )

        if not isinstance(
            observations,
            list,
        ):
            raise ValueError(
                "'observations' must be a list"
            )

    except Exception as json_exc:

        observations, salvage_rejected = (
            salvage_observation_objects(raw)
        )

        rejected.append({
            "item_no": -1,
            "reason": (
                "outer_json_invalid:"
                f"{json_exc}"
            ),
            "raw_item": "",
        })

        rejected.extend(
            salvage_rejected
        )

        if not observations:
            raise ValueError(
                "Could not parse or salvage "
                "any observation objects: "
                f"{json_exc}"
            ) from json_exc

        parse_method = (
            "frame_json_salvaged"
        )

    for i, item in enumerate(
        observations
    ):
        if not isinstance(item, dict):
            rejected.append({
                "item_no": i,
                "reason": (
                    "observation_not_object"
                ),
                "raw_item": repr(item),
            })
            continue

        description = str(
            item.get(
                "description",
                "",
            )
        ).strip()

        if not description:
            rejected.append({
                "item_no": i,
                "reason": (
                    "empty_description"
                ),
                "raw_item": json.dumps(
                    item,
                    ensure_ascii=False,
                ),
            })
            continue

        certainty = str(
            item.get(
                "certainty",
                "uncertain",
            )
        ).strip().lower()

        if certainty not in {
            "clear",
            "uncertain",
        }:
            certainty = "uncertain"

        try:
            start_frame = int(
                item["start_frame"]
            )

            end_frame = int(
                item["end_frame"]
            )

            start_sec, end_sec = (
                frame_range_to_seconds(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    frame_timestamps=(
                        frame_timestamps
                    ),
                    window_start=(
                        window_start
                    ),
                    window_end=(
                        window_end
                    ),
                )
            )

        except Exception as exc:
            rejected.append({
                "item_no": i,
                "reason": (
                    "invalid_frame_range:"
                    f"{exc}"
                ),
                "raw_item": json.dumps(
                    item,
                    ensure_ascii=False,
                ),
            })
            continue

        rows.append({
            "start_frame": (
                start_frame
            ),
            "end_frame": (
                end_frame
            ),
            "start_sec": (
                start_sec
            ),
            "end_sec": (
                end_sec
            ),
            "description": (
                description
            ),
            "certainty": (
                certainty
            ),
            "time_normalization": (
                "frame_index_to_timestamp"
            ),
        })

    return (
        rows,
        rejected,
        parse_method,
    )

def load_completed(details_jsonl):
    done = set()
    path = Path(details_jsonl)
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("status") == "ok":
                done.add(int(obj["segment_idx"]))
    return done


def append_jsonl(path, obj):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def rebuild_flat_csv(details_jsonl, output_csv):
    rows = []
    path = Path(details_jsonl)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("status") == "ok":
                    rows.extend(obj.get("observations", []))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["segment_idx", "start_sec", "end_sec"])
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")


class Backend:
    def __init__(self, args, temp_dir):
        self.kind = args.backend
        self.model_id = args.model_id or DEFAULT_MODELS[self.kind]
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        if self.kind == "qwen":
            self.runner = v5.QwenRunner(self.model_id)

        elif self.kind == "minicpm":
            from run_minicpm_v45_behavior_benchmark import MiniCPMRunner
            self.runner = MiniCPMRunner(
                self.model_id,
                load_in_8bit=False,
                use_flash_attn=False,
            )

        elif self.kind == "molmo":
            from run_molmo2_8b_behavior_benchmark import Molmo2Runner
            self.runner = Molmo2Runner(
                model_id=self.model_id,
                dtype=args.molmo_dtype,
            )

        elif self.kind == "internvl35":
            from run_internvl35_14b_behavior_benchmark import InternVL35Runner
            self.runner = InternVL35Runner(
                model_id=self.model_id,
                input_size=args.internvl_input_size,
                use_flash_attn=False,
            )
        else:
            raise ValueError(self.kind)

    def generate(
        self,
        frames,
        timestamps,
        prompt,
        window_start,
        window_end,
        max_new_tokens,
        temp_name,
        qwen_total_pixels,
    ):
        frames = list(frames)
        timestamps = list(timestamps)

        if len(frames) > 1 and timestamps[-1] > timestamps[0]:
            effective_fps = (len(frames) - 1) / (timestamps[-1] - timestamps[0])
        else:
            effective_fps = max(0.1, len(frames) / max(0.1, window_end - window_start))

        if self.kind == "qwen":
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,
                        "sample_fps": float(effective_fps),
                        "total_pixels": int(qwen_total_pixels),
                    },
                    {"type": "text", "text": prompt},
                ],
            }]
            return self.runner.generate(
                messages,
                max_new_tokens=int(max_new_tokens),
            )

        if self.kind == "minicpm":
            return self.runner.generate(
                prompt=prompt,
                frames=frames,
                timestamps=timestamps,
                window_start=window_start,
                max_new_tokens=int(max_new_tokens),
            )

        if self.kind == "molmo":
            from run_molmo2_8b_behavior_benchmark import write_window_video
            video_path = self.temp_dir / f"{temp_name}.mp4"
            # Critical: preserve the duration represented by the selected frames.
            write_window_video(frames, video_path, fps=float(effective_fps))
            return self.runner.generate(
                video_path=video_path,
                prompt=prompt,
                max_new_tokens=int(max_new_tokens),
            )

        if self.kind == "internvl35":
            return self.runner.generate(
                frames=frames,
                prompt=prompt,
                max_new_tokens=int(max_new_tokens),
            )

        raise ValueError(self.kind)


def run(args):
    out_dir = Path(args.output_dir) / args.backend
    out_dir.mkdir(parents=True, exist_ok=True)

    details_jsonl = out_dir / "open_description_details.jsonl"
    observations_csv = out_dir / "open_descriptions.csv"
    rejected_csv = out_dir / "rejected_rows.csv"
    temp_dir = out_dir / "_window_videos"
    preview_dir = out_dir / "track_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    role_cache_path = Path(args.role_cache)
    role_cache = v5.load_role_cache(role_cache_path)

    segments = pd.read_csv(args.segments_csv)
    required = {"segment_idx", "segment_path"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(f"segments CSV missing columns: {sorted(missing)}")

    present_forbidden = FORBIDDEN_MANIFEST_COLUMNS.intersection(segments.columns)
    if present_forbidden:
        raise ValueError(
            "LABEL FIREWALL: inference manifest contains human-label columns: "
            f"{sorted(present_forbidden)}"
        )

    segments["segment_idx"] = pd.to_numeric(
        segments["segment_idx"], errors="raise"
    ).astype(int)

    requested = parse_segment_indices(args.segment_indices)
    if requested is not None:
        segments = segments[segments["segment_idx"].isin(requested)].copy()

    done = load_completed(details_jsonl)

    face_detector = v5.get_face_detector(
        model_path=args.yunet_model,
        score_threshold=args.yunet_score_threshold,
        nms_threshold=args.yunet_nms_threshold,
        top_k=args.yunet_top_k,
    )

    qwen_total_pixels = int(args.qwen_video_token_budget * 32 * 32)

    print("")
    print("OPEN-ENDED VISUAL DESCRIPTION BENCHMARK")
    print("=" * 60)
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model_id or DEFAULT_MODELS[args.backend]}")
    print(f"Segments: {len(segments)}")
    print(f"Source sampling: {args.sample_fps} FPS")
    print(f"Window: {args.window_seconds}s")
    print(f"Common retained frames/window: {args.window_max_frames}")
    print(f"Patient crop width: {args.frame_width}")
    print("NO behavior vocabulary.")
    print("NO 3RS / rupture labels.")
    print("NO reference descriptions loaded during inference.")
    print("")

    backend = Backend(args, temp_dir)
    rejected_all = []

    for pos, row in enumerate(segments.itertuples(index=False), start=1):
        idx = int(row.segment_idx)
        if idx in done:
            print(f"[{pos}/{len(segments)}] segment {idx}: already done", flush=True)
            continue

        segment_path = Path(row.segment_path)
        print(
            f"\n[{pos}/{len(segments)}] segment {idx}: {segment_path.name}",
            flush=True,
        )
        started_segment = time.time()

        try:
            full_frames, timestamps, duration = v5.sample_full_frames(
                segment_path,
                sample_fps=args.sample_fps,
                max_duration=args.max_duration,
            )
            if not full_frames:
                raise RuntimeError("No frames sampled.")

            fw, fh = full_frames[0].size
            detections = []

            for frame_idx, (frame, ts) in enumerate(zip(full_frames, timestamps)):
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

            preview_path = preview_dir / f"segment_{idx:03d}_tracks.jpg"
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

            flat_rows = []
            window_details = []

            for w_no, window in enumerate(windows, start=1):
                frames, times = limit_frames_and_times(
                    window["frames"],
                    window["timestamps"],
                    args.window_max_frames,
                )

                print(
                    f"  window {w_no}/{len(windows)} "
                    f"{window['start']:.0f}-{window['end']:.0f}s | "
                    f"{len(frames)} frames",
                    flush=True,
                )

                prompt = open_description_prompt(
                    window_start=window["start"],
                    window_end=window["end"],
                    num_frames=len(frames),
                )

                raw, sec, peak_alloc, peak_reserved = backend.generate(
                    frames=frames,
                    timestamps=times,
                    prompt=prompt,
                    window_start=window["start"],
                    window_end=window["end"],
                    max_new_tokens=args.max_new_tokens,
                    temp_name=f"segment_{idx:03d}_window_{w_no:02d}",
                    qwen_total_pixels=qwen_total_pixels,
                )

                parsed, rejected, parse_method = parse_output(
                    raw=raw,
                    window_start=window["start"],
                    window_end=window["end"],
                    frame_timestamps=times,
                )

                for r in rejected:
                    rejected_all.append({
                        "model": args.backend,
                        "segment_idx": idx,
                        "window_start": window["start"],
                        "window_end": window["end"],
                        **r,
                    })

                for obs in parsed:
                    flat_rows.append({
                        "model": args.backend,
                        "model_id": args.model_id or DEFAULT_MODELS[args.backend],
                        "segment_idx": idx,
                        "video": getattr(row, "video", ""),
                        "patient_id": getattr(row, "patient_id", ""),
                        "session_id": getattr(row, "session_id", ""),
                        "window_start": float(window["start"]),
                        "window_end": float(window["end"]),
                        **obs,
                    })

                print(
                    f"    {len(parsed)} observations | {sec:.1f}s"
                    + (
                        f" | peak {peak_alloc:.1f} GB"
                        if peak_alloc is not None
                        else ""
                    ),
                    flush=True,
                )

                window_details.append({
                    "window_no": w_no,
                    "window_start": float(window["start"]),
                    "window_end": float(window["end"]),
                    "frame_timestamps": [float(x) for x in times],
                    "raw": raw,
                    "parse_method": parse_method,
                    "parsed": parsed,
                    "rejected": rejected,
                    "inference_sec": float(sec),
                    "peak_allocated_gb": (
                        None if peak_alloc is None else float(peak_alloc)
                    ),
                    "peak_reserved_gb": (
                        None if peak_reserved is None else float(peak_reserved)
                    ),
                })

            detail = {
                "segment_idx": idx,
                "status": "ok",
                "backend": args.backend,
                "model_id": args.model_id or DEFAULT_MODELS[args.backend],
                "segment_path": str(segment_path),
                "patient_track_id": selected["track_id"],
                "patient_selection_method": selection_method,
                "windows": window_details,
                "observations": flat_rows,
                "elapsed_sec": time.time() - started_segment,
            }
            append_jsonl(details_jsonl, detail)
            rebuild_flat_csv(details_jsonl, observations_csv)

            if rejected_all:
                pd.DataFrame(rejected_all).to_csv(
                    rejected_csv,
                    index=False,
                    encoding="utf-8-sig",
                )

            print(
                f"  complete: {len(flat_rows)} observations | "
                f"{time.time() - started_segment:.1f}s",
                flush=True,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            traceback.print_exc()
            append_jsonl(details_jsonl, {
                "segment_idx": idx,
                "status": "error",
                "backend": args.backend,
                "model_id": args.model_id or DEFAULT_MODELS[args.backend],
                "segment_path": str(segment_path),
                "error": repr(exc),
            })
            print(f"ERROR segment {idx}: {exc}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rebuild_flat_csv(details_jsonl, observations_csv)
    if rejected_all:
        pd.DataFrame(rejected_all).to_csv(
            rejected_csv,
            index=False,
            encoding="utf-8-sig",
        )

    print("")
    print("Finished.")
    print(f"Observations: {observations_csv}")
    print(f"Details: {details_jsonl}")
    print(f"Rejected: {rejected_csv}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backend",
        choices=["qwen", "minicpm", "molmo", "internvl35"],
        required=True,
    )
    p.add_argument("--model-id", default=None)
    p.add_argument("--segments-csv", required=True)
    p.add_argument(
        "--output-dir",
        default="./output/open_description_benchmark",
    )
    p.add_argument(
        "--segment-indices",
        default="4,10,16,63,65",
    )
    p.add_argument("--sample-fps", type=float, default=2.0)
    p.add_argument("--window-seconds", type=float, default=15.0)
    p.add_argument("--window-max-frames", type=int, default=16)
    p.add_argument("--max-duration", type=float, default=60.0)
    p.add_argument("--frame-width", type=int, default=320)
    p.add_argument("--max-new-tokens", type=int, default=900)
    p.add_argument(
        "--role-cache",
        default="./output/qwen3vl_visual_experiment_v5/patient_role_cache.json",
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
    p.add_argument("--patient-side", choices=["left", "right"], default=None)
    p.add_argument("--qwen-video-token-budget", type=int, default=4096)
    p.add_argument("--molmo-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--internvl-input-size", type=int, default=448)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
