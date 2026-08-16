import argparse
import json
import math
import re
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


PATIENT_VISUAL_DEFINITION = """
Classify the complete psychotherapy segment using VISUAL INFORMATION ONLY.

FOCUS ONLY ON THE PATIENT.
Ignore therapist rupture coding. The therapist may be partly visible or not visible.
Patient is the person on the LEFT side of the video.

Your task is to decide whether the PATIENT shows:

WD_P (patient withdrawal):
Movement away from the therapist or away from the work of therapy.

CF_P (patient confrontation):
Movement against the therapist or against the work of therapy.

NO_RUPTURE:
No sufficiently clear patient withdrawal or patient confrontation pattern is visible.

For this visual-only experiment, use only evidence that can actually be seen.

Possible visual signs of PATIENT WITHDRAWAL when clearly visible:
- reduced or avoided eye contact, especially if sustained or repeated
- looking away/down while the interaction continues
- collapsed, closed, or withdrawn posture
- crying, shutting down, or visibly retreating into self
- minimal visible response together with clear bodily withdrawal
- repeated nonverbal signs that together suggest movement away

Possible visual signs of PATIENT CONFRONTATION when clearly visible:
- negative head movement directed in interaction
- angry, tense, rejecting, or oppositional facial/bodily behavior
- pushing-away, dismissive, rejecting, or adversarial hand/arm gestures
- lip compression, mouth tension, or rigid body tension if it clearly occurs
  in an interactional pattern together with other cues
- repeated nonverbal signs that together suggest movement against

Important boundaries:
- A single cue alone is often not enough.
- Do not classify normal talking, ordinary hand gestures, smiling, brief gaze shifts,
  brief looking down, posture adjustments, or normal emotional expressiveness
  as rupture by themselves.
- There is NO audio and NO transcript.
- Do not infer speech content, tone, criticism, avoidance, or hostility unless the
  visible patient behavior clearly supports that pattern.
- If evidence is ambiguous, subtle, or weak, choose NO_RUPTURE.

Judge the whole segment across time, not one isolated frame.
""".strip()


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end < 0:
        raise ValueError(f"No JSON object found in model output:\n{text}")

    return json.loads(text[start:end + 1])


def parse_segment_indices(value):
    if not value:
        return None
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def get_video_duration(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not fps or fps <= 0:
        fps = 25.0

    if frame_count <= 0:
        return 60.0

    return frame_count / fps


def resize_keep_aspect(image, max_width):
    w, h = image.size
    if w <= max_width:
        return image

    scale = max_width / float(w)
    new_h = max(32, int(round((h * scale) / 32.0) * 32))
    new_w = max(32, int(round(max_width / 32.0) * 32))

    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def add_timestamp(image, timestamp):
    image = image.copy()
    draw = ImageDraw.Draw(image)

    label = f"{timestamp:05.1f}s"
    box = (8, 8, 92, 34)

    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((13, 12), label, fill=(255, 255, 255))

    return image


def sample_full_segment_frames(
    video_path,
    sample_fps=1.5,
    max_duration=60.0,
    frame_width=320,
):
    duration = min(float(max_duration), get_video_duration(video_path))
    if duration <= 0:
        raise RuntimeError(f"Invalid video duration: {duration}")

    step = 1.0 / sample_fps
    timestamps = np.arange(
        step / 2.0,
        duration,
        step,
        dtype=np.float32,
    )

    if len(timestamps) == 0:
        timestamps = np.array([0.0], dtype=np.float32)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    used_timestamps = []

    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image = resize_keep_aspect(image, frame_width)
        image = add_timestamp(image, float(ts))

        frames.append(image)
        used_timestamps.append(float(ts))

    cap.release()

    if len(frames) < 2:
        raise RuntimeError(
            f"Too few frames extracted from {video_path}: {len(frames)}"
        )

    return frames, used_timestamps, duration


def build_messages(
    frames,
    sample_fps,
    role_description,
    total_pixels,
):
    prompt = f"""
ROLE LAYOUT:
{role_description}

You are seeing ONE chronological video represented by sampled frames from the
complete segment. The timestamp printed on each frame is the time within the segment.

{PATIENT_VISUAL_DEFINITION}

Follow this reasoning procedure internally:
1. Examine the patient across the whole minute.
2. Note visible patient behavior in these four periods:
   - 0-15 s
   - 15-30 s
   - 30-45 s
   - 45-60 s
3. Decide whether the overall patient pattern shows:
   - movement away -> WD_P
   - movement against -> CF_P
   - both -> MIXED_P
   - neither clearly -> NO_RUPTURE

IMPORTANT BASELINE RULE:
Human ratings are not available to you and must not be inferred from filenames,
segment identifiers, or metadata. Base the decision only on the visible frames
and the visual definition above.

Return JSON only:

{{
  "primary_label": "NO_RUPTURE",
  "wd_p_present": 0,
  "cf_p_present": 0,
  "confidence": 0.00,
  "patient_visibility": "good|partial|poor",
  "observation_windows": [
    {{
      "window": "0-15",
      "patient_behavior": "literal visible description"
    }},
    {{
      "window": "15-30",
      "patient_behavior": "literal visible description"
    }},
    {{
      "window": "30-45",
      "patient_behavior": "literal visible description"
    }},
    {{
      "window": "45-60",
      "patient_behavior": "literal visible description"
    }}
  ],
  "evidence_for_wd_p": [
    "literal visible evidence for patient withdrawal"
  ],
  "evidence_for_cf_p": [
    "literal visible evidence for patient confrontation"
  ],
  "counterevidence": [
    "literal visible evidence against rupture"
  ],
  "reason": "short visual-only explanation"
}}

Rules:
- primary_label must be one of: NO_RUPTURE, WD_P, CF_P, MIXED_P.
- wd_p_present and cf_p_present must be 0 or 1.
- If primary_label is NO_RUPTURE, then wd_p_present = 0 and cf_p_present = 0.
- If primary_label is WD_P, then wd_p_present = 1 and cf_p_present = 0.
- If primary_label is CF_P, then wd_p_present = 0 and cf_p_present = 1.
- If primary_label is MIXED_P, then wd_p_present = 1 and cf_p_present = 1.
- confidence is from 0.0 to 1.0.
- Use literal visible descriptions in observation_windows and evidence fields.
- Avoid interpretive words like engaged, attentive, cooperative, defensive, avoidant,
  unless they are directly justified in the final classification.
- Do not infer speech content or tone of voice.
- If the patient is not clearly visible enough, mention this in patient_visibility
  and use NO_RUPTURE unless the visible evidence is still clearly sufficient.
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


def validate_result(result):
    allowed_labels = {"NO_RUPTURE", "WD_P", "CF_P", "MIXED_P"}
    primary_label = str(result.get("primary_label", "NO_RUPTURE")).upper().strip()
    if primary_label not in allowed_labels:
        primary_label = "NO_RUPTURE"

    try:
        wd_p_present = int(result.get("wd_p_present", 0))
    except Exception:
        wd_p_present = 0

    try:
        cf_p_present = int(result.get("cf_p_present", 0))
    except Exception:
        cf_p_present = 0

    wd_p_present = 1 if wd_p_present else 0
    cf_p_present = 1 if cf_p_present else 0

    if primary_label == "NO_RUPTURE":
        wd_p_present = 0
        cf_p_present = 0
    elif primary_label == "WD_P":
        wd_p_present = 1
        cf_p_present = 0
    elif primary_label == "CF_P":
        wd_p_present = 0
        cf_p_present = 1
    elif primary_label == "MIXED_P":
        wd_p_present = 1
        cf_p_present = 1

    try:
        confidence = float(result.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    patient_visibility = str(result.get("patient_visibility", "good")).lower().strip()
    if patient_visibility not in {"good", "partial", "poor"}:
        patient_visibility = "partial"

    result["primary_label"] = primary_label
    result["wd_p_present"] = wd_p_present
    result["cf_p_present"] = cf_p_present
    result["confidence"] = confidence
    result["patient_visibility"] = patient_visibility
    result.setdefault("observation_windows", [])
    result.setdefault("evidence_for_wd_p", [])
    result.setdefault("evidence_for_cf_p", [])
    result.setdefault("counterevidence", [])
    result.setdefault("reason", "")

    return result


class PatientVisualClassifier:
    def __init__(self, model_id):
        if torch.cuda.is_available():
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            dtype = torch.float32

        print(f"Model: {model_id}", flush=True)
        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        print(f"dtype: {dtype}", flush=True)

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.model.eval()

        print(
            f"Model device: {next(self.model.parameters()).device}",
            flush=True,
        )

    @property
    def device(self):
        return next(self.model.parameters()).device

    def predict(
        self,
        frames,
        sample_fps,
        role_description,
        total_pixels,
        max_new_tokens,
    ):
        messages = build_messages(
            frames=frames,
            sample_fps=sample_fps,
            role_description=role_description,
            total_pixels=total_pixels,
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
            for in_ids, out_ids in zip(
                inputs.input_ids,
                generated_ids,
            )
        ]

        raw = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        result = validate_result(extract_json(raw))

        peak_allocated_gb = None
        peak_reserved_gb = None

        if torch.cuda.is_available():
            peak_allocated_gb = (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
            )
            peak_reserved_gb = (
                torch.cuda.max_memory_reserved() / (1024 ** 3)
            )

        return (
            result,
            raw,
            inference_sec,
            peak_allocated_gb,
            peak_reserved_gb,
        )


def load_previous(csv_path):
    if not csv_path.exists():
        return [], set()

    df = pd.read_csv(csv_path)
    if df.empty:
        return [], set()

    rows = df.to_dict("records")
    completed = set()

    if "status" in df.columns:
        ok = df[df["status"] == "ok"]
        completed = set(
            pd.to_numeric(ok["segment_idx"], errors="coerce")
            .dropna()
            .astype(int)
        )

    return rows, completed


def save_rows(rows, csv_path):
    df = pd.DataFrame(rows)

    if not df.empty and "segment_idx" in df.columns:
        df["segment_idx"] = pd.to_numeric(df["segment_idx"], errors="coerce")
        df = (
            df.dropna(subset=["segment_idx"])
            .sort_values("segment_idx")
            .drop_duplicates(subset=["segment_idx"], keep="last")
        )

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "visual_experiment_v2_predictions.csv"
    jsonl_path = output_dir / "visual_experiment_v2_details.jsonl"

    segments = pd.read_csv(args.segments_csv)

    required = {"segment_idx", "segment_path"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(f"segments CSV missing columns: {sorted(missing)}")

    segments["segment_idx"] = pd.to_numeric(
        segments["segment_idx"],
        errors="raise",
    ).astype(int)

    requested = parse_segment_indices(args.segment_indices)
    if requested is not None:
        segments = segments[
            segments["segment_idx"].isin(requested)
        ].copy()

    if args.max_segments is not None:
        segments = segments.head(args.max_segments)

    rows, completed = load_previous(csv_path)

    total_pixels = int(args.video_token_budget * 32 * 32)

    classifier = PatientVisualClassifier(args.model_id)

    print(f"Segments in run: {len(segments)}", flush=True)
    print(f"Sampling: {args.sample_fps} fps", flush=True)
    print(f"Frame width: <= {args.frame_width}px", flush=True)
    print(
        f"Video total-pixel budget: {total_pixels:,} "
        f"({args.video_token_budget} x 32 x 32)",
        flush=True,
    )
    print(f"CSV: {csv_path}", flush=True)

    for pos, row in enumerate(segments.itertuples(index=False), start=1):
        segment_idx = int(row.segment_idx)
        segment_path = Path(row.segment_path)

        if segment_idx in completed:
            print(
                f"[{pos}/{len(segments)}] segment {segment_idx}: already done",
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
            frames, timestamps, duration = sample_full_segment_frames(
                video_path=segment_path,
                sample_fps=args.sample_fps,
                max_duration=args.max_duration,
                frame_width=args.frame_width,
            )

            print(
                f"Duration: {duration:.1f}s | one video object | "
                f"frames sent: {len(frames)}",
                flush=True,
            )

            (
                result,
                raw,
                inference_sec,
                peak_allocated_gb,
                peak_reserved_gb,
            ) = classifier.predict(
                frames=frames,
                sample_fps=args.sample_fps,
                role_description=args.role_description,
                total_pixels=total_pixels,
                max_new_tokens=args.max_new_tokens,
            )

            elapsed = time.time() - full_started

            output_row = {
                "segment_idx": segment_idx,
                "segment_path": str(segment_path),
                "video": getattr(row, "video", ""),
                "patient_id": getattr(row, "patient_id", ""),
                "session_id": getattr(row, "session_id", ""),
                "segment_id": getattr(row, "segment_id", ""),
                "segment_start_sec": getattr(row, "segment_start_sec", ""),
                "segment_duration_sec": getattr(row, "segment_duration_sec", ""),
                "status": "ok",
                "error": "",
                "primary_label": result["primary_label"],
                "wd_p_present": result["wd_p_present"],
                "cf_p_present": result["cf_p_present"],
                "confidence": result["confidence"],
                "patient_visibility": result["patient_visibility"],
                "observation_windows": json.dumps(
                    result.get("observation_windows", []),
                    ensure_ascii=False,
                ),
                "evidence_for_wd_p": json.dumps(
                    result.get("evidence_for_wd_p", []),
                    ensure_ascii=False,
                ),
                "evidence_for_cf_p": json.dumps(
                    result.get("evidence_for_cf_p", []),
                    ensure_ascii=False,
                ),
                "counterevidence": json.dumps(
                    result.get("counterevidence", []),
                    ensure_ascii=False,
                ),
                "reason": result.get("reason", ""),
                "duration_sec": round(duration, 3),
                "frames_sent": len(frames),
                "sample_fps": args.sample_fps,
                "frame_width": args.frame_width,
                "video_token_budget": args.video_token_budget,
                "peak_allocated_gb": (
                    round(peak_allocated_gb, 3)
                    if peak_allocated_gb is not None
                    else None
                ),
                "peak_reserved_gb": (
                    round(peak_reserved_gb, 3)
                    if peak_reserved_gb is not None
                    else None
                ),
                "inference_sec": round(inference_sec, 3),
                "elapsed_sec": round(elapsed, 3),
            }

            rows = [
                old for old in rows
                if int(float(old["segment_idx"])) != segment_idx
            ]
            rows.append(output_row)
            save_rows(rows, csv_path)

            detail = {
                "segment_idx": segment_idx,
                "segment_path": str(segment_path),
                "video": getattr(row, "video", ""),
                "patient_id": getattr(row, "patient_id", ""),
                "session_id": getattr(row, "session_id", ""),
                "segment_id": getattr(row, "segment_id", ""),
                "segment_start_sec": getattr(row, "segment_start_sec", ""),
                "segment_duration_sec": getattr(row, "segment_duration_sec", ""),
                "patient_visual_definition": PATIENT_VISUAL_DEFINITION,
                "settings": {
                    "model_id": args.model_id,
                    "sample_fps": args.sample_fps,
                    "frame_width": args.frame_width,
                    "video_token_budget": args.video_token_budget,
                    "total_pixels": total_pixels,
                    "role_description": args.role_description,
                },
                "duration_sec": duration,
                "sampled_timestamps_sec": timestamps,
                "prediction": result,
                "raw_model_output": raw,
                "peak_allocated_gb": peak_allocated_gb,
                "peak_reserved_gb": peak_reserved_gb,
                "inference_sec": inference_sec,
                "elapsed_sec": elapsed,
            }

            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

            completed.add(segment_idx)

            print(
                f"Prediction: {result['primary_label']} | "
                f"WD_P={result['wd_p_present']} CF_P={result['cf_p_present']} | "
                f"confidence={result['confidence']:.2f}",
                flush=True,
            )
            print(
                f"Patient visibility: {result['patient_visibility']}",
                flush=True,
            )

            if peak_allocated_gb is not None:
                print(
                    f"Peak allocated VRAM: {peak_allocated_gb:.2f} GiB",
                    flush=True,
                )
                print(
                    f"Peak reserved VRAM: {peak_reserved_gb:.2f} GiB",
                    flush=True,
                )

            print(
                f"Inference: {inference_sec:.1f}s | total: {elapsed:.1f}s",
                flush=True,
            )

            del frames

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            elapsed = time.time() - full_started

            print(f"ERROR segment {segment_idx}: {exc}", flush=True)
            traceback.print_exc()

            output_row = {
                "segment_idx": segment_idx,
                "segment_path": str(segment_path),
                "video": getattr(row, "video", ""),
                "patient_id": getattr(row, "patient_id", ""),
                "session_id": getattr(row, "session_id", ""),
                "segment_id": getattr(row, "segment_id", ""),
                "segment_start_sec": getattr(row, "segment_start_sec", ""),
                "segment_duration_sec": getattr(row, "segment_duration_sec", ""),
                "status": "error",
                "error": repr(exc),
                "elapsed_sec": round(elapsed, 3),
            }

            rows = [
                old for old in rows
                if int(float(old["segment_idx"])) != segment_idx
            ]
            rows.append(output_row)
            save_rows(rows, csv_path)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nFinished", flush=True)
    print(f"CSV: {csv_path}", flush=True)
    print(f"JSONL: {jsonl_path}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Visual Experiment V2: patient-only psychotherapy rupture detection "
            "with Qwen3-VL. Focus on WD_P and CF_P using one pre-sampled video object."
        )
    )

    parser.add_argument("--segments-csv", required=True)
    parser.add_argument(
        "--output-dir",
        default="./output/qwen3vl_visual_experiment_v2",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument(
        "--segment-indices",
        default=None,
        help="Comma-separated zero-based segment indices, e.g. 28 or 1,5,28",
    )
    parser.add_argument("--max-segments", type=int, default=None)

    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.5,
        help="Default 1.5 FPS -> about 90 frames for a 60-second segment.",
    )
    parser.add_argument("--frame-width", type=int, default=320)
    parser.add_argument("--max-duration", type=float, default=60.0)
    parser.add_argument(
        "--video-token-budget",
        type=int,
        default=6144,
        help="Qwen3-VL video total-pixel budget as N*32*32.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument(
        "--role-description",
        default=(
            "Patient is the person on the LEFT side of the video. "
            "Therapist is the person on the RIGHT side of the video, but therapist "
            "rupture coding should be ignored in this experiment."
        ),
    )

    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())