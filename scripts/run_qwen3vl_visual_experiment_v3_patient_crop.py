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
The input has been cropped to enlarge the PATIENT, who is normally on the LEFT
side of the original therapy video. Therapist rupture coding is not part of this experiment.

Your task is to decide whether the PATIENT shows:

WD_P (patient withdrawal):
Visible movement away from the therapist or away from the interaction/work of therapy.

CF_P (patient confrontation):
Visible movement against the therapist or against the interaction/work of therapy.

NO_RUPTURE:
No sufficiently clear patient withdrawal or patient confrontation pattern is visible.

IMPORTANT: describe what is literally visible before making the rupture decision.

CHECK THESE PATIENT BEHAVIORS ACROSS TIME:

HEAD / FACE ORIENTATION
- head/face oriented toward the therapist side
- head/face oriented downward or away
- change from toward -> downward/away
- repeated or sustained downward/away orientation

FACE
- visible crying or tear wiping
- wiping/touching around eyes, cheeks, mouth, chin, or forehead
- lip compression / lips pressed inward
- facial tension
- marked change in facial expression

HEAD MOVEMENT
- negative head shake
- repeated head movement
- abrupt head movement away

BODY / POSTURE
- shoulder shrug
- shoulders lifting then dropping
- pulling or leaning backward
- collapsing/slumping
- becoming more closed
- turning body away
- marked stillness after previously moving

HANDS / ARMS
- face/chin touching
- wiping tears/face
- arms becoming closed/crossed
- pushing-away, dismissive, rejecting, or adversarial gestures

TEMPORAL CHANGE
- compare later behavior with earlier behavior
- note repeated patterns
- note combinations of several weaker cues

PATIENT WITHDRAWAL may be visually supported by a CLEAR PATTERN such as:
- sustained/repeated downward or away head orientation
- visible retreat, closed/collapsed posture, or pulling back
- crying/tear wiping together with other withdrawal-like behavior
- minimal visible responding together with bodily withdrawal
- several weaker visual cues that together form movement away

PATIENT CONFRONTATION may be visually supported by a CLEAR PATTERN such as:
- negative head shaking together with other rejecting/oppositional behavior
- rejecting/pushing-away gestures
- facial or mouth tension together with clear interactional opposition
- body movement against/toward the therapist in an adversarial way
- several weaker visual cues that together form movement against

Important boundaries:
- A single cue alone is usually not enough.
- Do not classify ordinary talking, ordinary hand gestures, smiling, one brief gaze
  shift, one brief downward look, or normal emotional expressiveness as rupture by itself.
- There is NO audio and NO transcript.
- Do not infer speech content or tone.
- Do not claim exact eye contact unless it is genuinely visually clear.
  Prefer literal descriptions such as "head oriented toward therapist side",
  "head oriented downward", or "face turned away".
- If evidence is ambiguous or weak, choose NO_RUPTURE.
- Judge the WHOLE MINUTE and changes over time, not one isolated frame.
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


def crop_patient_left(image, crop_fraction=0.60):
    """
    Keep the left portion of the original therapy frame.

    Default crop_fraction=0.60 keeps the left 60% of the image.
    This enlarges the patient after resizing while retaining full vertical
    information for head, shoulders, arms, and posture.
    """
    w, h = image.size

    crop_fraction = max(0.35, min(0.90, float(crop_fraction)))
    right = max(1, int(round(w * crop_fraction)))

    return image.crop((0, 0, right, h))


def resize_keep_aspect(image, max_width):
    w, h = image.size

    if w <= max_width:
        return image

    scale = max_width / float(w)

    new_w = max(32, int(round(max_width / 32.0) * 32))
    new_h = max(32, int(round((h * scale) / 32.0) * 32))

    return image.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS,
    )


def add_timestamp(image, timestamp):
    image = image.copy()
    draw = ImageDraw.Draw(image)

    label = f"{timestamp:05.1f}s"
    box = (8, 8, 92, 34)

    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((13, 12), label, fill=(255, 255, 255))

    return image


def sample_patient_crop_frames(
    video_path,
    sample_fps=1.0,
    max_duration=60.0,
    frame_width=320,
    crop_fraction=0.60,
):
    duration = min(float(max_duration), get_video_duration(video_path))

    if duration <= 0:
        raise RuntimeError(f"Invalid video duration: {duration}")

    step = 1.0 / sample_fps

    # Midpoint sampling:
    # 1 FPS -> 0.5, 1.5, ..., 59.5 seconds.
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

        # V3: patient crop FIRST, then resize.
        image = crop_patient_left(
            image,
            crop_fraction=crop_fraction,
        )
        image = resize_keep_aspect(
            image,
            max_width=frame_width,
        )
        image = add_timestamp(
            image,
            timestamp=float(ts),
        )

        frames.append(image)
        used_timestamps.append(float(ts))

    cap.release()

    if len(frames) < 2:
        raise RuntimeError(
            f"Too few frames extracted from {video_path}: {len(frames)}"
        )

    return frames, used_timestamps, duration


def save_preview_contact_sheet(frames, output_path, max_frames=12):
    """
    Optional diagnostic contact sheet so you can verify that the patient crop
    is correct before trusting the VLM result.
    """
    if not frames:
        return

    indices = np.linspace(
        0,
        len(frames) - 1,
        min(max_frames, len(frames)),
        dtype=int,
    )

    selected = [frames[i].copy() for i in indices]

    thumb_w = 220
    thumbs = []

    for img in selected:
        scale = thumb_w / float(img.width)
        thumb_h = max(1, int(round(img.height * scale)))
        thumbs.append(
            img.resize(
                (thumb_w, thumb_h),
                Image.Resampling.LANCZOS,
            )
        )

    cols = 4
    rows = math.ceil(len(thumbs) / cols)
    cell_h = max(img.height for img in thumbs)

    sheet = Image.new(
        "RGB",
        (cols * thumb_w, rows * cell_h),
        "white",
    )

    for i, img in enumerate(thumbs):
        x = (i % cols) * thumb_w
        y = (i // cols) * cell_h
        sheet.paste(img, (x, y))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def build_messages(
    frames,
    sample_fps,
    role_description,
    total_pixels,
):
    prompt = f"""
ROLE / CROP INFORMATION:
{role_description}

You are seeing ONE chronological video represented by sampled frames from the
complete 60-second segment.

The frames are CROPPED to enlarge the patient.
The timestamp printed on each frame is the time within the segment.

{PATIENT_VISUAL_DEFINITION}

OBSERVATION PROCEDURE:

First inspect the patient's visible behavior in each 15-second period:

1. 0-15 s
2. 15-30 s
3. 30-45 s
4. 45-60 s

For every period, explicitly check:
- head/face orientation
- downward/away orientation
- face/mouth actions
- head movements
- shoulders/body posture
- hands/arms
- visible emotional actions such as crying/face wiping
- change from the previous period

Then list important events with approximate timestamps.

Only AFTER describing the visible behavior, decide:

WD_P:
clear visible patient withdrawal pattern.

CF_P:
clear visible patient confrontation pattern.

MIXED_P:
both withdrawal and confrontation are clearly visible.

NO_RUPTURE:
neither is sufficiently clear visually.

IMPORTANT BASELINE RULE:
Human ratings are not available to you.
Do not infer labels from filenames, identifiers, or metadata.
Use only the visible cropped patient frames.

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
      "head_face": "literal observation",
      "face_mouth": "literal observation",
      "body_posture": "literal observation",
      "hands_arms": "literal observation",
      "change": "literal change from earlier behavior or none"
    }},
    {{
      "window": "15-30",
      "head_face": "literal observation",
      "face_mouth": "literal observation",
      "body_posture": "literal observation",
      "hands_arms": "literal observation",
      "change": "literal change from earlier behavior or none"
    }},
    {{
      "window": "30-45",
      "head_face": "literal observation",
      "face_mouth": "literal observation",
      "body_posture": "literal observation",
      "hands_arms": "literal observation",
      "change": "literal change from earlier behavior or none"
    }},
    {{
      "window": "45-60",
      "head_face": "literal observation",
      "face_mouth": "literal observation",
      "body_posture": "literal observation",
      "hands_arms": "literal observation",
      "change": "literal change from earlier behavior or none"
    }}
  ],

  "behavior_events": [
    {{
      "time_sec": 0.0,
      "behavior": "literal visible patient event"
    }}
  ],

  "evidence_for_wd_p": [
    "literal visible evidence"
  ],

  "evidence_for_cf_p": [
    "literal visible evidence"
  ],

  "counterevidence": [
    "literal visible evidence against rupture"
  ],

  "reason": "short visual-only explanation based on the observed temporal pattern"
}}

Rules:
- primary_label must be one of:
  NO_RUPTURE, WD_P, CF_P, MIXED_P.
- wd_p_present and cf_p_present must each be 0 or 1.
- NO_RUPTURE -> wd_p_present=0, cf_p_present=0
- WD_P       -> wd_p_present=1, cf_p_present=0
- CF_P       -> wd_p_present=0, cf_p_present=1
- MIXED_P    -> wd_p_present=1, cf_p_present=1
- confidence must be 0.0 to 1.0.
- Do NOT claim exact eye contact unless truly visible.
- Prefer:
  "head/face oriented toward therapist side",
  "head lowered",
  "face turned away",
  "head moved side-to-side",
  "shoulders lifted",
  "hand wiped cheek/eye area",
  etc.
- Do not describe inferred speech content or tone.
- Do not use vague words such as engaged, attentive, cooperative, defensive,
  avoidant, comfortable, or hostile in observation fields.
- If visibility is poor and behavior cannot be judged reliably, choose
  NO_RUPTURE unless there is still clearly sufficient visible evidence.
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
    allowed_labels = {
        "NO_RUPTURE",
        "WD_P",
        "CF_P",
        "MIXED_P",
    }

    primary_label = str(
        result.get("primary_label", "NO_RUPTURE")
    ).upper().strip()

    if primary_label not in allowed_labels:
        primary_label = "NO_RUPTURE"

    if primary_label == "NO_RUPTURE":
        wd_p_present = 0
        cf_p_present = 0
    elif primary_label == "WD_P":
        wd_p_present = 1
        cf_p_present = 0
    elif primary_label == "CF_P":
        wd_p_present = 0
        cf_p_present = 1
    else:
        wd_p_present = 1
        cf_p_present = 1

    try:
        confidence = float(result.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    patient_visibility = str(
        result.get("patient_visibility", "partial")
    ).lower().strip()

    if patient_visibility not in {
        "good",
        "partial",
        "poor",
    }:
        patient_visibility = "partial"

    result["primary_label"] = primary_label
    result["wd_p_present"] = wd_p_present
    result["cf_p_present"] = cf_p_present
    result["confidence"] = confidence
    result["patient_visibility"] = patient_visibility

    result.setdefault("observation_windows", [])
    result.setdefault("behavior_events", [])
    result.setdefault("evidence_for_wd_p", [])
    result.setdefault("evidence_for_cf_p", [])
    result.setdefault("counterevidence", [])
    result.setdefault("reason", "")

    return result


class PatientCropVisualClassifier:
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

        self.processor = AutoProcessor.from_pretrained(
            model_id,
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa",
        )

        self.model.eval()

        print(
            f"Model device: "
            f"{next(self.model.parameters()).device}",
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

        result = validate_result(
            extract_json(raw)
        )

        peak_allocated_gb = None
        peak_reserved_gb = None

        if torch.cuda.is_available():
            peak_allocated_gb = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 3)
            )
            peak_reserved_gb = (
                torch.cuda.max_memory_reserved()
                / (1024 ** 3)
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
            pd.to_numeric(
                ok["segment_idx"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
        )

    return rows, completed


def save_rows(rows, csv_path):
    df = pd.DataFrame(rows)

    if not df.empty and "segment_idx" in df.columns:
        df["segment_idx"] = pd.to_numeric(
            df["segment_idx"],
            errors="coerce",
        )

        df = (
            df.dropna(subset=["segment_idx"])
            .sort_values("segment_idx")
            .drop_duplicates(
                subset=["segment_idx"],
                keep="last",
            )
        )

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview_dir = output_dir / "crop_previews"

    csv_path = (
        output_dir
        / "visual_experiment_v3_patient_crop_predictions.csv"
    )

    jsonl_path = (
        output_dir
        / "visual_experiment_v3_patient_crop_details.jsonl"
    )

    segments = pd.read_csv(
        args.segments_csv,
    )

    required = {
        "segment_idx",
        "segment_path",
    }

    missing = required - set(segments.columns)

    if missing:
        raise ValueError(
            f"segments CSV missing columns: "
            f"{sorted(missing)}"
        )

    segments["segment_idx"] = pd.to_numeric(
        segments["segment_idx"],
        errors="raise",
    ).astype(int)

    requested = parse_segment_indices(
        args.segment_indices,
    )

    if requested is not None:
        segments = segments[
            segments["segment_idx"].isin(requested)
        ].copy()

    if args.max_segments is not None:
        segments = segments.head(
            args.max_segments,
        )

    rows, completed = load_previous(
        csv_path,
    )

    total_pixels = int(
        args.video_token_budget
        * 32
        * 32
    )

    classifier = PatientCropVisualClassifier(
        args.model_id,
    )

    print(
        f"Segments in run: {len(segments)}",
        flush=True,
    )
    print(
        f"Sampling: {args.sample_fps} fps",
        flush=True,
    )
    print(
        f"Patient crop: LEFT "
        f"{args.patient_crop_fraction:.2f} "
        f"of original frame",
        flush=True,
    )
    print(
        f"Cropped frame width: "
        f"<= {args.frame_width}px",
        flush=True,
    )
    print(
        f"Video total-pixel budget: "
        f"{total_pixels:,} "
        f"({args.video_token_budget} x 32 x 32)",
        flush=True,
    )
    print(
        f"CSV: {csv_path}",
        flush=True,
    )

    for pos, row in enumerate(
        segments.itertuples(index=False),
        start=1,
    ):
        segment_idx = int(
            row.segment_idx
        )

        segment_path = Path(
            row.segment_path
        )

        if segment_idx in completed:
            print(
                f"[{pos}/{len(segments)}] "
                f"segment {segment_idx}: "
                f"already done",
                flush=True,
            )
            continue

        print(
            f"\n[{pos}/{len(segments)}] "
            f"segment {segment_idx}: "
            f"{segment_path.name}",
            flush=True,
        )

        full_started = time.time()

        try:
            (
                frames,
                timestamps,
                duration,
            ) = sample_patient_crop_frames(
                video_path=segment_path,
                sample_fps=args.sample_fps,
                max_duration=args.max_duration,
                frame_width=args.frame_width,
                crop_fraction=args.patient_crop_fraction,
            )

            print(
                f"Duration: {duration:.1f}s | "
                f"patient crop | "
                f"one video object | "
                f"frames sent: {len(frames)}",
                flush=True,
            )

            if args.save_crop_previews:
                preview_path = (
                    preview_dir
                    / f"segment_{segment_idx:03d}_patient_crop.jpg"
                )

                save_preview_contact_sheet(
                    frames,
                    preview_path,
                )

                print(
                    f"Crop preview: {preview_path}",
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

            elapsed = (
                time.time()
                - full_started
            )

            output_row = {
                "segment_idx": segment_idx,
                "segment_path": str(segment_path),
                "video": getattr(
                    row,
                    "video",
                    "",
                ),
                "patient_id": getattr(
                    row,
                    "patient_id",
                    "",
                ),
                "session_id": getattr(
                    row,
                    "session_id",
                    "",
                ),
                "segment_id": getattr(
                    row,
                    "segment_id",
                    "",
                ),
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
                "primary_label": result[
                    "primary_label"
                ],
                "wd_p_present": result[
                    "wd_p_present"
                ],
                "cf_p_present": result[
                    "cf_p_present"
                ],
                "confidence": result[
                    "confidence"
                ],
                "patient_visibility": result[
                    "patient_visibility"
                ],
                "observation_windows": json.dumps(
                    result.get(
                        "observation_windows",
                        [],
                    ),
                    ensure_ascii=False,
                ),
                "behavior_events": json.dumps(
                    result.get(
                        "behavior_events",
                        [],
                    ),
                    ensure_ascii=False,
                ),
                "evidence_for_wd_p": json.dumps(
                    result.get(
                        "evidence_for_wd_p",
                        [],
                    ),
                    ensure_ascii=False,
                ),
                "evidence_for_cf_p": json.dumps(
                    result.get(
                        "evidence_for_cf_p",
                        [],
                    ),
                    ensure_ascii=False,
                ),
                "counterevidence": json.dumps(
                    result.get(
                        "counterevidence",
                        [],
                    ),
                    ensure_ascii=False,
                ),
                "reason": result.get(
                    "reason",
                    "",
                ),
                "duration_sec": round(
                    duration,
                    3,
                ),
                "frames_sent": len(frames),
                "sample_fps": args.sample_fps,
                "patient_crop_fraction": (
                    args.patient_crop_fraction
                ),
                "frame_width": (
                    args.frame_width
                ),
                "video_token_budget": (
                    args.video_token_budget
                ),
                "peak_allocated_gb": (
                    round(
                        peak_allocated_gb,
                        3,
                    )
                    if peak_allocated_gb
                    is not None
                    else None
                ),
                "peak_reserved_gb": (
                    round(
                        peak_reserved_gb,
                        3,
                    )
                    if peak_reserved_gb
                    is not None
                    else None
                ),
                "inference_sec": round(
                    inference_sec,
                    3,
                ),
                "elapsed_sec": round(
                    elapsed,
                    3,
                ),
            }

            rows = [
                old
                for old in rows
                if int(
                    float(
                        old["segment_idx"]
                    )
                )
                != segment_idx
            ]

            rows.append(
                output_row
            )

            save_rows(
                rows,
                csv_path,
            )

            detail = {
                "segment_idx": segment_idx,
                "segment_path": str(
                    segment_path
                ),
                "video": getattr(
                    row,
                    "video",
                    "",
                ),
                "patient_id": getattr(
                    row,
                    "patient_id",
                    "",
                ),
                "session_id": getattr(
                    row,
                    "session_id",
                    "",
                ),
                "segment_id": getattr(
                    row,
                    "segment_id",
                    "",
                ),
                "patient_visual_definition": (
                    PATIENT_VISUAL_DEFINITION
                ),
                "settings": {
                    "model_id": (
                        args.model_id
                    ),
                    "sample_fps": (
                        args.sample_fps
                    ),
                    "patient_crop_fraction": (
                        args.patient_crop_fraction
                    ),
                    "frame_width": (
                        args.frame_width
                    ),
                    "video_token_budget": (
                        args.video_token_budget
                    ),
                    "total_pixels": (
                        total_pixels
                    ),
                    "role_description": (
                        args.role_description
                    ),
                },
                "duration_sec": duration,
                "sampled_timestamps_sec": (
                    timestamps
                ),
                "prediction": result,
                "raw_model_output": raw,
                "peak_allocated_gb": (
                    peak_allocated_gb
                ),
                "peak_reserved_gb": (
                    peak_reserved_gb
                ),
                "inference_sec": (
                    inference_sec
                ),
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

            completed.add(
                segment_idx
            )

            print(
                f"Prediction: "
                f"{result['primary_label']} | "
                f"WD_P="
                f"{result['wd_p_present']} "
                f"CF_P="
                f"{result['cf_p_present']} | "
                f"confidence="
                f"{result['confidence']:.2f}",
                flush=True,
            )

            print(
                f"Patient visibility: "
                f"{result['patient_visibility']}",
                flush=True,
            )

            if peak_allocated_gb is not None:
                print(
                    f"Peak allocated VRAM: "
                    f"{peak_allocated_gb:.2f} GiB",
                    flush=True,
                )

                print(
                    f"Peak reserved VRAM: "
                    f"{peak_reserved_gb:.2f} GiB",
                    flush=True,
                )

            print(
                f"Inference: "
                f"{inference_sec:.1f}s | "
                f"total: "
                f"{elapsed:.1f}s",
                flush=True,
            )

            del frames

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            elapsed = (
                time.time()
                - full_started
            )

            print(
                f"ERROR segment "
                f"{segment_idx}: {exc}",
                flush=True,
            )

            traceback.print_exc()

            output_row = {
                "segment_idx": (
                    segment_idx
                ),
                "segment_path": str(
                    segment_path
                ),
                "video": getattr(
                    row,
                    "video",
                    "",
                ),
                "patient_id": getattr(
                    row,
                    "patient_id",
                    "",
                ),
                "session_id": getattr(
                    row,
                    "session_id",
                    "",
                ),
                "segment_id": getattr(
                    row,
                    "segment_id",
                    "",
                ),
                "status": "error",
                "error": repr(exc),
                "elapsed_sec": round(
                    elapsed,
                    3,
                ),
            }

            rows = [
                old
                for old in rows
                if int(
                    float(
                        old["segment_idx"]
                    )
                )
                != segment_idx
            ]

            rows.append(
                output_row
            )

            save_rows(
                rows,
                csv_path,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(
        "\nFinished",
        flush=True,
    )

    print(
        f"CSV: {csv_path}",
        flush=True,
    )

    print(
        f"JSONL: {jsonl_path}",
        flush=True,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Visual Experiment V3: "
            "patient-only Qwen3-VL rupture detection "
            "using a left-side patient crop. "
            "Focus: WD_P and CF_P."
        )
    )

    parser.add_argument(
        "--segments-csv",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "./output/"
            "qwen3vl_visual_experiment_v3_patient_crop"
        ),
    )

    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--segment-indices",
        default=None,
        help=(
            "Comma-separated segment indices, "
            "e.g. 4,10,16,63,65"
        ),
    )

    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
    )

    # V3 intentionally goes back to 1 FPS.
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help=(
            "Default 1 FPS -> about 60 frames "
            "for a 60-second segment."
        ),
    )

    # Crop itself enlarges the patient, so 320 px is
    # a safer starting point than sending 90 frames.
    parser.add_argument(
        "--frame-width",
        type=int,
        default=320,
        help=(
            "Width after patient crop. "
            "Try 384 later if memory permits."
        ),
    )

    parser.add_argument(
        "--patient-crop-fraction",
        type=float,
        default=0.60,
        help=(
            "Fraction of original image width kept "
            "from the LEFT side. Default 0.60."
        ),
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--video-token-budget",
        type=int,
        default=6144,
        help=(
            "Qwen video total-pixel budget "
            "as N*32*32."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--save-crop-previews",
        action="store_true",
        help=(
            "Save a 12-frame contact sheet for "
            "each segment so the crop can be checked."
        ),
    )

    parser.add_argument(
        "--role-description",
        default=(
            "The original therapy recording places "
            "the patient on the LEFT and therapist on "
            "the RIGHT. The input has been cropped to "
            "the LEFT portion to enlarge the patient. "
            "Focus only on patient WD_P and CF_P."
        ),
    )

    return parser


if __name__ == "__main__":
    run(
        build_parser().parse_args()
    )