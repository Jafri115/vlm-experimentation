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
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


VISUAL_DEFINITION = """
Classify the complete psychotherapy segment using VISUAL INFORMATION ONLY.

The 3RS distinguishes two rupture directions:

WITHDRAWAL:
Movement away from the other person or from the work of therapy.

CONFRONTATION:
Movement against the other person or the work of therapy.

For this visual-only experiment, use only rupture evidence that can actually
be supported by visible nonverbal behavior.

Examples explicitly supported by the 3RS manual include:

WITHDRAWAL / SHUTTING DOWN
- collapsed posture together with avoiding eye contact

CONFRONTATION / COMPLAINING OR CRITICIZING
- an expression of disgust directed toward the other person when the
  interactional target is visually clear

CONFRONTATION / PUSHING BACK
- sitting with arms crossed together with an angry facial expression

CONFRONTATION / CONTROL OR PRESSURE
- imposing or intimidating body posture directed toward the other person

Important boundaries:

- A visible action is not automatically a rupture.
- Not every smile, laugh, neutral/straight facial expression, gaze change,
  pause, posture, or ordinary gesture is movement away or against.
- Healthy or ordinary interaction should not be classified as rupture merely
  because one person is expressive, still, looking away briefly, or gesturing.
- Some 3RS markers depend on speech content, tone, or conversational context.
  Those markers cannot be established from video frames alone.
- Do not invent speech content, disagreement, criticism, avoidance, pressure,
  hostility, or therapeutic meaning.
- If the available visual information is insufficient to establish movement
  away or movement against, classify the segment as NO_RUPTURE.

RUPTURE:
At least one sufficiently clear visually supported withdrawal or confrontation
pattern is present.

NO_RUPTURE:
No sufficiently clear visually supported withdrawal or confrontation pattern
is present.

Judge the complete segment across time rather than one isolated frame.
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


def sample_full_segment_frames(
    video_path,
    sample_fps=1.0,
    max_duration=60.0,
    frame_width=384,
):
    duration = min(float(max_duration), get_video_duration(video_path))
    if duration <= 0:
        raise RuntimeError(f"Invalid video duration: {duration}")

    # 1 FPS over 60 seconds -> about 60 frames.
    n_frames = max(2, int(math.ceil(duration * sample_fps)))

    # Use the midpoint of each sampling interval.
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

{VISUAL_DEFINITION}

Make ONE binary decision for the complete segment.

Return JSON only:

{{
  "label": "RUPTURE",
  "confidence": 0.00,
  "visual_type": "withdrawal|confrontation|mixed|unclear|none",
  "actor": "patient|therapist|both|unclear|none",
  "evidence_windows": [
    {{
      "start_sec": 0.0,
      "end_sec": 10.0,
      "visual_evidence": "brief literal description of directly visible evidence"
    }}
  ],
  "counterevidence": [
    "brief literal visual observation arguing against rupture"
  ],
  "reason": "short visual-only explanation"
}}

Rules:
- label must be exactly RUPTURE or NO_RUPTURE.
- Do not infer any spoken words or tone of voice.
- Do not describe someone as speaking; say visible mouth movement.
- Do not use "engaged", "attentive", "disagreeing", "criticizing",
  "defensive", "avoidant", or similar inferred psychological/interactional terms
  unless the term is the final rupture classification itself.
- Evidence must be directly visible.
- Use evidence across the full segment, not one isolated frame.
- If evidence is ambiguous, choose NO_RUPTURE.
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
    label = str(result.get("label", "NO_RUPTURE")).upper().strip()
    if label not in {"RUPTURE", "NO_RUPTURE"}:
        label = "NO_RUPTURE"

    try:
        confidence = float(result.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    allowed_types = {
        "withdrawal",
        "confrontation",
        "mixed",
        "unclear",
        "none",
    }
    visual_type = str(result.get("visual_type", "none")).lower().strip()
    if visual_type not in allowed_types:
        visual_type = "unclear"

    allowed_actors = {
        "patient",
        "therapist",
        "both",
        "unclear",
        "none",
    }
    actor = str(result.get("actor", "none")).lower().strip()
    if actor not in allowed_actors:
        actor = "unclear"

    if label == "NO_RUPTURE":
        if visual_type not in {"none", "unclear"}:
            visual_type = "unclear"
        if actor not in {"none", "unclear"}:
            actor = "unclear"

    result["label"] = label
    result["confidence"] = confidence
    result["visual_type"] = visual_type
    result["actor"] = actor
    result.setdefault("evidence_windows", [])
    result.setdefault("counterevidence", [])
    result.setdefault("reason", "")

    return result


class DirectVisualClassifier:
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
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        output_dir
        / "direct_visual_rupture_predictions.csv"
    )
    jsonl_path = (
        output_dir
        / "direct_visual_rupture_details.jsonl"
    )

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

    requested = parse_segment_indices(args.segment_indices)
    if requested is not None:
        segments = segments[
            segments["segment_idx"].isin(requested)
        ].copy()

    if args.max_segments is not None:
        segments = segments.head(args.max_segments)

    rows, completed = load_previous(csv_path)

    total_pixels = int(
        args.video_token_budget * 32 * 32
    )

    classifier = DirectVisualClassifier(args.model_id)

    print(f"Segments in run: {len(segments)}", flush=True)
    print(f"Sampling: {args.sample_fps} fps", flush=True)
    print(f"Frame width: <= {args.frame_width}px", flush=True)
    print(
        f"Video total-pixel budget: "
        f"{total_pixels:,} "
        f"({args.video_token_budget} x 32 x 32)",
        flush=True,
    )
    print(f"CSV: {csv_path}", flush=True)

    for pos, row in enumerate(
        segments.itertuples(index=False),
        start=1,
    ):
        segment_idx = int(row.segment_idx)
        segment_path = Path(row.segment_path)

        if segment_idx in completed:
            print(
                f"[{pos}/{len(segments)}] "
                f"segment {segment_idx}: already done",
                flush=True,
            )
            continue

        print(
            f"\n[{pos}/{len(segments)}] "
            f"segment {segment_idx}: {segment_path.name}",
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
                f"Duration: {duration:.1f}s | "
                f"one video object | "
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
                "status": "ok",
                "error": "",
                "label": result["label"],
                "confidence": result["confidence"],
                "visual_type": result["visual_type"],
                "actor": result["actor"],
                "evidence_windows": json.dumps(
                    result.get("evidence_windows", []),
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
                "visual_definition": VISUAL_DEFINITION,
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
                f"Prediction: {result['label']} "
                f"| confidence={result['confidence']:.2f}",
                flush=True,
            )
            print(
                f"Type: {result['visual_type']} "
                f"| actor: {result['actor']}",
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
                f"Inference: {inference_sec:.1f}s | "
                f"total: {elapsed:.1f}s",
                flush=True,
            )

            del frames

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
            "Direct visual-only binary psychotherapy rupture baseline "
            "with Qwen3-VL. One complete pre-sampled video object, "
            "no RAG, no transcript, no audio."
        )
    )

    parser.add_argument(
        "--segments-csv",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="./output/qwen3vl_direct_visual_v3",
    )

    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--segment-indices",
        default=None,
        help="Comma-separated zero-based segment indices, e.g. 28 or 1,5,28",
    )

    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="Default 1 FPS -> about 60 frames for a 60-second segment.",
    )

    parser.add_argument(
        "--frame-width",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--video-token-budget",
        type=int,
        default=8192,
        help=(
            "Qwen3-VL video total-pixel budget as N*32*32. "
            "Default N=8192 is intentionally conservative."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--role-description",
        default=(
            "Patient is the person on the LEFT side of the video. "
            "Therapist is the person on the RIGHT side of the video."
        ),
    )

    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())