"""
Visual Experiment V6
====================
InternVL3.5-38B 4-bit PERCEPTION BENCHMARK.

Purpose
-------
V6 intentionally does NOT classify rupture.

It asks a stronger VLM to detect literal patient visual behavior on the same
five diagnostic clips used in V5, using the same patient-localisation idea
and four independent 15-second windows.

Pipeline
--------
60-second segment
  -> YuNet persistent face tracks
  -> choose patient / reuse V5 patient-side cache
  -> tight patient crop
  -> 2 FPS
  -> four independent 15-second windows (~30 frames each)
  -> InternVL3.5-38B-Instruct, NF4 4-bit
  -> literal visual observations only
  -> deterministic timestamp normalization
  -> aggregate evidence flags
  -> optional Qwen8 V5 comparison CSV

NO 3RS rupture definition is shown to InternVL.
NO WD_P / CF_P / NO_RUPTURE classification is produced.

The point of V6 is to answer:
"Does a stronger/different visual model SEE the manually visible behavior
that Qwen3-VL-8B missed?"

Official InternVL video-style inference represents sampled frames as
individual <image> items, with num_patches_list. To control memory, V6 uses
one 448x448 visual patch per sampled frame.

Recommended dependencies:
    pip install -U transformers accelerate bitsandbytes torchvision safetensors
"""

import argparse
import gc
import json
import math
import os
import re
import sys
import time
import traceback
import urllib.request
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENCV_FORCE_DNN_ENGINE", "4")

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = "OpenGVLab/InternVL3_5-38B-Instruct"

BEHAVIOR_KEYS = [
    "head_down",
    "face_or_head_away",
    "negative_head_shake",
    "abrupt_head_turn_away",
    "crying_visible",
    "face_or_tear_wiping",
    "lip_compression_or_mouth_tension",
    "face_or_chin_touching",
    "shoulder_shrug_or_lift_drop",
    "lean_or_pull_backward",
    "slump_or_collapse",
    "body_turn_away",
    "arms_become_closed_or_crossed",
    "pushing_away_or_rejecting_gesture",
    "marked_reduction_in_movement",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

YUNET_RELATIVE_PATH = (
    "models/face_detection_yunet/face_detection_yunet_2026may.onnx"
)
YUNET_MEDIA_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    + YUNET_RELATIVE_PATH
)
YUNET_POINTER_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
    + YUNET_RELATIVE_PATH
)
YUNET_LFS_BATCH_URL = (
    "https://github.com/opencv/opencv_zoo.git/info/lfs/objects/batch"
)


# =====================================================================
# Generic helpers
# =====================================================================

def parse_segment_indices(value):
    if not value:
        return None
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def strip_code_fences(text):
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_candidate(text):
    text = strip_code_fences(text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in model output.")
    return text[start:end + 1]


def deterministic_json_cleanup(candidate):
    cleaned = candidate
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    cleaned = re.sub(r"(?m)^\s*//.*$", "", cleaned)
    cleaned = re.sub(r":\s*None(\s*[,}])", r": null\1", cleaned)
    cleaned = re.sub(r":\s*True(\s*[,}])", r": true\1", cleaned)
    cleaned = re.sub(r":\s*False(\s*[,}])", r": false\1", cleaned)
    return cleaned


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False)


# =====================================================================
# YuNet
# =====================================================================

def _download_bytes(url, headers=None, timeout=120):
    h = {"User-Agent": "Mozilla/5.0"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _parse_lfs_pointer(data):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None

    if not text.startswith("version https://git-lfs.github.com/spec/v1"):
        return None

    oid = re.search(r"^oid sha256:([0-9a-fA-F]{64})$", text, re.M)
    size = re.search(r"^size (\d+)$", text, re.M)
    if not oid or not size:
        return None
    return oid.group(1), int(size.group(1))


def _download_lfs_object(oid, size):
    payload = json.dumps({
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{"oid": oid, "size": int(size)}],
    }).encode("utf-8")

    request = urllib.request.Request(
        YUNET_LFS_BATCH_URL,
        data=payload,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        batch = json.loads(response.read().decode("utf-8"))

    obj = batch["objects"][0]
    if "error" in obj:
        raise RuntimeError(obj["error"])

    action = obj["actions"]["download"]
    return _download_bytes(
        action["href"],
        headers=action.get("header", {}),
        timeout=120,
    )


def ensure_yunet_model(model_path):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists():
        data = model_path.read_bytes()
        if len(data) > 50_000 and _parse_lfs_pointer(data) is None:
            return model_path
        model_path.unlink(missing_ok=True)

    print(f"Downloading YuNet to {model_path}", flush=True)

    data = None
    errors = []

    try:
        candidate = _download_bytes(YUNET_MEDIA_URL)
        pointer = _parse_lfs_pointer(candidate)
        if pointer:
            data = _download_lfs_object(*pointer)
        elif len(candidate) > 50_000:
            data = candidate
        else:
            raise RuntimeError(f"unexpected response: {len(candidate)} bytes")
    except Exception as exc:
        errors.append(f"media endpoint: {exc}")

    if data is None:
        try:
            pointer_data = _download_bytes(YUNET_POINTER_URL)
            pointer = _parse_lfs_pointer(pointer_data)
            if not pointer:
                raise RuntimeError("raw GitHub response is not an LFS pointer")
            data = _download_lfs_object(*pointer)
        except Exception as exc:
            errors.append(f"LFS endpoint: {exc}")

    if data is None or len(data) < 50_000:
        raise RuntimeError(
            "Could not obtain YuNet model:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    model_path.write_bytes(data)
    return model_path


def get_face_detector(model_path, score_threshold=0.60, nms_threshold=0.30):
    model_path = ensure_yunet_model(model_path)
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("This OpenCV build does not provide FaceDetectorYN.")

    return cv2.FaceDetectorYN.create(
        model=str(model_path),
        config="",
        input_size=(320, 320),
        score_threshold=float(score_threshold),
        nms_threshold=float(nms_threshold),
        top_k=5000,
    )


# =====================================================================
# Video / face tracking
# =====================================================================

def video_duration(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not fps or fps <= 0:
        fps = 25.0
    if count <= 0:
        return 60.0
    return count / fps


def sample_full_frames(video_path, sample_fps=2.0, max_duration=60.0):
    duration = min(float(max_duration), video_duration(video_path))
    step = 1.0 / float(sample_fps)
    timestamps = np.arange(step / 2.0, duration, step, dtype=np.float32)

    if len(timestamps) == 0:
        timestamps = np.array([0.0], dtype=np.float32)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames, used_times = [], []

    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
        used_times.append(float(ts))

    cap.release()

    if len(frames) < 2:
        raise RuntimeError(f"Too few frames extracted: {video_path}")

    return frames, used_times, duration


def detect_faces(image, detector, min_face_px=24):
    rgb = np.asarray(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    detector.setInputSize((int(w), int(h)))
    result = detector.detect(bgr)
    faces = result[1] if isinstance(result, tuple) else result

    if faces is None:
        return []

    output = []
    for face in faces:
        x, y, fw, fh = map(float, face[:4])
        if min(fw, fh) < float(min_face_px):
            continue
        output.append({
            "x": x, "y": y, "w": fw, "h": fh,
            "score": float(face[-1]),
        })
    return output


def bbox_center(det):
    return det["x"] + det["w"] / 2.0, det["y"] + det["h"] / 2.0


def cluster_static_faces(
    all_detections,
    frame_width,
    frame_height,
    center_threshold=0.14,
    size_ratio_limit=3.0,
):
    clusters = []

    for item in all_detections:
        det = item["bbox"]
        cx, cy = bbox_center(det)
        cxn = cx / frame_width
        cyn = cy / frame_height

        best_idx = None
        best_dist = None

        for idx, cluster in enumerate(clusters):
            median_center = np.median(np.asarray(cluster["centers"]), axis=0)
            dist = math.sqrt(
                (cxn - median_center[0]) ** 2
                + (cyn - median_center[1]) ** 2
            )

            current_size = max(det["w"], det["h"])
            median_size = float(np.median(cluster["sizes"]))
            ratio = max(
                current_size / max(median_size, 1.0),
                median_size / max(current_size, 1.0),
            )

            if dist <= center_threshold and ratio <= size_ratio_limit:
                if best_dist is None or dist < best_dist:
                    best_idx = idx
                    best_dist = dist

        if best_idx is None:
            clusters.append({
                "items": [item],
                "centers": [(cxn, cyn)],
                "sizes": [max(det["w"], det["h"])],
            })
        else:
            clusters[best_idx]["items"].append(item)
            clusters[best_idx]["centers"].append((cxn, cyn))
            clusters[best_idx]["sizes"].append(max(det["w"], det["h"]))

    tracks = []

    for cluster in clusters:
        boxes = np.asarray([
            [
                item["bbox"]["x"],
                item["bbox"]["y"],
                item["bbox"]["w"],
                item["bbox"]["h"],
            ]
            for item in cluster["items"]
        ], dtype=float)

        median_box = np.median(boxes, axis=0)
        cx = median_box[0] + median_box[2] / 2.0
        cy = median_box[1] + median_box[3] / 2.0

        tracks.append({
            "items": cluster["items"],
            "detection_count": len(cluster["items"]),
            "median_face_bbox": {
                "x": float(median_box[0]),
                "y": float(median_box[1]),
                "w": float(median_box[2]),
                "h": float(median_box[3]),
            },
            "median_center_x_norm": float(cx / frame_width),
            "median_center_y_norm": float(cy / frame_height),
            "median_face_size_px": float(
                np.median(np.maximum(boxes[:, 2], boxes[:, 3]))
            ),
        })

    tracks.sort(
        key=lambda t: (-t["detection_count"], -t["median_face_size_px"])
    )
    for i, track in enumerate(tracks):
        track["track_id"] = i

    return tracks


def filter_candidate_tracks(
    tracks,
    total_frames,
    min_detection_count=6,
    min_detection_fraction=0.08,
):
    minimum = max(
        int(min_detection_count),
        int(math.ceil(total_frames * min_detection_fraction)),
    )
    kept = [t for t in tracks if t["detection_count"] >= minimum]

    for i, track in enumerate(kept):
        track["track_id"] = i
    return kept


def face_to_person_roi(
    face_bbox,
    frame_width,
    frame_height,
    width_mult=4.5,
    top_mult=1.0,
    bottom_mult=5.0,
):
    x, y = face_bbox["x"], face_bbox["y"]
    w, h = face_bbox["w"], face_bbox["h"]
    cx = x + w / 2.0

    roi_w = width_mult * w
    x1 = max(0.0, cx - roi_w / 2.0)
    x2 = min(float(frame_width), cx + roi_w / 2.0)
    y1 = max(0.0, y - top_mult * h)
    y2 = min(float(frame_height), y + h + bottom_mult * h)

    return {
        "x1": int(round(x1)),
        "y1": int(round(y1)),
        "x2": int(round(x2)),
        "y2": int(round(y2)),
    }


def add_timestamp(image, timestamp):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 112, 37), fill=(0, 0, 0))
    draw.text((13, 12), f"{timestamp:05.1f}s", fill=(255, 255, 255))
    return image


def crop_patient_frames(full_frames, timestamps, roi):
    crops = []
    for frame, ts in zip(full_frames, timestamps):
        crop = frame.crop((
            roi["x1"], roi["y1"], roi["x2"], roi["y2"]
        ))
        crops.append(add_timestamp(crop, ts))
    return crops


# =====================================================================
# Previews / role cache
# =====================================================================

def make_contact_sheet(
    frames,
    output_path,
    cols=4,
    max_frames=12,
    thumb_width=220,
):
    if not frames:
        return

    indices = np.linspace(
        0, len(frames) - 1,
        min(max_frames, len(frames)),
        dtype=int,
    )

    thumbs = []
    for idx in indices:
        img = frames[idx].copy()
        scale = thumb_width / float(img.width)
        h = max(1, int(round(img.height * scale)))
        thumbs.append(
            img.resize((thumb_width, h), Image.Resampling.LANCZOS)
        )

    cell_h = max(x.height for x in thumbs)
    rows = math.ceil(len(thumbs) / cols)

    sheet = Image.new(
        "RGB",
        (cols * thumb_width, rows * cell_h),
        "white",
    )

    for i, img in enumerate(thumbs):
        sheet.paste(
            img,
            ((i % cols) * thumb_width, (i // cols) * cell_h),
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def annotate_tracks(frames, timestamps, tracks, output_path, max_frames=4):
    indices = np.linspace(
        0, len(frames) - 1,
        min(max_frames, len(frames)),
        dtype=int,
    )
    annotated = []

    for idx in indices:
        img = frames[idx].copy()
        draw = ImageDraw.Draw(img)

        for track in tracks:
            box = track["median_face_bbox"]
            x1, y1 = int(box["x"]), int(box["y"])
            x2 = int(box["x"] + box["w"])
            y2 = int(box["y"] + box["h"])

            draw.rectangle((x1, y1, x2, y2), outline="white", width=5)
            draw.rectangle(
                (x1, max(0, y1 - 28), x1 + 110, y1),
                fill=(0, 0, 0),
            )
            draw.text(
                (x1 + 4, max(0, y1 - 24)),
                f"TRACK {track['track_id']}",
                fill=(255, 255, 255),
            )

        draw.rectangle(
            (8, img.height - 36, 105, img.height - 8),
            fill=(0, 0, 0),
        )
        draw.text(
            (13, img.height - 32),
            f"{timestamps[idx]:05.1f}s",
            fill=(255, 255, 255),
        )

        annotated.append(img)

    make_contact_sheet(
        annotated,
        output_path,
        cols=2,
        max_frames=len(annotated),
        thumb_width=360,
    )


def side_of_track(track):
    return "left" if track["median_center_x_norm"] < 0.5 else "right"


def track_by_side(tracks, side):
    if side == "left":
        return min(tracks, key=lambda x: x["median_center_x_norm"])
    if side == "right":
        return max(tracks, key=lambda x: x["median_center_x_norm"])
    raise ValueError(side)


def load_json_file(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_json_file(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def patient_selection_key(row):
    video = str(getattr(row, "video", "") or "")
    if video:
        return video
    return (
        f"{getattr(row, 'patient_id', '')}::"
        f"{getattr(row, 'session_id', '')}"
    )


def select_patient_track(
    tracks,
    row,
    preview_path,
    role_cache,
    role_cache_path,
    patient_side=None,
):
    if len(tracks) == 1:
        return tracks[0], "single_candidate"

    key = patient_selection_key(row)

    if key in role_cache:
        side = role_cache[key].get("patient_side")
        if side in {"left", "right"}:
            return track_by_side(tracks, side), f"cached_{side}"

    if patient_side in {"left", "right"}:
        selected = track_by_side(tracks, patient_side)
        role_cache[key] = {
            "patient_side": patient_side,
            "source": "v6_forced",
        }
        save_json_file(role_cache_path, role_cache)
        return selected, f"forced_{patient_side}"

    print(f"\nPatient-role confirmation needed: {preview_path}", flush=True)
    for t in sorted(tracks, key=lambda x: x["median_center_x_norm"]):
        print(
            f"  TRACK {t['track_id']} | "
            f"side={side_of_track(t)} | "
            f"x={t['median_center_x_norm']:.3f} | "
            f"detections={t['detection_count']} | "
            f"face={t['median_face_size_px']:.1f}px",
            flush=True,
        )

    if not sys.stdin.isatty():
        raise RuntimeError(
            "Multiple people detected and no patient-side cache entry exists. "
            "Run once interactively or provide --patient-side."
        )

    valid = {t["track_id"]: t for t in tracks}

    while True:
        raw = input("Enter PATIENT TRACK ID: ").strip()
        try:
            track_id = int(raw)
        except Exception:
            print("Please enter an integer.")
            continue

        if track_id not in valid:
            print(f"Valid IDs: {sorted(valid)}")
            continue

        selected = valid[track_id]
        side = side_of_track(selected)

        role_cache[key] = {
            "patient_side": side,
            "source": "v6_interactive",
            "selected_track_id_at_creation": track_id,
        }
        save_json_file(role_cache_path, role_cache)

        print(f"Saved patient side '{side}' for {key}.", flush=True)
        return selected, "interactive"


# =====================================================================
# InternVL image preparation
# =====================================================================

def build_internvl_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize(
            (input_size, input_size),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def frames_to_pixel_values(frames, transform):
    # Exactly one visual patch per frame, matching InternVL's documented
    # video inference strategy with max_num=1.
    tensors = [transform(frame) for frame in frames]
    return torch.stack(tensors, dim=0), [1] * len(tensors)


def get_input_device(model):
    # InternVL custom chat consumes pixel_values through the vision model.
    for path in [
        ("vision_model",),
        ("model", "vision_model"),
        ("language_model",),
    ]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            param = next(obj.parameters())
            return param.device
        except Exception:
            continue

    return next(model.parameters()).device


# =====================================================================
# InternVL runner
# =====================================================================

class InternVLRunner:
    def __init__(
        self,
        model_id,
        cache_dir=None,
        quant_type="nf4",
        double_quant=True,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "V6 InternVL3.5-38B 4-bit requires a CUDA GPU."
            )

        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        print(f"Model: {model_id}", flush=True)
        print("Quantization: bitsandbytes 4-bit", flush=True)
        print(f"4-bit type: {quant_type}", flush=True)
        print(f"Compute dtype: {compute_dtype}", flush=True)
        print(f"CUDA GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(
            f"GPU total: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB",
            flush=True,
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(double_quant),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
            cache_dir=cache_dir,
        )

        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
            use_flash_attn=False,
            cache_dir=cache_dir,
        ).eval()

        self.compute_dtype = compute_dtype
        self.input_device = get_input_device(self.model)

        print(f"InternVL visual input device: {self.input_device}", flush=True)

        if hasattr(self.model, "get_memory_footprint"):
            try:
                footprint = self.model.get_memory_footprint() / 1024**3
                print(
                    f"Model memory footprint reported by Transformers: "
                    f"{footprint:.2f} GiB",
                    flush=True,
                )
            except Exception:
                pass

    def _chat(
        self,
        pixel_values,
        question,
        num_patches_list,
        max_new_tokens,
    ):
        generation_config = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
        }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        started = time.time()

        with torch.inference_mode():
            result = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            )

        elapsed = time.time() - started

        if isinstance(result, tuple):
            text = result[0]
        else:
            text = result

        peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

        return str(text), elapsed, peak_alloc, peak_reserved

    def observe_window(
        self,
        frames,
        timestamps,
        window_start,
        window_end,
        transform,
        max_new_tokens,
    ):
        pixel_values, num_patches_list = frames_to_pixel_values(
            frames,
            transform,
        )

        pixel_values = pixel_values.to(
            device=self.input_device,
            dtype=self.compute_dtype,
            non_blocking=True,
        )

        frame_prefix = "".join(
            f"Frame-{i+1} (global {ts:.1f}s): <image>\n"
            for i, ts in enumerate(timestamps)
        )

        prompt = observer_prompt(window_start, window_end)
        question = frame_prefix + "\n" + prompt

        try:
            return self._chat(
                pixel_values=pixel_values,
                question=question,
                num_patches_list=num_patches_list,
                max_new_tokens=max_new_tokens,
            )
        finally:
            del pixel_values
            gc.collect()
            torch.cuda.empty_cache()

    def repair_json(self, raw_text, max_new_tokens=900):
        question = f"""
Repair the following malformed JSON.

Rules:
- Return JSON only.
- Fix syntax only.
- Preserve every observation and value.
- Do not add any behavior that was not already present.

MALFORMED JSON:
{raw_text}
""".strip()

        generation_config = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
        }

        with torch.inference_mode():
            result = self.model.chat(
                self.tokenizer,
                None,
                question,
                generation_config,
                history=None,
                return_history=False,
            )

        if isinstance(result, tuple):
            return str(result[0])
        return str(result)


# =====================================================================
# Observer and normalization
# =====================================================================

def observer_prompt(window_start, window_end):
    allowed = ", ".join(BEHAVIOR_KEYS)

    return f"""
You are a LITERAL VISUAL BEHAVIOR OBSERVER.

You are seeing only the PATIENT during the {window_start:.0f}-{window_end:.0f}
second window of a psychotherapy recording. There is no audio and no
transcript.

This is a PERCEPTION benchmark.

DO NOT:
- decide whether there is a rupture
- use the terms withdrawal or confrontation
- infer thoughts, feelings, motivation, engagement, hostility, agreement,
  disagreement, attentiveness, or therapeutic meaning
- claim exact eye contact unless unmistakable

Your job is to report visible physical behavior with high recall.

Pay special attention to:
- sustained or repeated head lowering
- head/face turning away
- side-to-side head movement
- abrupt head turns
- visible crying
- wiping the eye/cheek/tear area
- lip pressing, mouth tightening, visible mouth tension
- touching cheek, chin, mouth, forehead, or face
- shoulder shrug, lift, recoil, or lift-then-drop
- torso leaning/pulling backward
- slumping/collapse
- body turning away
- arms becoming crossed/closed
- pushing-away or rejecting-looking hand/body movement
- substantial reduction in visible movement

The frame labels give GLOBAL time in the full 60-second segment.
When possible, report GLOBAL timestamps from those frame labels.
If you are uncertain about an event, include it and explicitly say
"possible" or "ambiguous" rather than silently omitting it.

Allowed behavior type strings:
{allowed}

Return COMPACT JSON only. Do not include behavior types that are absent.

{{
  "window": "{window_start:.0f}-{window_end:.0f}",
  "visibility": "good|partial|poor",
  "start_state": {{
    "head_face": "",
    "face_mouth": "",
    "body_shoulders": "",
    "hands_arms": ""
  }},
  "events": [
    {{
      "time_sec": 0.0,
      "behavior": "literal visible event",
      "certainty": "clear|possible|ambiguous"
    }}
  ],
  "detected_behaviors": [
    {{
      "type": "one allowed behavior type",
      "times_sec": [0.0],
      "notes": "literal visual description"
    }}
  ],
  "end_state": {{
    "head_face": "",
    "face_mouth": "",
    "body_shoulders": "",
    "hands_arms": ""
  }},
  "within_window_change": ""
}}

Keep events concise. Focus on observable changes, not a frame-by-frame narration.
""".strip()


def parse_model_json(raw_text, runner=None):
    candidate = extract_json_candidate(raw_text)

    try:
        return json.loads(candidate), "strict"
    except Exception:
        pass

    cleaned = deterministic_json_cleanup(candidate)
    try:
        return json.loads(cleaned), "deterministic_repair"
    except Exception as exc:
        if runner is None:
            raise exc

    repaired = runner.repair_json(candidate)
    repaired = deterministic_json_cleanup(extract_json_candidate(repaired))
    return json.loads(repaired), "internvl_text_repair"


def normalize_time(value, window_start, window_end):
    try:
        t = float(value)
    except Exception:
        return None

    duration = float(window_end - window_start)

    # Already a plausible global time.
    if window_start - 0.75 <= t <= window_end + 0.75:
        return round(max(window_start, min(t, window_end)), 2)

    # Plausible window-relative time -> convert to global.
    if 0.0 <= t <= duration + 0.75:
        return round(min(window_start + t, window_end), 2)

    return round(t, 2)


def normalize_window_result(result, window_start, window_end):
    result = dict(result)
    result["window_start_global"] = float(window_start)
    result["window_end_global"] = float(window_end)

    normalized_events = []
    for event in result.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        event = dict(event)
        t = normalize_time(
            event.get("time_sec"),
            window_start,
            window_end,
        )
        if t is not None:
            event["time_sec"] = t
        normalized_events.append(event)
    result["events"] = normalized_events

    normalized_behaviors = []
    for item in result.get("detected_behaviors", []) or []:
        if not isinstance(item, dict):
            continue

        key = str(item.get("type", "")).strip()
        if key not in BEHAVIOR_KEYS:
            continue

        times = []
        for raw_t in item.get("times_sec", []) or []:
            t = normalize_time(raw_t, window_start, window_end)
            if t is not None:
                times.append(t)

        normalized_behaviors.append({
            "type": key,
            "times_sec": sorted(set(times)),
            "notes": str(item.get("notes", "") or "").strip(),
        })

    result["detected_behaviors"] = normalized_behaviors
    return result


def aggregate_behavior_flags(window_results):
    aggregate = {
        key: {"present": 0, "times_sec": [], "notes": []}
        for key in BEHAVIOR_KEYS
    }

    for result in window_results:
        for item in result.get("detected_behaviors", []) or []:
            key = item.get("type")
            if key not in aggregate:
                continue

            aggregate[key]["present"] = 1
            aggregate[key]["times_sec"].extend(
                item.get("times_sec", []) or []
            )

            note = str(item.get("notes", "") or "").strip()
            if note:
                aggregate[key]["notes"].append(note)

    for key in aggregate:
        aggregate[key]["times_sec"] = sorted(set(
            round(float(x), 2)
            for x in aggregate[key]["times_sec"]
        ))
        aggregate[key]["notes"] = list(dict.fromkeys(
            aggregate[key]["notes"]
        ))

    return aggregate


def split_windows(
    frames,
    timestamps,
    window_seconds=15.0,
    max_duration=60.0,
):
    windows = []
    start = 0.0

    while start < max_duration - 1e-6:
        end = min(start + window_seconds, max_duration)
        indices = [
            i for i, ts in enumerate(timestamps)
            if start <= ts < end
        ]

        if indices:
            windows.append({
                "start": start,
                "end": end,
                "frames": [frames[i] for i in indices],
                "timestamps": [timestamps[i] for i in indices],
            })

        start = end

    return windows


# =====================================================================
# Resume + V5 comparison
# =====================================================================

def load_previous(csv_path):
    path = Path(csv_path)
    if not path.exists():
        return [], set()

    df = pd.read_csv(path)
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

    if not df.empty:
        df["segment_idx"] = pd.to_numeric(
            df["segment_idx"],
            errors="coerce",
        )
        df = (
            df.dropna(subset=["segment_idx"])
            .sort_values("segment_idx")
            .drop_duplicates("segment_idx", keep="last")
        )

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def safe_parse_json(value):
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    try:
        return json.loads(str(value))
    except Exception:
        return {}


def write_qwen_comparison(v6_csv, qwen_v5_csv, output_csv):
    qwen_path = Path(qwen_v5_csv)
    if not qwen_path.exists():
        print(
            f"Qwen V5 comparison skipped; file not found: {qwen_path}",
            flush=True,
        )
        return

    v6 = pd.read_csv(v6_csv)
    qwen = pd.read_csv(qwen_path)

    if "aggregate_behavior_flags" not in qwen.columns:
        print("Qwen V5 CSV has no aggregate_behavior_flags column.", flush=True)
        return

    qwen = qwen[["segment_idx", "aggregate_behavior_flags"]].copy()
    qwen = qwen.rename(columns={
        "aggregate_behavior_flags": "qwen8_aggregate_behavior_flags"
    })

    merged = v6.merge(qwen, on="segment_idx", how="left")

    comparison_rows = []

    for row in merged.itertuples(index=False):
        internvl_flags = safe_parse_json(
            getattr(row, "aggregate_behavior_flags", "{}")
        )
        qwen_flags = safe_parse_json(
            getattr(row, "qwen8_aggregate_behavior_flags", "{}")
        )

        out = {
            "segment_idx": int(row.segment_idx),
            "video": getattr(row, "video", ""),
            "internvl38_detected_count": 0,
            "qwen8_detected_count": 0,
        }

        new_detections = []
        qwen_only = []

        for key in BEHAVIOR_KEYS:
            iv = int(bool(
                internvl_flags.get(key, {}).get("present", 0)
            ))
            qw = int(bool(
                qwen_flags.get(key, {}).get("present", 0)
            ))

            out[f"internvl38_{key}"] = iv
            out[f"qwen8_{key}"] = qw
            out["internvl38_detected_count"] += iv
            out["qwen8_detected_count"] += qw

            if iv and not qw:
                new_detections.append(key)
            if qw and not iv:
                qwen_only.append(key)

        out["new_in_internvl38"] = ", ".join(new_detections)
        out["qwen8_only"] = ", ".join(qwen_only)

        comparison_rows.append(out)

    pd.DataFrame(comparison_rows).to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Qwen8 vs InternVL38 comparison: {output_csv}", flush=True)


# =====================================================================
# Main
# =====================================================================

def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    track_preview_dir = output_dir / "track_previews"
    patient_preview_dir = output_dir / "patient_crop_previews"

    csv_path = output_dir / "visual_experiment_v6_internvl38_predictions.csv"
    jsonl_path = output_dir / "visual_experiment_v6_internvl38_details.jsonl"
    comparison_path = output_dir / "v6_qwen8_vs_internvl38_comparison.csv"

    # Reuse V5 role cache by default so the same patient choice is preserved.
    role_cache_path = Path(args.role_cache)
    role_cache = load_json_file(role_cache_path)

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

    face_detector = get_face_detector(
        args.yunet_model,
        score_threshold=args.yunet_score_threshold,
        nms_threshold=args.yunet_nms_threshold,
    )

    print("Visual Experiment V6", flush=True)
    print("Purpose: perception benchmark ONLY", flush=True)
    print(f"Segments: {len(segments)}", flush=True)
    print(f"Sampling: {args.sample_fps} FPS", flush=True)
    print(
        f"Windowing: {args.window_seconds:.0f}s "
        f"(~{int(args.sample_fps * args.window_seconds)} frames/call)",
        flush=True,
    )
    print(
        f"InternVL image size: {args.internvl_image_size} "
        f"(one patch/frame)",
        flush=True,
    )
    print(f"Role cache: {role_cache_path}", flush=True)

    runner = InternVLRunner(
        model_id=args.model_id,
        cache_dir=args.hf_cache_dir,
        quant_type=args.quant_type,
        double_quant=not args.disable_double_quant,
    )

    transform = build_internvl_transform(args.internvl_image_size)

    for pos, row in enumerate(
        segments.itertuples(index=False),
        start=1,
    ):
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

        total_started = time.time()

        try:
            full_frames, timestamps, duration = sample_full_frames(
                segment_path,
                sample_fps=args.sample_fps,
                max_duration=args.max_duration,
            )

            fw, fh = full_frames[0].size

            detections = []
            for frame_idx, (frame, ts) in enumerate(
                zip(full_frames, timestamps)
            ):
                for face in detect_faces(
                    frame,
                    face_detector,
                    min_face_px=args.min_face_px,
                ):
                    detections.append({
                        "frame_idx": frame_idx,
                        "timestamp": float(ts),
                        "bbox": face,
                    })

            tracks = cluster_static_faces(
                detections,
                frame_width=fw,
                frame_height=fh,
                center_threshold=args.track_center_threshold,
            )

            tracks = filter_candidate_tracks(
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
                raise RuntimeError("No persistent face candidates detected.")

            for track in tracks:
                print(
                    f"  TRACK {track['track_id']}: "
                    f"x={track['median_center_x_norm']:.3f}, "
                    f"detections={track['detection_count']}, "
                    f"face={track['median_face_size_px']:.1f}px",
                    flush=True,
                )

            track_preview = (
                track_preview_dir
                / f"segment_{segment_idx:03d}_tracks.jpg"
            )
            annotate_tracks(
                full_frames,
                timestamps,
                tracks,
                track_preview,
            )

            selected_track, selection_method = select_patient_track(
                tracks,
                row,
                track_preview,
                role_cache,
                role_cache_path,
                patient_side=args.patient_side,
            )

            roi = face_to_person_roi(
                selected_track["median_face_bbox"],
                frame_width=fw,
                frame_height=fh,
                width_mult=args.roi_width_face_mult,
                top_mult=args.roi_top_face_mult,
                bottom_mult=args.roi_bottom_face_mult,
            )

            patient_frames = crop_patient_frames(
                full_frames,
                timestamps,
                roi,
            )

            patient_preview = (
                patient_preview_dir
                / f"segment_{segment_idx:03d}_patient.jpg"
            )
            make_contact_sheet(
                patient_frames,
                patient_preview,
                cols=4,
                max_frames=12,
                thumb_width=220,
            )

            print(
                f"Patient: TRACK {selected_track['track_id']} "
                f"via {selection_method}; side={side_of_track(selected_track)}",
                flush=True,
            )
            print(
                f"ROI x={roi['x1']}:{roi['x2']} "
                f"y={roi['y1']}:{roi['y2']}",
                flush=True,
            )

            windows = split_windows(
                patient_frames,
                timestamps,
                window_seconds=args.window_seconds,
                max_duration=min(args.max_duration, duration),
            )

            window_results = []
            window_details = []
            peak_allocs = []
            peak_reserved = []
            observer_total_sec = 0.0

            for window_no, window in enumerate(windows, start=1):
                print(
                    f"  Window {window_no}/{len(windows)} "
                    f"{window['start']:.0f}-{window['end']:.0f}s "
                    f"| {len(window['frames'])} frames",
                    flush=True,
                )

                raw, sec, peak_a, peak_r = runner.observe_window(
                    frames=window["frames"],
                    timestamps=window["timestamps"],
                    window_start=window["start"],
                    window_end=window["end"],
                    transform=transform,
                    max_new_tokens=args.observer_max_new_tokens,
                )

                parsed, parse_method = parse_model_json(
                    raw,
                    runner=runner,
                )

                parsed = normalize_window_result(
                    parsed,
                    window_start=window["start"],
                    window_end=window["end"],
                )

                observer_total_sec += sec
                peak_allocs.append(peak_a)
                peak_reserved.append(peak_r)
                window_results.append(parsed)

                positive = [
                    item["type"]
                    for item in parsed.get("detected_behaviors", [])
                    if item.get("type") in BEHAVIOR_KEYS
                ]

                print(
                    "    detected: "
                    + (", ".join(positive) if positive else "none")
                    + f" | {sec:.1f}s | JSON={parse_method}",
                    flush=True,
                )

                window_details.append({
                    "window_start": window["start"],
                    "window_end": window["end"],
                    "frame_global_times": window["timestamps"],
                    "parsed_normalized": parsed,
                    "raw": raw,
                    "json_parse_method": parse_method,
                    "inference_sec": sec,
                    "peak_allocated_gb": peak_a,
                    "peak_reserved_gb": peak_r,
                })

                gc.collect()
                torch.cuda.empty_cache()

            aggregate = aggregate_behavior_flags(window_results)
            positive_aggregate = [
                key
                for key, value in aggregate.items()
                if value["present"]
            ]

            print(
                "Aggregate literal behavior: "
                + (
                    ", ".join(positive_aggregate)
                    if positive_aggregate
                    else "none"
                ),
                flush=True,
            )

            elapsed = time.time() - total_started

            output_row = {
                "segment_idx": segment_idx,
                "segment_path": str(segment_path),
                "video": getattr(row, "video", ""),
                "patient_id": getattr(row, "patient_id", ""),
                "session_id": getattr(row, "session_id", ""),
                "segment_id": getattr(row, "segment_id", ""),
                "segment_start_sec": getattr(
                    row, "segment_start_sec", ""
                ),
                "status": "ok",
                "error": "",
                "model": args.model_id,
                "quantization": f"bnb-4bit-{args.quant_type}",
                "patient_track_id": selected_track["track_id"],
                "patient_selection_method": selection_method,
                "patient_side": side_of_track(selected_track),
                "face_candidate_count": len(tracks),
                "patient_roi": json_dumps(roi),
                "aggregate_behavior_flags": json_dumps(aggregate),
                "detected_behavior_count": len(positive_aggregate),
                "detected_behavior_types": ", ".join(positive_aggregate),
                "window_observations": json_dumps(window_results),
                "duration_sec": round(duration, 3),
                "frames_sampled": len(patient_frames),
                "sample_fps": args.sample_fps,
                "window_seconds": args.window_seconds,
                "internvl_image_size": args.internvl_image_size,
                "patches_per_frame": 1,
                "peak_allocated_gb": round(max(peak_allocs), 3),
                "peak_reserved_gb": round(max(peak_reserved), 3),
                "observer_total_sec": round(observer_total_sec, 3),
                "elapsed_sec": round(elapsed, 3),
            }

            # Flat behavior columns make quick inspection/evaluation easier.
            for key in BEHAVIOR_KEYS:
                output_row[f"{key}_present"] = aggregate[key]["present"]
                output_row[f"{key}_times"] = json_dumps(
                    aggregate[key]["times_sec"]
                )

            rows = [
                old for old in rows
                if int(float(old["segment_idx"])) != segment_idx
            ]
            rows.append(output_row)
            save_rows(rows, csv_path)

            detail = {
                "experiment": "V6 InternVL3.5-38B 4-bit perception benchmark",
                "purpose": "literal visual perception only; no 3RS judgment",
                "segment_idx": segment_idx,
                "segment_path": str(segment_path),
                "patient_tracking": {
                    "selected_track_id": selected_track["track_id"],
                    "selection_method": selection_method,
                    "patient_side": side_of_track(selected_track),
                    "roi": roi,
                    "track_preview": str(track_preview),
                    "patient_preview": str(patient_preview),
                },
                "settings": {
                    "model": args.model_id,
                    "quantization": f"bnb-4bit-{args.quant_type}",
                    "sample_fps": args.sample_fps,
                    "window_seconds": args.window_seconds,
                    "internvl_image_size": args.internvl_image_size,
                    "patches_per_frame": 1,
                },
                "window_observations": window_details,
                "aggregate_behavior_flags": aggregate,
                "observer_total_sec": observer_total_sec,
                "elapsed_sec": elapsed,
            }

            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

            completed.add(segment_idx)

            print(
                f"Perception complete | behaviors={len(positive_aggregate)} "
                f"| observer={observer_total_sec:.1f}s "
                f"| total={elapsed:.1f}s "
                f"| peak={max(peak_allocs):.2f} GiB",
                flush=True,
            )

            del full_frames
            del patient_frames
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as exc:
            elapsed = time.time() - total_started
            print(f"ERROR segment {segment_idx}: {exc}", flush=True)
            traceback.print_exc()

            output_row = {
                "segment_idx": segment_idx,
                "segment_path": str(segment_path),
                "video": getattr(row, "video", ""),
                "patient_id": getattr(row, "patient_id", ""),
                "session_id": getattr(row, "session_id", ""),
                "status": "error",
                "error": repr(exc),
                "model": args.model_id,
                "quantization": f"bnb-4bit-{args.quant_type}",
                "elapsed_sec": round(elapsed, 3),
            }

            rows = [
                old for old in rows
                if int(float(old["segment_idx"])) != segment_idx
            ]
            rows.append(output_row)
            save_rows(rows, csv_path)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nFinished V6", flush=True)
    print(f"CSV: {csv_path}", flush=True)
    print(f"JSONL: {jsonl_path}", flush=True)

    if csv_path.exists() and args.qwen_v5_csv:
        write_qwen_comparison(
            v6_csv=csv_path,
            qwen_v5_csv=args.qwen_v5_csv,
            output_csv=comparison_path,
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "V6: InternVL3.5-38B NF4 4-bit literal visual perception "
            "benchmark. No rupture classification."
        )
    )

    parser.add_argument(
        "--segments-csv",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        default="./output/internvl35_38b_visual_experiment_v6",
    )
    parser.add_argument(
        "--segment-indices",
        default="4,10,16,63,65",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help=(
            "Optional Hugging Face cache directory. The official 38B "
            "checkpoint is large, so use a drive with sufficient free space."
        ),
    )
    parser.add_argument(
        "--quant-type",
        choices=["nf4", "fp4"],
        default="nf4",
    )
    parser.add_argument(
        "--disable-double-quant",
        action="store_true",
    )

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
        "--internvl-image-size",
        type=int,
        default=448,
    )
    parser.add_argument(
        "--observer-max-new-tokens",
        type=int,
        default=900,
    )

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
        "--role-cache",
        default=(
            "./output/qwen3vl_visual_experiment_v5/"
            "patient_role_cache.json"
        ),
        help="Defaults to the patient-role cache created by V5.",
    )
    parser.add_argument(
        "--patient-side",
        choices=["left", "right"],
        default=None,
    )

    parser.add_argument(
        "--qwen-v5-csv",
        default=(
            "./output/qwen3vl_visual_experiment_v5/"
            "visual_experiment_v5_predictions.csv"
        ),
        help=(
            "If present, V6 writes a behavior-level Qwen8 vs InternVL38 "
            "comparison CSV after the run."
        ),
    )

    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())