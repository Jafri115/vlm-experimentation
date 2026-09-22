import argparse
import json
import math
import os
import re
import sys
import time
import traceback
import urllib.request
from pathlib import Path

# OpenCV 5 dynamic YuNet model prefers the ONNX Runtime DNN engine when available.
# If the wheel was built without ONNX Runtime, OpenCV will automatically fall back.
os.environ.setdefault("OPENCV_FORCE_DNN_ENGINE", "4")

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# ---------------------------------------------------------------------
# VISUAL EXPERIMENT V5
# ---------------------------------------------------------------------
# Goal:
#   Patient-only visual WD_P / CF_P experiment.
#
# Main change from V4:
#   DO NOT ask Qwen to summarize the entire minute in one visual call.
#
# Pipeline:
#   full video
#      -> YuNet face tracks
#      -> select patient once
#      -> tight patient ROI
#      -> sample 2 FPS
#      -> four independent 15-s windows (~30 frames each)
#      -> literal visual observer for each window (NO rupture terminology)
#      -> combine four compact observation JSONs
#      -> text-only WD_P / CF_P judge
#
# Human 3RS labels are NEVER given to the VLM.
# ---------------------------------------------------------------------


YUNET_2026_PATH = (
    "models/face_detection_yunet/face_detection_yunet_2026may.onnx"
)

YUNET_2026_MEDIA_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    + YUNET_2026_PATH
)

YUNET_2026_POINTER_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
    + YUNET_2026_PATH
)

YUNET_LFS_BATCH_URL = (
    "https://github.com/opencv/opencv_zoo.git/info/lfs/objects/batch"
)


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


WD_CF_DEFINITION = """
PATIENT WITHDRAWAL (WD_P):
Visible movement away from the therapist or away from the interaction/work of therapy.

PATIENT CONFRONTATION (CF_P):
Visible movement against the therapist or against the interaction/work of therapy.

Use ONLY the literal visual observations supplied below.

Possible visual support for WD_P includes a TEMPORAL PATTERN such as:
- sustained or repeated downward/away head or face orientation
- visible retreat, pulling back, slumping, closing posture, or turning away
- crying/face wiping together with other movement-away behavior
- marked reduction in movement together with other movement-away cues
- several weaker cues combining into a clear movement-away pattern

Possible visual support for CF_P includes a TEMPORAL PATTERN such as:
- negative head shaking together with rejecting/oppositional visible behavior
- pushing-away, dismissive, or rejecting gestures
- facial/mouth tension together with visible interactional opposition
- body movement visibly against the interaction
- several weaker cues combining into a clear movement-against pattern

IMPORTANT BOUNDARIES:
- A single weak cue is usually insufficient.
- Do not invent speech content, tone, criticism, hostility, disagreement,
  avoidance, therapeutic meaning, or motivation.
- Ordinary talking, smiling, one brief downward look, one ordinary face touch,
  or one ordinary gesture should not by itself become rupture.
- If the visual evidence is weak, isolated, or ambiguous, choose NO_RUPTURE.
""".strip()


# =====================================================================
# Generic helpers
# =====================================================================

def parse_segment_indices(value):
    if not value:
        return None
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def strip_code_fences(text):
    text = text.strip()
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
    """
    Conservative cleanup for common model JSON errors.
    We intentionally avoid aggressive transformations that could silently
    change the content.
    """
    cleaned = candidate

    # Remove trailing commas before } or ].
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    # Remove accidental comments.
    cleaned = re.sub(r"(?m)^\s*//.*$", "", cleaned)

    # Replace Python literals only when they appear as JSON values.
    cleaned = re.sub(r":\s*None(\s*[,}])", r": null\1", cleaned)
    cleaned = re.sub(r":\s*True(\s*[,}])", r": true\1", cleaned)
    cleaned = re.sub(r":\s*False(\s*[,}])", r": false\1", cleaned)

    return cleaned


# =====================================================================
# YuNet model download
# =====================================================================

def _download_bytes(url, headers=None, timeout=60):
    request_headers = {"User-Agent": "Mozilla/5.0"}
    if headers:
        request_headers.update(headers)

    request = urllib.request.Request(url, headers=request_headers)

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _parse_lfs_pointer(data):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None

    if not text.startswith("version https://git-lfs.github.com/spec/v1"):
        return None

    oid_match = re.search(r"^oid sha256:([0-9a-fA-F]{64})$", text, re.M)
    size_match = re.search(r"^size (\d+)$", text, re.M)

    if not oid_match or not size_match:
        return None

    return oid_match.group(1), int(size_match.group(1))


def _download_lfs_object(oid, size):
    payload = json.dumps(
        {
            "operation": "download",
            "transfers": ["basic"],
            "objects": [{"oid": oid, "size": int(size)}],
        }
    ).encode("utf-8")

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

    objects = batch.get("objects", [])
    if not objects:
        raise RuntimeError("Git-LFS batch response contained no objects.")

    obj = objects[0]

    if "error" in obj:
        raise RuntimeError(f"Git-LFS server error: {obj['error']}")

    action = obj.get("actions", {}).get("download", {})
    href = action.get("href")

    if not href:
        raise RuntimeError("Git-LFS response contained no download URL.")

    return _download_bytes(
        href,
        headers=action.get("header", {}) or {},
        timeout=120,
    )


def ensure_yunet_model(model_path):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists():
        data = model_path.read_bytes()

        if len(data) > 50_000 and _parse_lfs_pointer(data) is None:
            return model_path

        print(
            f"Removing invalid YuNet file: {model_path} "
            f"({len(data)} bytes)",
            flush=True,
        )
        model_path.unlink(missing_ok=True)

    print(
        f"YuNet model missing. Downloading to:\n  {model_path}",
        flush=True,
    )

    errors = []
    data = None

    try:
        candidate = _download_bytes(YUNET_2026_MEDIA_URL, timeout=120)
        pointer = _parse_lfs_pointer(candidate)

        if pointer is None and len(candidate) > 50_000:
            data = candidate
        elif pointer is not None:
            data = _download_lfs_object(*pointer)
        else:
            raise RuntimeError(
                f"unexpected media response size {len(candidate)} bytes"
            )
    except Exception as exc:
        errors.append(f"media endpoint: {exc}")

    if data is None:
        try:
            pointer_bytes = _download_bytes(
                YUNET_2026_POINTER_URL,
                timeout=60,
            )
            pointer = _parse_lfs_pointer(pointer_bytes)

            if pointer is None:
                raise RuntimeError("raw response is not a Git-LFS pointer")

            data = _download_lfs_object(*pointer)
        except Exception as exc:
            errors.append(f"Git-LFS API: {exc}")

    if data is None or len(data) < 50_000:
        raise RuntimeError(
            "Could not obtain YuNet ONNX model.\n"
            + "\n".join(f"  - {x}" for x in errors)
        )

    model_path.write_bytes(data)

    print(
        f"YuNet model ready: {model_path} "
        f"({model_path.stat().st_size:,} bytes)",
        flush=True,
    )

    return model_path


# =====================================================================
# Video / face tracking
# =====================================================================

def get_face_detector(
    model_path,
    score_threshold=0.60,
    nms_threshold=0.30,
    top_k=5000,
):
    model_path = ensure_yunet_model(model_path)

    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError(
            "cv2.FaceDetectorYN is unavailable. "
            f"OpenCV version: {getattr(cv2, '__version__', 'unknown')}"
        )

    return cv2.FaceDetectorYN.create(
        model=str(model_path),
        config="",
        input_size=(320, 320),
        score_threshold=float(score_threshold),
        nms_threshold=float(nms_threshold),
        top_k=int(top_k),
    )


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
    used_times = []

    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000.0)
        ok, frame = cap.read()

        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
        used_times.append(float(ts))

    cap.release()

    if len(frames) < 2:
        raise RuntimeError(f"Too few frames extracted from {video_path}")

    return frames, used_times, duration


def detect_faces(image, detector, min_face_px=24):
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]
    detector.setInputSize((int(w), int(h)))

    result = detector.detect(bgr)
    faces = result[1] if isinstance(result, tuple) else result

    if faces is None:
        return []

    out = []

    for face in faces:
        x, y, fw, fh = map(float, face[:4])

        if min(fw, fh) < float(min_face_px):
            continue

        score = float(face[-1]) if len(face) >= 5 else 1.0

        out.append(
            {
                "x": x,
                "y": y,
                "w": fw,
                "h": fh,
                "score": score,
            }
        )

    return out


def bbox_center(det):
    return (
        det["x"] + det["w"] / 2.0,
        det["y"] + det["h"] / 2.0,
    )


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
            centers = np.asarray(cluster["centers"], dtype=float)
            median_center = np.median(centers, axis=0)

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
            clusters.append(
                {
                    "items": [item],
                    "centers": [(cxn, cyn)],
                    "sizes": [max(det["w"], det["h"])],
                }
            )
        else:
            clusters[best_idx]["items"].append(item)
            clusters[best_idx]["centers"].append((cxn, cyn))
            clusters[best_idx]["sizes"].append(max(det["w"], det["h"]))

    tracks = []

    for cluster in clusters:
        boxes = np.array(
            [
                [
                    item["bbox"]["x"],
                    item["bbox"]["y"],
                    item["bbox"]["w"],
                    item["bbox"]["h"],
                ]
                for item in cluster["items"]
            ],
            dtype=float,
        )

        median_box = np.median(boxes, axis=0)
        cx = median_box[0] + median_box[2] / 2.0
        cy = median_box[1] + median_box[3] / 2.0

        tracks.append(
            {
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
            }
        )

    tracks.sort(
        key=lambda t: (
            -t["detection_count"],
            -t["median_face_size_px"],
        )
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

    kept = [
        t for t in tracks
        if t["detection_count"] >= minimum
    ]

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
    x = face_bbox["x"]
    y = face_bbox["y"]
    w = face_bbox["w"]
    h = face_bbox["h"]

    cx = x + w / 2.0

    roi_w = width_mult * w

    x1 = cx - roi_w / 2.0
    x2 = cx + roi_w / 2.0
    y1 = y - top_mult * h
    y2 = y + h + bottom_mult * h

    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(frame_width), x2)
    y2 = min(float(frame_height), y2)

    return {
        "x1": int(round(x1)),
        "y1": int(round(y1)),
        "x2": int(round(x2)),
        "y2": int(round(y2)),
    }


def resize_keep_aspect(image, max_width):
    w, h = image.size

    if w <= max_width:
        return image

    scale = max_width / float(w)

    new_w = max(
        32,
        int(round(max_width / 32.0) * 32),
    )
    new_h = max(
        32,
        int(round((h * scale) / 32.0) * 32),
    )

    return image.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS,
    )


def add_timestamp(image, timestamp):
    image = image.copy()
    draw = ImageDraw.Draw(image)

    draw.rectangle((8, 8, 94, 35), fill=(0, 0, 0))
    draw.text(
        (13, 12),
        f"{timestamp:05.1f}s",
        fill=(255, 255, 255),
    )

    return image


def crop_patient_frames(
    full_frames,
    timestamps,
    roi,
    frame_width,
):
    out = []

    for frame, ts in zip(full_frames, timestamps):
        crop = frame.crop(
            (
                roi["x1"],
                roi["y1"],
                roi["x2"],
                roi["y2"],
            )
        )

        crop = resize_keep_aspect(crop, frame_width)
        crop = add_timestamp(crop, ts)
        out.append(crop)

    return out


# =====================================================================
# Previews / patient role
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
        0,
        len(frames) - 1,
        min(max_frames, len(frames)),
        dtype=int,
    )

    thumbs = []

    for idx in indices:
        img = frames[idx].copy()

        scale = thumb_width / float(img.width)
        thumb_h = max(1, int(round(img.height * scale)))

        thumbs.append(
            img.resize(
                (thumb_width, thumb_h),
                Image.Resampling.LANCZOS,
            )
        )

    cell_h = max(x.height for x in thumbs)
    rows = math.ceil(len(thumbs) / cols)

    sheet = Image.new(
        "RGB",
        (cols * thumb_width, rows * cell_h),
        "white",
    )

    for i, img in enumerate(thumbs):
        x = (i % cols) * thumb_width
        y = (i // cols) * cell_h
        sheet.paste(img, (x, y))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def annotate_tracks(
    frames,
    timestamps,
    tracks,
    output_path,
    max_frames=4,
):
    """
    Small preview only. It is NOT sent to Qwen by default.
    """
    indices = np.linspace(
        0,
        len(frames) - 1,
        min(max_frames, len(frames)),
        dtype=int,
    )

    annotated = []

    for idx in indices:
        img = frames[idx].copy()
        draw = ImageDraw.Draw(img)

        for track in tracks:
            box = track["median_face_bbox"]

            x1 = int(box["x"])
            y1 = int(box["y"])
            x2 = int(box["x"] + box["w"])
            y2 = int(box["y"] + box["h"])

            draw.rectangle(
                (x1, y1, x2, y2),
                outline="white",
                width=5,
            )

            label = f"TRACK {track['track_id']}"
            draw.rectangle(
                (x1, max(0, y1 - 28), x1 + 105, y1),
                fill=(0, 0, 0),
            )
            draw.text(
                (x1 + 4, max(0, y1 - 24)),
                label,
                fill=(255, 255, 255),
            )

        draw.rectangle(
            (8, img.height - 36, 100, img.height - 8),
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


def load_role_cache(path):
    path = Path(path)

    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_role_cache(path, cache):
    Path(path).write_text(
        json.dumps(cache, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def patient_selection_key(row):
    """
    Cache role by source video/session, not by minute.
    This prevents asking patient-vs-therapist repeatedly for every segment
    from the same recording.
    """
    video = str(getattr(row, "video", "") or "")
    if video:
        return video

    patient_id = str(getattr(row, "patient_id", "") or "")
    session_id = str(getattr(row, "session_id", "") or "")

    return f"{patient_id}::{session_id}"


def select_patient_track(
    tracks,
    row,
    preview_path,
    cache,
    cache_path,
    mode="interactive",
    forced_side=None,
):
    if not tracks:
        raise RuntimeError("No persistent face candidates.")

    if len(tracks) == 1:
        return tracks[0], "single_candidate"

    cache_key = patient_selection_key(row)

    if cache_key in cache:
        side = cache[cache_key].get("patient_side")

        if side in {"left", "right"}:
            selected = track_by_side(tracks, side)
            print(
                f"Patient side from cache for {cache_key}: {side}",
                flush=True,
            )
            return selected, f"cached_{side}"

    if forced_side in {"left", "right"}:
        selected = track_by_side(tracks, forced_side)

        cache[cache_key] = {
            "patient_side": forced_side,
            "source": "forced",
        }
        save_role_cache(cache_path, cache)

        return selected, f"forced_{forced_side}"

    if mode == "leftmost":
        selected = track_by_side(tracks, "left")
        return selected, "leftmost"

    if mode == "rightmost":
        selected = track_by_side(tracks, "right")
        return selected, "rightmost"

    if mode == "largest":
        selected = max(
            tracks,
            key=lambda t: (
                t["median_face_size_px"],
                t["detection_count"],
            ),
        )
        return selected, "largest"

    # SCIENTIFIC DEFAULT: do not ask a VLM to infer patient/therapist role
    # from every minute. Ask the researcher once per video/session, then cache
    # left/right for all later segments from that same recording.
    print("", flush=True)
    print(f"Patient-role confirmation needed: {preview_path}", flush=True)

    for t in sorted(tracks, key=lambda x: x["median_center_x_norm"]):
        print(
            f"  TRACK {t['track_id']} | "
            f"x={t['median_center_x_norm']:.3f} | "
            f"side={side_of_track(t)} | "
            f"detections={t['detection_count']} | "
            f"face={t['median_face_size_px']:.1f}px",
            flush=True,
        )

    if not sys.stdin.isatty():
        raise RuntimeError(
            "Multiple people detected and no cached patient side exists. "
            "Rerun interactively or use --patient-side left/right."
        )

    valid = {t["track_id"]: t for t in tracks}

    while True:
        raw = input("Enter PATIENT TRACK ID: ").strip()

        try:
            track_id = int(raw)
        except Exception:
            print("Please enter an integer track ID.")
            continue

        if track_id not in valid:
            print(f"Valid IDs: {sorted(valid)}")
            continue

        selected = valid[track_id]
        side = side_of_track(selected)

        cache[cache_key] = {
            "patient_side": side,
            "source": "interactive",
            "selected_track_id_at_creation": track_id,
        }
        save_role_cache(cache_path, cache)

        print(
            f"Saved patient side '{side}' for {cache_key}.",
            flush=True,
        )

        return selected, "interactive"


# =====================================================================
# Qwen
# =====================================================================

class QwenRunner:
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

    def generate(
        self,
        messages,
        max_new_tokens,
    ):
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

        kwargs = {
            "text": [chat_text],
            "padding": True,
            "return_tensors": "pt",
        }

        if image_inputs is not None:
            kwargs["images"] = image_inputs

        if videos is not None:
            kwargs["videos"] = videos
            kwargs["video_metadata"] = video_metadatas

        kwargs.update(video_kwargs)
        kwargs["do_resize"] = False

        inputs = self.processor(**kwargs)
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

        elapsed = time.time() - started

        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(
                inputs.input_ids,
                generated_ids,
            )
        ]

        raw = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        peak_alloc = None
        peak_reserved = None

        if torch.cuda.is_available():
            peak_alloc = (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
            )
            peak_reserved = (
                torch.cuda.max_memory_reserved() / (1024 ** 3)
            )

        return raw, elapsed, peak_alloc, peak_reserved


def json_repair_messages(raw_text):
    prompt = f"""
Repair the following malformed JSON.

Rules:
- Return JSON only.
- Preserve the original meaning.
- Do not add new observations or facts.
- Fix only syntax/formatting required for valid JSON.

MALFORMED JSON:
{raw_text}
""".strip()

    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]


def parse_model_json(raw_text, qwen=None):
    """
    1) strict parse
    2) conservative deterministic repair
    3) optional text-only Qwen syntax repair
    """
    candidate = extract_json_candidate(raw_text)

    try:
        return json.loads(candidate), "strict"
    except Exception:
        pass

    cleaned = deterministic_json_cleanup(candidate)

    try:
        return json.loads(cleaned), "deterministic_repair"
    except Exception as deterministic_error:
        if qwen is None:
            raise deterministic_error

    repair_raw, _, _, _ = qwen.generate(
        json_repair_messages(candidate),
        max_new_tokens=900,
    )

    repaired_candidate = extract_json_candidate(repair_raw)
    repaired_candidate = deterministic_json_cleanup(repaired_candidate)

    return json.loads(repaired_candidate), "qwen_text_repair"


# =====================================================================
# V5 observer: one 15-second window at a time
# =====================================================================

def observer_messages(
    window_frames,
    window_start,
    window_end,
    sample_fps,
    total_pixels,
):
    flags_template = ",\n    ".join(
        [
            (
                f'"{key}": '
                '{"present": 0, "times_sec": [], "notes": ""}'
            )
            for key in BEHAVIOR_KEYS
        ]
    )

    prompt = f"""
You are a LITERAL VISUAL BEHAVIOR OBSERVER.

You are seeing ONLY the PATIENT from approximately
{window_start:.0f}-{window_end:.0f} seconds of a psychotherapy session.

The timestamps printed on the frames are GLOBAL timestamps within the full
60-second segment.

There is NO audio and NO transcript.

CRITICAL:
You are NOT deciding whether a rupture, withdrawal, or confrontation exists.
Do not use those concepts.

Describe only what is visibly present in THIS short window.

Inspect carefully:
1. head and face orientation
2. downward or away orientation and whether brief/repeated/sustained
3. mouth/lip actions
4. side-to-side or abrupt head movements
5. visible crying / wiping around eyes or cheeks
6. touching face/chin/forehead
7. shoulder lift, shrug, recoil, or drop
8. torso leaning/pulling backward, slumping, turning
9. arms/hands becoming closed/crossed
10. pushing-away/dismissive-looking gestures
11. noticeable reduction in movement
12. changes occurring within the window

Do NOT infer:
- exact speech content
- tone of voice
- thoughts/feelings
- motivation
- engagement/attentiveness
- hostility/cooperation
- exact eye contact unless unmistakable

Use literal language such as:
"head lowers"
"face turns to therapist side"
"face turns away"
"hand touches chin"
"hand wipes cheek/eye area"
"lips press inward"
"head moves side-to-side"
"shoulders lift then drop"
"torso leans backward"

Return COMPACT JSON only:

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
      "behavior": "literal visible event"
    }}
  ],

  "end_state": {{
    "head_face": "",
    "face_mouth": "",
    "body_shoulders": "",
    "hands_arms": ""
  }},

  "behavior_flags": {{
    {flags_template}
  }},

  "within_window_change": "literal change from beginning to end or none"
}}
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


def split_windows(
    patient_frames,
    timestamps,
    window_seconds=15.0,
    max_duration=60.0,
):
    windows = []

    start = 0.0

    while start < max_duration - 1e-6:
        end = min(start + window_seconds, max_duration)

        indices = [
            i
            for i, ts in enumerate(timestamps)
            if start <= ts < end
        ]

        frames = [patient_frames[i] for i in indices]
        times = [timestamps[i] for i in indices]

        if frames:
            windows.append(
                {
                    "start": start,
                    "end": end,
                    "frames": frames,
                    "timestamps": times,
                }
            )

        start = end

    return windows


def aggregate_behavior_flags(window_results):
    aggregate = {}

    for key in BEHAVIOR_KEYS:
        all_times = []
        notes = []

        for result in window_results:
            flags = result.get("behavior_flags", {})
            item = flags.get(key, {})

            if int(bool(item.get("present", 0))):
                for ts in item.get("times_sec", []) or []:
                    try:
                        all_times.append(float(ts))
                    except Exception:
                        pass

                note = str(item.get("notes", "") or "").strip()
                if note:
                    notes.append(note)

        aggregate[key] = {
            "present": 1 if all_times or notes else 0,
            "times_sec": sorted(set(round(x, 2) for x in all_times)),
            "notes": notes,
        }

    return aggregate


# =====================================================================
# Final text-only judge
# =====================================================================

def judge_messages(window_results, aggregate_flags):
    payload = {
        "windows": window_results,
        "aggregate_behavior_flags": aggregate_flags,
    }

    payload_text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
You are the FINAL visual-only patient 3RS judge.

You do NOT see the video.
You receive observations from FOUR INDEPENDENT short-window visual analyses.

{WD_CF_DEFINITION}

Pay special attention to:
- repetition across multiple windows
- persistence/sustained patterns
- changes from earlier to later windows
- combinations of several weaker visual cues

OBSERVATIONS:
{payload_text}

Return JSON only:

{{
  "primary_label": "NO_RUPTURE",
  "wd_p_present": 0,
  "cf_p_present": 0,

  "temporal_pattern": "short description of the cross-window visual pattern",

  "evidence_used": [
    "specific observed evidence used"
  ],

  "counterevidence": [
    "specific observed evidence against the classification"
  ],

  "reason": "short visual-only justification"
}}

Allowed primary_label:
- NO_RUPTURE
- WD_P
- CF_P
- MIXED_P

Consistency:
- NO_RUPTURE -> wd_p_present=0, cf_p_present=0
- WD_P       -> wd_p_present=1, cf_p_present=0
- CF_P       -> wd_p_present=0, cf_p_present=1
- MIXED_P    -> wd_p_present=1, cf_p_present=1

Do not invent behavior absent from the observations.
Do not output a confidence score.
""".strip()

    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]


def validate_judge(result):
    allowed = {
        "NO_RUPTURE",
        "WD_P",
        "CF_P",
        "MIXED_P",
    }

    label = str(
        result.get("primary_label", "NO_RUPTURE")
    ).upper().strip()

    if label not in allowed:
        label = "NO_RUPTURE"

    if label == "NO_RUPTURE":
        wd, cf = 0, 0
    elif label == "WD_P":
        wd, cf = 1, 0
    elif label == "CF_P":
        wd, cf = 0, 1
    else:
        wd, cf = 1, 1

    result["primary_label"] = label
    result["wd_p_present"] = wd
    result["cf_p_present"] = cf
    result.setdefault("temporal_pattern", "")
    result.setdefault("evidence_used", [])
    result.setdefault("counterevidence", [])
    result.setdefault("reason", "")

    # Deliberately remove generative confidence if model produced one anyway.
    result.pop("confidence", None)

    return result


# =====================================================================
# CSV resume helpers
# =====================================================================

def load_previous(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        return [], set()

    df = pd.read_csv(csv_path)

    if df.empty:
        return [], set()

    rows = df.to_dict("records")

    if "status" not in df.columns:
        return rows, set()

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


# =====================================================================
# Main
# =====================================================================

def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    track_preview_dir = output_dir / "track_previews"
    patient_preview_dir = output_dir / "patient_crop_previews"

    csv_path = output_dir / "visual_experiment_v5_predictions.csv"
    jsonl_path = output_dir / "visual_experiment_v5_details.jsonl"
    role_cache_path = output_dir / "patient_role_cache.json"

    role_cache = load_role_cache(role_cache_path)

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

    # Each 15-sec window has ~30 frames at 2 FPS.
    # A smaller per-window token budget is enough and protects VRAM.
    window_total_pixels = int(
        args.window_video_token_budget * 32 * 32
    )

    face_detector = get_face_detector(
        model_path=args.yunet_model,
        score_threshold=args.yunet_score_threshold,
        nms_threshold=args.yunet_nms_threshold,
        top_k=args.yunet_top_k,
    )

    print(f"Face detector: YuNet ({args.yunet_model})", flush=True)
    print(f"Segments in run: {len(segments)}", flush=True)
    print(f"Sampling: {args.sample_fps} FPS", flush=True)
    print(
        f"Expected frames/minute: ~{int(args.sample_fps * 60)}",
        flush=True,
    )
    print(
        f"Windowing: {args.window_seconds:.0f}s "
        f"(~{int(args.sample_fps * args.window_seconds)} frames/call)",
        flush=True,
    )
    print(
        f"Patient role mode: {args.patient_selection}",
        flush=True,
    )
    print(
        f"Patient crop width: <= {args.frame_width}px",
        flush=True,
    )
    print(
        f"Per-window video pixel budget: {window_total_pixels:,} "
        f"({args.window_video_token_budget} x 32 x 32)",
        flush=True,
    )

    qwen = QwenRunner(args.model_id)

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

        full_started = time.time()

        try:
            # ---------------------------------------------------------
            # 1. Sample full minute at 2 FPS.
            # ---------------------------------------------------------
            full_frames, timestamps, duration = sample_full_frames(
                segment_path,
                sample_fps=args.sample_fps,
                max_duration=args.max_duration,
            )

            fw, fh = full_frames[0].size

            # ---------------------------------------------------------
            # 2. Detect/cluster persistent faces.
            # ---------------------------------------------------------
            detections = []

            for frame_idx, (frame, ts) in enumerate(
                zip(full_frames, timestamps)
            ):
                faces = detect_faces(
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
            # 3. Track preview + patient selection once per video.
            # ---------------------------------------------------------
            track_preview_path = (
                track_preview_dir
                / f"segment_{segment_idx:03d}_tracks.jpg"
            )

            annotate_tracks(
                full_frames,
                timestamps,
                tracks,
                track_preview_path,
                max_frames=4,
            )

            selected_track, selection_method = select_patient_track(
                tracks=tracks,
                row=row,
                preview_path=track_preview_path,
                cache=role_cache,
                cache_path=role_cache_path,
                mode=args.patient_selection,
                forced_side=args.patient_side,
            )

            patient_roi = face_to_person_roi(
                selected_track["median_face_bbox"],
                frame_width=fw,
                frame_height=fh,
                width_mult=args.roi_width_face_mult,
                top_mult=args.roi_top_face_mult,
                bottom_mult=args.roi_bottom_face_mult,
            )

            print(
                f"Selected patient TRACK {selected_track['track_id']} "
                f"via {selection_method}",
                flush=True,
            )
            print(
                f"Patient ROI: "
                f"x={patient_roi['x1']}:{patient_roi['x2']} "
                f"y={patient_roi['y1']}:{patient_roi['y2']}",
                flush=True,
            )

            # ---------------------------------------------------------
            # 4. Create tight patient video.
            # ---------------------------------------------------------
            patient_frames = crop_patient_frames(
                full_frames,
                timestamps,
                patient_roi,
                frame_width=args.frame_width,
            )

            patient_preview_path = (
                patient_preview_dir
                / f"segment_{segment_idx:03d}_patient.jpg"
            )

            make_contact_sheet(
                patient_frames,
                patient_preview_path,
                cols=4,
                max_frames=12,
                thumb_width=220,
            )

            print(
                f"Track preview: {track_preview_path}",
                flush=True,
            )
            print(
                f"Patient preview: {patient_preview_path}",
                flush=True,
            )

            # ---------------------------------------------------------
            # 5. FOUR INDEPENDENT 15-second visual calls.
            # ---------------------------------------------------------
            windows = split_windows(
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

            for window_no, window in enumerate(windows, start=1):
                print(
                    f"  Window {window_no}/{len(windows)}: "
                    f"{window['start']:.0f}-{window['end']:.0f}s | "
                    f"{len(window['frames'])} frames",
                    flush=True,
                )

                raw, sec, peak_alloc, peak_reserved = qwen.generate(
                    observer_messages(
                        window_frames=window["frames"],
                        window_start=window["start"],
                        window_end=window["end"],
                        sample_fps=args.sample_fps,
                        total_pixels=window_total_pixels,
                    ),
                    max_new_tokens=args.observer_max_new_tokens,
                )

                parsed, repair_method = parse_model_json(
                    raw,
                    qwen=qwen,
                )

                total_observer_sec += sec

                if peak_alloc is not None:
                    peak_alloc_values.append(peak_alloc)

                if peak_reserved is not None:
                    peak_reserved_values.append(peak_reserved)

                window_results.append(parsed)

                window_details.append(
                    {
                        "window_start": window["start"],
                        "window_end": window["end"],
                        "timestamps": window["timestamps"],
                        "parsed": parsed,
                        "raw": raw,
                        "json_parse_method": repair_method,
                        "inference_sec": sec,
                        "peak_allocated_gb": peak_alloc,
                        "peak_reserved_gb": peak_reserved,
                    }
                )

                flags = parsed.get("behavior_flags", {})
                positives = [
                    key
                    for key, item in flags.items()
                    if isinstance(item, dict)
                    and int(bool(item.get("present", 0)))
                ]

                print(
                    f"    detected flags: "
                    f"{', '.join(positives) if positives else 'none'} "
                    f"| {sec:.1f}s "
                    f"| JSON={repair_method}",
                    flush=True,
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ---------------------------------------------------------
            # 6. Aggregate literal observations.
            # ---------------------------------------------------------
            aggregate_flags = aggregate_behavior_flags(
                window_results
            )

            positive_aggregate = [
                key
                for key, item in aggregate_flags.items()
                if item["present"]
            ]

            print(
                "Aggregate detected behavior: "
                + (
                    ", ".join(positive_aggregate)
                    if positive_aggregate
                    else "none"
                ),
                flush=True,
            )

            # ---------------------------------------------------------
            # 7. Text-only WD_P / CF_P judge.
            # ---------------------------------------------------------
            judge_raw, judge_sec, _, _ = qwen.generate(
                judge_messages(
                    window_results,
                    aggregate_flags,
                ),
                max_new_tokens=args.judge_max_new_tokens,
            )

            judge_parsed, judge_repair = parse_model_json(
                judge_raw,
                qwen=qwen,
            )

            judge_result = validate_judge(judge_parsed)

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
                "patient_track_id": selected_track["track_id"],
                "patient_selection_method": selection_method,
                "patient_side": side_of_track(selected_track),
                "face_candidate_count": len(tracks),
                "patient_roi": json.dumps(patient_roi),
                "primary_label": judge_result["primary_label"],
                "wd_p_present": judge_result["wd_p_present"],
                "cf_p_present": judge_result["cf_p_present"],
                "temporal_pattern": judge_result.get(
                    "temporal_pattern",
                    "",
                ),
                "aggregate_behavior_flags": json.dumps(
                    aggregate_flags,
                    ensure_ascii=False,
                ),
                "window_observations": json.dumps(
                    window_results,
                    ensure_ascii=False,
                ),
                "judge_evidence_used": json.dumps(
                    judge_result.get("evidence_used", []),
                    ensure_ascii=False,
                ),
                "judge_counterevidence": json.dumps(
                    judge_result.get("counterevidence", []),
                    ensure_ascii=False,
                ),
                "judge_reason": judge_result.get("reason", ""),
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
                "judge_sec": round(judge_sec, 3),
                "judge_json_parse_method": judge_repair,
                "elapsed_sec": round(elapsed, 3),
            }

            rows = [
                old
                for old in rows
                if int(float(old["segment_idx"])) != segment_idx
            ]

            rows.append(output_row)
            save_rows(rows, csv_path)

            detail = {
                "experiment": "Visual Experiment V5",
                "segment_idx": segment_idx,
                "segment_path": str(segment_path),
                "patient_tracking": {
                    "candidates": tracks,
                    "selected_track_id": selected_track["track_id"],
                    "selection_method": selection_method,
                    "patient_side": side_of_track(selected_track),
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
                },
                "window_observations": window_details,
                "aggregate_behavior_flags": aggregate_flags,
                "judge": {
                    "parsed": judge_result,
                    "raw": judge_raw,
                    "json_parse_method": judge_repair,
                    "inference_sec": judge_sec,
                },
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
                f"Prediction: {judge_result['primary_label']} | "
                f"WD_P={judge_result['wd_p_present']} "
                f"CF_P={judge_result['cf_p_present']}",
                flush=True,
            )

            print(
                f"Observer total: {total_observer_sec:.1f}s | "
                f"Judge: {judge_sec:.1f}s | "
                f"Total: {elapsed:.1f}s",
                flush=True,
            )

            if peak_allocated is not None:
                print(
                    f"Peak allocated VRAM (window calls): "
                    f"{peak_allocated:.2f} GiB",
                    flush=True,
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            del full_frames
            del patient_frames

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
                "elapsed_sec": round(elapsed, 3),
            }

            rows = [
                old
                for old in rows
                if int(float(old["segment_idx"])) != segment_idx
            ]

            rows.append(output_row)
            save_rows(rows, csv_path)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nFinished", flush=True)
    print(f"CSV: {csv_path}", flush=True)
    print(f"JSONL: {jsonl_path}", flush=True)
    print(f"Role cache: {role_cache_path}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Visual Experiment V5: YuNet patient ROI + 2 FPS + "
            "four independent 15-second literal visual observers + "
            "text-only WD_P/CF_P judge."
        )
    )

    parser.add_argument(
        "--segments-csv",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="./output/qwen3vl_visual_experiment_v5",
    )

    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--segment-indices",
        default=None,
        help="Comma-separated segment indices, e.g. 4,10,16,63,65",
    )

    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
    )

    # V5 core temporal settings.
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="V5 default: 2 FPS = ~120 frames/minute.",
    )

    parser.add_argument(
        "--window-seconds",
        type=float,
        default=15.0,
        help="V5 default: four independent 15-second windows.",
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
        help="Width of tight patient crop.",
    )

    parser.add_argument(
        "--window-video-token-budget",
        type=int,
        default=4096,
        help=(
            "Per-15-second Qwen video token/pixel budget. "
            "Lower than V4 because each call has only ~30 frames."
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

    # Patient ROI.
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

    # Patient selection.
    parser.add_argument(
        "--patient-selection",
        choices=[
            "interactive",
            "leftmost",
            "rightmost",
            "largest",
        ],
        default="interactive",
        help=(
            "Default: researcher confirms patient track once per source video; "
            "left/right role is cached for later segments from the same video."
        ),
    )

    parser.add_argument(
        "--patient-side",
        choices=["left", "right"],
        default=None,
        help=(
            "Optional forced patient side. Useful only when you already know "
            "the source recording layout."
        ),
    )

    # Generation.
    parser.add_argument(
        "--observer-max-new-tokens",
        type=int,
        default=700,
    )

    parser.add_argument(
        "--judge-max-new-tokens",
        type=int,
        default=400,
    )

    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())