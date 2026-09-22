#!/usr/bin/env python
"""Standalone LLaVA-Video runner for Open Description Benchmark v1.
Does NOT import Qwen3-VL or run_qwen3vl_visual_experiment_v5.py.
"""
from __future__ import annotations

import argparse, copy, json, re, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ---------------------------------------------------------------------
# Standalone copies of the stable V5 preprocessing utilities.
#
# These are copied from run_qwen3vl_visual_experiment_py so this
# LLaVA runner does NOT import Qwen3-VL classes or require a newer
# Transformers build. The crop/tracking behavior is intentionally kept
# the same for the frozen benchmark.
# ---------------------------------------------------------------------

import math
import os
import urllib.request

import cv2
from PIL import Image, ImageDraw

os.environ.setdefault("OPENCV_FORCE_DNN_ENGINE", "4")

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


def parse_segment_indices(value):
    if not value:
        return None
    return {int(x.strip()) for x in value.split(",") if x.strip()}


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
        return track_by_side(tracks, "left"), "leftmost"

    if mode == "rightmost":
        return track_by_side(tracks, "right"), "rightmost"

    if mode == "largest":
        selected = max(
            tracks,
            key=lambda t: (
                t["median_face_size_px"],
                t["detection_count"],
            ),
        )
        return selected, "largest"

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

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model

MODEL_ID = "lmms-lab/LLaVA-Video-7B-Qwen2"
BACKEND = "llavavideo"
CSV_COLUMNS = [
    "model","model_id","segment_idx","video","patient_id","session_id",
    "window_start","window_end","start_frame","end_frame","start_sec","end_sec",
    "description","certainty","time_normalization",
]


def limit_frames_and_times(frames, timestamps, max_frames):
    frames, timestamps = list(frames), list(timestamps)
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


def open_description_prompt(window_start, window_end, num_frames):
    # Frozen Open-Ended Visual Perception Benchmark v1 prompt.
    return f"""
Watch these {num_frames} sampled images carefully.

They are ordered chronologically as:

Frame 1, Frame 2, ..., Frame {num_frames}

They represent the patient/person being observed during one 15-second portion
of a psychotherapy video.

Your task is to report ONLY what is directly and visibly happening to the person.

IMPORTANT:
Each observation must describe ONE visible fact only.
Do NOT combine several different observations into one sentence.

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

Pay particular attention to small or brief visible changes that could easily be missed.

A mostly still person is NOT an empty observation window.
If there are few movements, still report meaningful stable visible states such as
head orientation, gaze direction, hand position, posture, or clearly reduced visible movement.

However, do NOT report generic scene facts that are not useful for describing the person's visible behavior.
Do NOT report things such as:
- "the person is seated in a chair"
- clothing descriptions
- furniture
- room contents
- background objects
- physical appearance
unless necessary to explain an occlusion or why something cannot be seen.

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
Use the smallest reasonable frame range in which the movement or change is actually visible.
Use the full frame range only for a genuinely stable visible state.
If the exact first or last frame is uncertain, choose the closest reasonable frame and set certainty to "uncertain".

OUTPUT RULES:
- One observation per JSON item.
- Do not repeat the same unchanged observation several times.
- Do not invent observations merely to increase the number of outputs.
- When several distinct visible observations are present, report them separately.
- Approximately 2-8 observations may be appropriate in a 15-second window, but there is NO required number.
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
ONLY if the person is not sufficiently visible to make any reliable visual observation.
Do not include explanations outside the JSON.
""".strip()


def frame_range_to_seconds(start_frame, end_frame, frame_timestamps, window_start, window_end):
    n = len(frame_timestamps)
    if n == 0:
        raise ValueError("No frame timestamps available")
    start_frame, end_frame = int(start_frame), int(end_frame)
    if not 1 <= start_frame <= n or not 1 <= end_frame <= n:
        raise ValueError(f"frame range {start_frame}-{end_frame} outside 1-{n}")
    if end_frame < start_frame:
        raise ValueError("end_frame < start_frame")
    si, ei = start_frame - 1, end_frame - 1
    start_sec = float(frame_timestamps[si])
    if ei < n - 1:
        end_sec = (float(frame_timestamps[ei]) + float(frame_timestamps[ei + 1])) / 2.0
    else:
        end_sec = float(window_end)
    return round(max(float(window_start), start_sec), 3), round(min(float(window_end), end_sec), 3)


def extract_json_candidate(raw):
    text = str(raw).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    first, last = text.find("{"), text.rfind("}")
    if first < 0 or last < first:
        raise ValueError("No JSON object found in model output")
    return text[first:last + 1]


def salvage_observation_objects(raw):
    text = str(raw)
    match = re.search(r'"observations"\s*:\s*\[', text)
    if not match:
        return [], [{"item_no": -1, "reason": "salvage_no_observations_array", "raw_item": ""}]
    decoder, pos, items, rejected = json.JSONDecoder(), match.end(), [], []
    while pos < len(text):
        array_end, obj_start = text.find("]", pos), text.find("{", pos)
        if obj_start < 0 or (array_end >= 0 and array_end < obj_start):
            break
        try:
            obj, obj_end = decoder.raw_decode(text, obj_start)
            if isinstance(obj, dict):
                items.append(obj)
            pos = obj_end
        except json.JSONDecodeError as exc:
            next_obj = text.find("{", obj_start + 1)
            stop = next_obj if next_obj >= 0 else min(len(text), obj_start + 500)
            rejected.append({"item_no": -1, "reason": f"malformed_object:{exc}", "raw_item": text[obj_start:stop]})
            if next_obj < 0:
                break
            pos = next_obj
    return items, rejected


def parse_output(raw, window_start, window_end, frame_timestamps):
    rejected, rows, method = [], [], "frame_json"
    try:
        obj = json.loads(extract_json_candidate(raw))
        observations = obj.get("observations", [])
        if not isinstance(observations, list):
            raise ValueError("'observations' must be a list")
    except Exception as exc:
        observations, salvage_rejected = salvage_observation_objects(raw)
        rejected.append({"item_no": -1, "reason": f"outer_json_invalid:{exc}", "raw_item": ""})
        rejected.extend(salvage_rejected)
        if not observations:
            raise ValueError(f"Could not parse/salvage observations: {exc}") from exc
        method = "frame_json_salvaged"

    for i, item in enumerate(observations):
        if not isinstance(item, dict):
            rejected.append({"item_no": i, "reason": "observation_not_object", "raw_item": repr(item)})
            continue
        description = str(item.get("description", "")).strip()
        if not description:
            rejected.append({"item_no": i, "reason": "empty_description", "raw_item": json.dumps(item, ensure_ascii=False)})
            continue
        certainty = str(item.get("certainty", "uncertain")).strip().lower()
        if certainty not in {"clear", "uncertain"}:
            certainty = "uncertain"
        try:
            sf, ef = int(item["start_frame"]), int(item["end_frame"])
            ss, es = frame_range_to_seconds(sf, ef, frame_timestamps, window_start, window_end)
        except Exception as exc:
            rejected.append({"item_no": i, "reason": f"invalid_frame_range:{exc}", "raw_item": json.dumps(item, ensure_ascii=False)})
            continue
        rows.append({
            "start_frame": sf, "end_frame": ef, "start_sec": ss, "end_sec": es,
            "description": description, "certainty": certainty,
            "time_normalization": "frame_index_to_timestamp",
        })
    return rows, rejected, method


class LlavaVideoRunner:
    def __init__(self, model_id, max_new_tokens=1100):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Loading {model_id} | dtype={self.dtype} | attention=sdpa", flush=True)
        dtype_name = "bfloat16" if self.dtype == torch.bfloat16 else "float16"
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_id, None, "llava_qwen", torch_dtype=dtype_name,
            device_map="auto", attn_implementation="sdpa",
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"Model device: {self.device}", flush=True)

    def generate(self, frames, prompt):
        video_np = np.stack([np.asarray(x.convert("RGB"), dtype=np.uint8) for x in frames], axis=0)
        video_tensor = self.image_processor.preprocess(video_np, return_tensors="pt")["pixel_values"]
        video_tensor = video_tensor.to(device=self.device, dtype=self.dtype)
        question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        with torch.inference_mode():
            out = self.model.generate(
                input_ids,
                images=[video_tensor],
                modalities=["video"],
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        elapsed = time.time() - started

        # Handle both continuation-only and input+continuation generate APIs.
        if out.ndim == 2 and out.shape[1] >= input_ids.shape[1] and torch.equal(out[0, :input_ids.shape[1]], input_ids[0]):
            decode_ids = out[:, input_ids.shape[1]:]
        else:
            decode_ids = out
        raw = self.tokenizer.batch_decode(decode_ids, skip_special_tokens=True)[0].strip()
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        return raw, elapsed, peak_alloc, peak_reserved


def read_latest_details(path):
    latest = {}
    if not path.exists():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[int(r["segment_idx"])] = r
    return latest


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rebuild_outputs(details_path, csv_path, rejected_path):
    latest = read_latest_details(details_path)
    rows, rejected = [], []
    for seg in sorted(latest):
        d = latest[seg]
        if d.get("status") != "ok":
            continue
        rows.extend(d.get("observations", []))
        for w in d.get("windows", []):
            for r in w.get("rejected", []):
                rejected.append({
                    "model": BACKEND, "model_id": d.get("model_id", MODEL_ID),
                    "segment_idx": seg, "window_start": w.get("window_start"),
                    "window_end": w.get("window_end"), **r,
                })
    (pd.DataFrame(rows) if rows else pd.DataFrame(columns=CSV_COLUMNS)).to_csv(csv_path, index=False, encoding="utf-8-sig")
    rej_cols = ["model","model_id","segment_idx","window_start","window_end","item_no","reason","raw_item"]
    (pd.DataFrame(rejected) if rejected else pd.DataFrame(columns=rej_cols)).to_csv(rejected_path, index=False, encoding="utf-8-sig")


def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    details_path = out / "open_description_details.jsonl"
    csv_path = out / "open_descriptions.csv"
    rejected_path = out / "rejected_rows.csv"

    role_cache_path = Path(args.role_cache)
    role_cache = load_role_cache(role_cache_path)

    segments = pd.read_csv(args.segments_csv)
    segments["segment_idx"] = pd.to_numeric(segments["segment_idx"], errors="raise").astype(int)
    wanted = parse_segment_indices(args.segment_indices)
    if wanted is not None:
        segments = segments[segments["segment_idx"].isin(wanted)].copy()

    latest = read_latest_details(details_path)
    completed = {i for i, r in latest.items() if r.get("status") == "ok"}

    face_detector = get_face_detector(
        model_path=args.yunet_model,
        score_threshold=args.yunet_score_threshold,
        nms_threshold=args.yunet_nms_threshold,
        top_k=args.yunet_top_k,
    )

    print("\nOPEN-ENDED VISUAL DESCRIPTION BENCHMARK - LLaVA-VIDEO")
    print("=" * 64)
    print(f"Model: {args.model_id}")
    print(f"Segments: {len(segments)}")
    print(f"Sampling: {args.sample_fps} FPS | windows: {args.window_seconds}s | frames/window: {args.window_max_frames}")
    print("NO behavior vocabulary. NO 3RS/rupture labels. NO reference loaded.\n")

    runner = LlavaVideoRunner(args.model_id, args.max_new_tokens)

    for pos, row in enumerate(segments.itertuples(index=False), start=1):
        idx = int(row.segment_idx)
        if idx in completed:
            print(f"[{pos}/{len(segments)}] segment {idx}: already done")
            continue
        path = Path(row.segment_path)
        print(f"\n[{pos}/{len(segments)}] segment {idx}: {path.name}")
        started_segment = time.time()
        try:
            full_frames, timestamps, duration = sample_full_frames(path, sample_fps=args.sample_fps, max_duration=args.max_duration)
            fw, fh = full_frames[0].size
            detections = []
            for frame_idx, (frame, ts) in enumerate(zip(full_frames, timestamps)):
                for face in detect_faces(frame, face_detector, min_face_px=args.min_face_px):
                    detections.append({"frame_idx": frame_idx, "timestamp": float(ts), "bbox": face})
            tracks = cluster_static_faces(detections, frame_width=fw, frame_height=fh, center_threshold=args.track_center_threshold)
            tracks = filter_candidate_tracks(
                tracks, total_frames=len(full_frames),
                min_detection_count=args.min_track_detections,
                min_detection_fraction=args.min_track_fraction,
            )
            if not tracks:
                raise RuntimeError("No persistent face candidate")

            preview = out / "track_previews" / f"segment_{idx:03d}_tracks.jpg"
            annotate_tracks(full_frames, timestamps, tracks, preview, max_frames=4)
            selected, method = select_patient_track(
                tracks=tracks, row=row, preview_path=preview,
                cache=role_cache, cache_path=role_cache_path,
                mode=args.patient_selection, forced_side=args.patient_side,
            )
            roi = face_to_person_roi(
                selected["median_face_bbox"], frame_width=fw, frame_height=fh,
                width_mult=args.roi_width_face_mult,
                top_mult=args.roi_top_face_mult,
                bottom_mult=args.roi_bottom_face_mult,
            )
            patient_frames = crop_patient_frames(full_frames, timestamps, roi, frame_width=args.frame_width)
            windows = split_windows(
                patient_frames, timestamps,
                window_seconds=args.window_seconds,
                max_duration=min(args.max_duration, duration),
            )

            segment_obs, window_details = [], []
            for wno, w in enumerate(windows, start=1):
                frames, times = limit_frames_and_times(w["frames"], w["timestamps"], args.window_max_frames)
                print(f"  window {wno}/{len(windows)} {w['start']:.0f}-{w['end']:.0f}s | {len(frames)} frames")
                prompt = open_description_prompt(w["start"], w["end"], len(frames))
                raw, sec, peak_a, peak_r = runner.generate(frames, prompt)
                parsed, rejected, parse_method = parse_output(raw, w["start"], w["end"], times)

                flat = []
                for obs in parsed:
                    row_out = {
                        "model": BACKEND, "model_id": args.model_id, "segment_idx": idx,
                        "video": getattr(row, "video", ""), "patient_id": getattr(row, "patient_id", ""),
                        "session_id": getattr(row, "session_id", ""),
                        "window_start": float(w["start"]), "window_end": float(w["end"]), **obs,
                    }
                    segment_obs.append(row_out)
                    flat.append(row_out)

                window_details.append({
                    "window_no": wno, "window_start": float(w["start"]), "window_end": float(w["end"]),
                    "frame_timestamps": [float(x) for x in times], "raw": raw,
                    "parse_method": parse_method, "parsed": parsed, "rejected": rejected,
                    "inference_sec": sec, "peak_allocated_gb": peak_a, "peak_reserved_gb": peak_r,
                })
                print(f"    {len(parsed)} observations | {sec:.1f}s | peak {peak_a:.1f} GB")
                torch.cuda.empty_cache()

            elapsed = time.time() - started_segment
            append_jsonl(details_path, {
                "segment_idx": idx, "status": "ok", "backend": BACKEND, "model_id": args.model_id,
                "segment_path": str(path), "patient_track_id": selected["track_id"],
                "patient_selection_method": method, "windows": window_details,
                "observations": segment_obs, "elapsed_sec": elapsed,
            })
            rebuild_outputs(details_path, csv_path, rejected_path)
            print(f"  complete: {len(segment_obs)} observations | {elapsed:.1f}s")

        except Exception as exc:
            elapsed = time.time() - started_segment
            print(f"ERROR segment {idx}: {exc}")
            traceback.print_exc()
            append_jsonl(details_path, {
                "segment_idx": idx, "status": "error", "backend": BACKEND, "model_id": args.model_id,
                "segment_path": str(path), "error": repr(exc), "elapsed_sec": elapsed,
            })
            rebuild_outputs(details_path, csv_path, rejected_path)
            torch.cuda.empty_cache()

    print("\nFinished.")
    print(f"Observations: {csv_path}")
    print(f"Details: {details_path}")
    print(f"Rejected: {rejected_path}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--segments-csv", required=True)
    p.add_argument("--output-dir", default="./output/open_description_benchmark_v1_frozen/llavavideo")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--segment-indices", default="4,10,16,63,65")
    p.add_argument("--sample-fps", type=float, default=2.0)
    p.add_argument("--window-seconds", type=float, default=15.0)
    p.add_argument("--window-max-frames", type=int, default=16)
    p.add_argument("--max-duration", type=float, default=60.0)
    p.add_argument("--frame-width", type=int, default=320)
    p.add_argument("--max-new-tokens", type=int, default=1100)
    p.add_argument("--role-cache", default="./output/qwen3vl_visual_experiment_v5/patient_role_cache.json")
    p.add_argument("--yunet-model", default="./models/face_detection_yunet_2026may.onnx")
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
    p.add_argument("--patient-selection", choices=["interactive","leftmost","rightmost","largest"], default="interactive")
    p.add_argument("--patient-side", choices=["left","right"], default=None)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())