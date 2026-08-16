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
# V4.2 DESIGN — OpenCV 5 / YuNet + Git-LFS-safe download
# ---------------------------------------------------------------------
# 1. Sample ~60 full frames (1 FPS).
# 2. Detect face locations with OpenCV's bundled Haar face detector.
# 3. Cluster persistent YuNet face locations into candidate person tracks.
# 4. Select which track is the PATIENT:
#       - automatically with Qwen using annotated track previews, OR
#       - interactively if Qwen is uncertain.
# 5. Build a tight, fixed patient-centered ROI (head + upper body/hands).
# 6. Qwen call A = literal behavior observer ONLY (no rupture decision).
# 7. Qwen call B = text-only WD_P / CF_P judge from the observer JSON.
#
# Human coder labels are never provided to the model.
# ---------------------------------------------------------------------


BEHAVIOR_KEYS = [
    "head_down_sustained",
    "head_or_face_away_repeated",
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

Use only the literal observations supplied by the visual observer.

Possible visual support for WD_P includes a temporal pattern such as:
- sustained/repeated downward or away head/face orientation
- visible retreat, pulling back, slumping, closing posture, or turning away
- crying/face wiping together with other withdrawal-like behavior
- marked reduction in movement together with other movement-away cues
- several weaker cues combining into a clear movement-away pattern

Possible visual support for CF_P includes a temporal pattern such as:
- negative head shaking together with rejecting/oppositional behavior
- pushing-away/dismissive/rejecting gestures
- facial or mouth tension together with visible interactional opposition
- body movement that is visibly against the interaction
- several weaker cues combining into a clear movement-against pattern

Important:
- A single weak cue is usually insufficient.
- Do not invent speech content, tone, criticism, hostility, disagreement, avoidance,
  or therapeutic meaning.
- Ordinary talking, smiling, one brief downward look, one face touch, or one ordinary
  gesture should not by itself become rupture.
- If the observer evidence is weak or ambiguous, use NO_RUPTURE.
""".strip()


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end < start:
        raise ValueError(f"No JSON object found in output:\n{text}")

    return json.loads(text[start:end + 1])


def parse_segment_indices(value):
    if not value:
        return None
    return {int(x.strip()) for x in value.split(",") if x.strip()}


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


def sample_full_frames(video_path, sample_fps=1.0, max_duration=60.0):
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
        raise RuntimeError(f"Too few frames extracted: {len(frames)}")

    return frames, used_times, duration


YUNET_2026_PATH = (
    "models/face_detection_yunet/face_detection_yunet_2026may.onnx"
)

# GitHub stores this ONNX file with Git LFS.
# raw.githubusercontent.com returns only the 131-byte LFS pointer.
# media.githubusercontent.com normally resolves the actual LFS object.
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


def _download_bytes(url, headers=None, timeout=60):
    request_headers = {
        "User-Agent": "Mozilla/5.0",
    }
    if headers:
        request_headers.update(headers)

    request = urllib.request.Request(
        url,
        headers=request_headers,
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _parse_lfs_pointer(data):
    """
    Return (oid, size) if bytes contain a Git-LFS pointer; otherwise None.
    """
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
    """
    Ask GitHub's public Git-LFS batch endpoint for the real object URL,
    then download the actual ONNX bytes.
    """
    payload = json.dumps(
        {
            "operation": "download",
            "transfers": ["basic"],
            "objects": [
                {
                    "oid": oid,
                    "size": int(size),
                }
            ],
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
        raise RuntimeError(
            f"Git-LFS batch response contained no objects: {batch}"
        )

    obj = objects[0]

    if "error" in obj:
        raise RuntimeError(
            f"Git-LFS server returned an error: {obj['error']}"
        )

    download_action = (
        obj.get("actions", {})
        .get("download", {})
    )

    href = download_action.get("href")
    if not href:
        raise RuntimeError(
            f"Git-LFS response did not contain a download URL: {obj}"
        )

    action_headers = download_action.get("header", {}) or {}

    return _download_bytes(
        href,
        headers=action_headers,
        timeout=120,
    )


def ensure_yunet_model(model_path):
    """
    Ensure that the actual YuNet ONNX binary exists locally.

    The OpenCV Zoo ONNX is stored through Git LFS. A normal raw GitHub URL
    returns only a ~131-byte Git-LFS pointer, not the model itself.

    Download strategy:
      1. media.githubusercontent.com (normally resolves Git LFS)
      2. fetch the LFS pointer and resolve it through GitHub's LFS batch API
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # A real model is ~229 KB. Reject tiny stale/pointer files.
    if model_path.exists():
        existing = model_path.read_bytes()

        if len(existing) > 50_000 and _parse_lfs_pointer(existing) is None:
            return model_path

        print(
            f"Removing invalid/stale YuNet file: {model_path} "
            f"({len(existing)} bytes)",
            flush=True,
        )
        model_path.unlink(missing_ok=True)

    print(
        f"YuNet model not found. Downloading actual Git-LFS object to:\n"
        f"  {model_path}",
        flush=True,
    )

    errors = []
    data = None

    # First try GitHub's media endpoint.
    try:
        print("  Trying GitHub media endpoint...", flush=True)
        candidate = _download_bytes(
            YUNET_2026_MEDIA_URL,
            timeout=120,
        )

        pointer = _parse_lfs_pointer(candidate)

        if pointer is None and len(candidate) > 50_000:
            data = candidate
        elif pointer is not None:
            print(
                "  Media endpoint returned an LFS pointer; "
                "resolving through LFS API...",
                flush=True,
            )
            data = _download_lfs_object(*pointer)
        else:
            raise RuntimeError(
                f"unexpected response size: {len(candidate)} bytes"
            )

    except Exception as exc:
        errors.append(f"media endpoint: {exc}")

    # Fallback: get the known pointer and use Git-LFS batch resolution.
    if data is None:
        try:
            print(
                "  Resolving through GitHub Git-LFS batch API...",
                flush=True,
            )

            pointer_bytes = _download_bytes(
                YUNET_2026_POINTER_URL,
                timeout=60,
            )

            pointer = _parse_lfs_pointer(pointer_bytes)

            if pointer is None:
                raise RuntimeError(
                    "GitHub raw response was not a valid Git-LFS pointer. "
                    f"Received {len(pointer_bytes)} bytes."
                )

            oid, size = pointer

            print(
                f"  LFS object: sha256:{oid[:12]}... "
                f"expected size={size:,} bytes",
                flush=True,
            )

            data = _download_lfs_object(oid, size)

        except Exception as exc:
            errors.append(f"LFS batch API: {exc}")

    if data is None:
        raise RuntimeError(
            "Could not download the YuNet ONNX model.\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n\nManual fallback:\n"
            + f"Download the actual ONNX binary and save it as:\n  {model_path}"
        )

    if _parse_lfs_pointer(data) is not None:
        raise RuntimeError(
            "Download still produced a Git-LFS pointer instead of the ONNX binary."
        )

    if len(data) < 50_000:
        raise RuntimeError(
            f"Downloaded YuNet file is unexpectedly small: {len(data)} bytes."
        )

    # The Git-LFS pointer says the expected model is 229,738 bytes.
    # We don't hard-fail on a future upstream model size change, but print it.
    model_path.write_bytes(data)

    print(
        f"YuNet model downloaded successfully: {model_path} "
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
            "This OpenCV build does not expose cv2.FaceDetectorYN. "
            f"OpenCV version: {getattr(cv2, '__version__', 'unknown')}"
        )

    detector = cv2.FaceDetectorYN.create(
        model=str(model_path),
        config="",
        input_size=(320, 320),
        score_threshold=float(score_threshold),
        nms_threshold=float(nms_threshold),
        top_k=int(top_k),
    )

    return detector


def detect_faces(
    image,
    detector,
    scale_factor=1.08,   # retained for CLI compatibility; unused by YuNet
    min_neighbors=5,     # retained for CLI compatibility; unused by YuNet
    min_face_px=28,
):
    """
    Detect faces with OpenCV YuNet / FaceDetectorYN.

    FaceDetectorYN returns rows whose first four values are x, y, w, h.
    The last value is the detection score.
    """
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
    """
    Therapy cameras are mostly static, so face tracks can be created by
    clustering detections that occupy the same spatial location over time.

    all_detections:
      list of {"frame_idx": int, "timestamp": float, "bbox": {...}}
    """
    clusters = []

    for item in all_detections:
        det = item["bbox"]
        cx, cy = bbox_center(det)

        cxn = cx / frame_width
        cyn = cy / frame_height

        best_idx = None
        best_dist = None

        for idx, cluster in enumerate(clusters):
            centers = np.array(cluster["centers"], dtype=float)
            median_c = np.median(centers, axis=0)

            dist = math.sqrt(
                (cxn - median_c[0]) ** 2
                + (cyn - median_c[1]) ** 2
            )

            sizes = np.array(cluster["sizes"], dtype=float)
            median_size = float(np.median(sizes))
            current_size = max(det["w"], det["h"])

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

    # Summarize and sort by persistence.
    summarized = []

    for cluster in clusters:
        boxes = np.array(
            [
                [
                    x["bbox"]["x"],
                    x["bbox"]["y"],
                    x["bbox"]["w"],
                    x["bbox"]["h"],
                ]
                for x in cluster["items"]
            ],
            dtype=float,
        )

        median_box = np.median(boxes, axis=0)
        cx = median_box[0] + median_box[2] / 2.0
        cy = median_box[1] + median_box[3] / 2.0

        summarized.append(
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
                    np.median(
                        np.maximum(boxes[:, 2], boxes[:, 3])
                    )
                ),
            }
        )

    summarized.sort(
        key=lambda x: (
            -x["detection_count"],
            -x["median_face_size_px"],
        )
    )

    for track_id, track in enumerate(summarized):
        track["track_id"] = track_id

    return summarized


def filter_candidate_tracks(
    tracks,
    total_frames,
    min_detection_count=4,
    min_detection_fraction=0.08,
):
    min_count = max(
        int(min_detection_count),
        int(math.ceil(total_frames * min_detection_fraction)),
    )

    kept = [
        t
        for t in tracks
        if t["detection_count"] >= min_count
    ]

    # Re-number after filtering.
    for i, t in enumerate(kept):
        t["track_id"] = i

    return kept


def face_to_person_roi(
    face_bbox,
    frame_width,
    frame_height,
    width_mult=5.0,
    top_mult=1.2,
    bottom_mult=5.5,
):
    """
    Convert median face position into a constant patient-centered ROI
    containing face, shoulders, torso, and as much hand area as possible.
    """
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

    # Clamp.
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(frame_width), x2)
    y2 = min(float(frame_height), y2)

    # Add a small safety margin if crop became too narrow.
    if x2 - x1 < 2.5 * w:
        needed = 2.5 * w - (x2 - x1)
        x1 = max(0.0, x1 - needed / 2.0)
        x2 = min(float(frame_width), x2 + needed / 2.0)

    return {
        "x1": int(round(x1)),
        "y1": int(round(y1)),
        "x2": int(round(x2)),
        "y2": int(round(y2)),
    }


def crop_image(image, roi):
    return image.crop(
        (
            roi["x1"],
            roi["y1"],
            roi["x2"],
            roi["y2"],
        )
    )


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

    label = f"{timestamp:05.1f}s"
    draw.rectangle((8, 8, 92, 34), fill=(0, 0, 0))
    draw.text((13, 12), label, fill=(255, 255, 255))

    return image


def make_contact_sheet(
    frames,
    output_path,
    cols=4,
    max_frames=12,
    thumb_width=240,
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
        h = max(1, int(round(img.height * scale)))

        thumbs.append(
            img.resize(
                (thumb_width, h),
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


def annotate_candidate_tracks(
    frames,
    timestamps,
    tracks,
    output_path,
    max_frames=8,
):
    """
    Save representative full-scene frames with TRACK IDs and the proposed
    upper-body ROI for each detected person.
    """
    if not frames:
        return []

    indices = np.linspace(
        0,
        len(frames) - 1,
        min(max_frames, len(frames)),
        dtype=int,
    )

    annotated = []

    fw, fh = frames[0].size

    for idx in indices:
        img = frames[idx].copy()
        draw = ImageDraw.Draw(img)

        for track in tracks:
            face = track["median_face_bbox"]
            roi = face_to_person_roi(
                face,
                fw,
                fh,
            )

            # Face box.
            fx1 = int(face["x"])
            fy1 = int(face["y"])
            fx2 = int(face["x"] + face["w"])
            fy2 = int(face["y"] + face["h"])

            draw.rectangle(
                (roi["x1"], roi["y1"], roi["x2"], roi["y2"]),
                outline="white",
                width=4,
            )
            draw.rectangle(
                (fx1, fy1, fx2, fy2),
                outline="black",
                width=3,
            )

            label = (
                f"TRACK {track['track_id']} "
                f"({track['detection_count']} detections)"
            )

            tx = max(4, roi["x1"] + 4)
            ty = max(4, roi["y1"] + 4)

            draw.rectangle(
                (tx - 2, ty - 2, tx + 225, ty + 22),
                fill=(0, 0, 0),
            )
            draw.text(
                (tx, ty),
                label,
                fill=(255, 255, 255),
            )

        # Frame time.
        draw.rectangle(
            (8, fh - 34, 100, fh - 8),
            fill=(0, 0, 0),
        )
        draw.text(
            (13, fh - 30),
            f"{timestamps[idx]:05.1f}s",
            fill=(255, 255, 255),
        )

        annotated.append(img)

    make_contact_sheet(
        annotated,
        output_path=output_path,
        cols=2,
        max_frames=len(annotated),
        thumb_width=420,
    )

    return annotated


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

    def generate_from_messages(
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

        kwargs = dict(
            text=[chat_text],
            padding=True,
            return_tensors="pt",
        )

        if image_inputs is not None:
            kwargs["images"] = image_inputs

        if videos is not None:
            kwargs["videos"] = videos
            kwargs["video_metadata"] = video_metadatas

        kwargs.update(video_kwargs)

        # Frames are already explicitly resized/cropped.
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


def role_selector_messages(annotated_frames, candidate_ids):
    prompt = f"""
These are representative frames from a psychotherapy recording.

Each persistent person candidate is marked with a TRACK ID.
Candidate track IDs are: {candidate_ids}

Identify which TRACK is the PATIENT rather than the therapist.

Use only visible role cues from the therapy scene, for example:
- who appears to be receiving therapy rather than conducting it
- therapist-like note taking, clipboard/document use, or clinician positioning
- interactional seating and role behavior across the supplied frames

Do NOT identify the person by real-world identity.
Do NOT use filenames or metadata.

If the role cannot be determined visually with reasonable confidence, return
patient_track_id as null.

Return JSON only:

{{
  "patient_track_id": 0,
  "confidence": 0.00,
  "reason": "short visible-role explanation"
}}
""".strip()

    content = []

    for frame in annotated_frames:
        content.append(
            {
                "type": "image",
                "image": frame,
            }
        )

    content.append(
        {
            "type": "text",
            "text": prompt,
        }
    )

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def observer_messages(
    patient_frames,
    sample_fps,
    total_pixels,
):
    flags_template = ",\n    ".join(
        [
            f'"{key}": {{"present": 0, "times_sec": [], "notes": ""}}'
            for key in BEHAVIOR_KEYS
        ]
    )

    prompt = f"""
You are a literal visual behavior observer.

The video has already been cropped to follow ONE PATIENT in a psychotherapy
session. You are NOT deciding whether a rupture exists.

There is NO audio and NO transcript.

Your job is only to report visible patient behavior across time.

IMPORTANT:
- Do not use rupture, withdrawal, confrontation, alliance, engaged, attentive,
  cooperative, defensive, avoidant, hostile, comfortable, uncomfortable, or
  similar psychological interpretations.
- Do not infer what the patient says or feels.
- Do not claim exact eye contact unless it is unmistakable.
- Prefer literal wording such as:
    head lowered
    face turned toward therapist side
    face turned away
    hand touches chin
    hand wipes cheek/eye area
    lips pressed inward
    head moves side-to-side
    shoulders lift then drop
    torso leans backward
    arms cross
- Distinguish a brief event from a sustained/repeated pattern.
- Compare later behavior with earlier behavior.

Inspect:
0-15 s
15-30 s
30-45 s
45-60 s

Return JSON only:

{{
  "patient_visibility": "good|partial|poor",

  "windows": [
    {{
      "window": "0-15",
      "head_face": "",
      "face_mouth": "",
      "body_shoulders": "",
      "hands_arms": "",
      "movement_change": ""
    }},
    {{
      "window": "15-30",
      "head_face": "",
      "face_mouth": "",
      "body_shoulders": "",
      "hands_arms": "",
      "movement_change": ""
    }},
    {{
      "window": "30-45",
      "head_face": "",
      "face_mouth": "",
      "body_shoulders": "",
      "hands_arms": "",
      "movement_change": ""
    }},
    {{
      "window": "45-60",
      "head_face": "",
      "face_mouth": "",
      "body_shoulders": "",
      "hands_arms": "",
      "movement_change": ""
    }}
  ],

  "events": [
    {{
      "time_sec": 0.0,
      "behavior": "literal visible event"
    }}
  ],

  "behavior_flags": {{
    {flags_template}
  }},

  "overall_literal_summary": "short literal description only"
}}
""".strip()

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": patient_frames,
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


def judge_messages(observer_result):
    observer_json = json.dumps(
        observer_result,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
You are the second-stage visual-only 3RS judge.

You do NOT see the video.
You receive only literal observations produced by a separate visual observer.

{WD_CF_DEFINITION}

OBSERVER OUTPUT:
{observer_json}

Decide only PATIENT withdrawal/confrontation.

Return JSON only:

{{
  "primary_label": "NO_RUPTURE",
  "wd_p_present": 0,
  "cf_p_present": 0,
  "confidence": 0.00,
  "evidence_used": [
    "specific observer evidence used"
  ],
  "counterevidence": [
    "specific observer evidence against the decision"
  ],
  "reason": "short explanation"
}}

Allowed primary_label:
- NO_RUPTURE
- WD_P
- CF_P
- MIXED_P

Consistency rules:
- NO_RUPTURE -> wd_p_present=0, cf_p_present=0
- WD_P       -> wd_p_present=1, cf_p_present=0
- CF_P       -> wd_p_present=0, cf_p_present=1
- MIXED_P    -> wd_p_present=1, cf_p_present=1

Do not invent any behavior that is absent from the observer output.
""".strip()

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
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
        wd = 0
        cf = 0
    elif label == "WD_P":
        wd = 1
        cf = 0
    elif label == "CF_P":
        wd = 0
        cf = 1
    else:
        wd = 1
        cf = 1

    try:
        conf = float(result.get("confidence", 0.0))
    except Exception:
        conf = 0.0

    result["primary_label"] = label
    result["wd_p_present"] = wd
    result["cf_p_present"] = cf
    result["confidence"] = max(0.0, min(1.0, conf))

    result.setdefault("evidence_used", [])
    result.setdefault("counterevidence", [])
    result.setdefault("reason", "")

    return result


def load_json(path, default):
    path = Path(path)

    if not path.exists():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path, obj):
    Path(path).write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def choose_track_by_side(tracks, side):
    if side == "left":
        return min(
            tracks,
            key=lambda t: t["median_center_x_norm"],
        )

    if side == "right":
        return max(
            tracks,
            key=lambda t: t["median_center_x_norm"],
        )

    raise ValueError(side)


def maybe_open_windows(path):
    try:
        if os.name == "nt":
            os.startfile(str(Path(path).resolve()))
    except Exception:
        pass


def select_patient_track(
    args,
    qwen,
    segment_idx,
    tracks,
    annotated_frames,
    preview_path,
    cache,
):
    cache_key = str(segment_idx)

    # Reuse prior reproducible selection.
    if cache_key in cache:
        cached_id = int(cache[cache_key]["patient_track_id"])

        valid_ids = {t["track_id"] for t in tracks}
        if cached_id in valid_ids:
            print(
                f"Patient track from cache: {cached_id}",
                flush=True,
            )
            return (
                cached_id,
                cache[cache_key].get("selection_method", "cache"),
                float(cache[cache_key].get("confidence", 1.0)),
                cache[cache_key].get("reason", "cached"),
            )

    if args.patient_selection == "leftmost":
        track = choose_track_by_side(tracks, "left")
        return (
            track["track_id"],
            "leftmost",
            1.0,
            "forced leftmost candidate",
        )

    if args.patient_selection == "rightmost":
        track = choose_track_by_side(tracks, "right")
        return (
            track["track_id"],
            "rightmost",
            1.0,
            "forced rightmost candidate",
        )

    if args.patient_selection == "largest":
        track = max(
            tracks,
            key=lambda t: (
                t["median_face_size_px"],
                t["detection_count"],
            ),
        )
        return (
            track["track_id"],
            "largest",
            1.0,
            "forced largest candidate",
        )

    if len(tracks) == 1:
        return (
            tracks[0]["track_id"],
            "single_candidate",
            1.0,
            "only one persistent face candidate detected",
        )

    if args.patient_selection == "vlm":
        candidate_ids = [t["track_id"] for t in tracks]

        raw, sec, _, _ = qwen.generate_from_messages(
            role_selector_messages(
                annotated_frames,
                candidate_ids,
            ),
            max_new_tokens=180,
        )

        role = extract_json(raw)

        selected = role.get("patient_track_id", None)

        try:
            selected = int(selected) if selected is not None else None
        except Exception:
            selected = None

        try:
            conf = float(role.get("confidence", 0.0))
        except Exception:
            conf = 0.0

        reason = str(role.get("reason", ""))

        print(
            f"Role selector: track={selected} "
            f"confidence={conf:.2f} ({sec:.1f}s)",
            flush=True,
        )
        print(f"Role reason: {reason}", flush=True)

        valid = {t["track_id"] for t in tracks}

        if (
            selected in valid
            and conf >= args.role_confidence_threshold
        ):
            return selected, "vlm_role_selector", conf, reason

    # VLM unclear/invalid, or explicit interactive mode.
    if sys.stdin.isatty():
        print("", flush=True)
        print(
            f"Patient role needs confirmation. Preview:\n{preview_path}",
            flush=True,
        )

        maybe_open_windows(preview_path)

        for t in tracks:
            print(
                f"  TRACK {t['track_id']}: "
                f"detections={t['detection_count']}, "
                f"x={t['median_center_x_norm']:.2f}, "
                f"face_size={t['median_face_size_px']:.1f}px",
                flush=True,
            )

        valid = {t["track_id"] for t in tracks}

        while True:
            value = input(
                "Enter the PATIENT track ID: "
            ).strip()

            try:
                chosen = int(value)
            except Exception:
                print("Please enter an integer track ID.")
                continue

            if chosen in valid:
                return (
                    chosen,
                    "interactive",
                    1.0,
                    "user selected patient track from preview",
                )

            print(f"Valid track IDs: {sorted(valid)}")

    raise RuntimeError(
        "Could not determine patient track automatically. "
        f"Open {preview_path} and rerun with "
        "--patient-selection interactive, or use "
        "--patient-selection leftmost/rightmost/largest if appropriate."
    )


def load_previous(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        return [], set()

    df = pd.read_csv(csv_path)

    if df.empty:
        return [], set()

    rows = df.to_dict("records")

    ok = df[df["status"] == "ok"] if "status" in df.columns else df.iloc[0:0]

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

    track_preview_dir = output_dir / "track_previews"
    patient_preview_dir = output_dir / "patient_crop_previews"

    csv_path = (
        output_dir
        / "visual_experiment_v4_patient_tracking_predictions.csv"
    )

    jsonl_path = (
        output_dir
        / "visual_experiment_v4_patient_tracking_details.jsonl"
    )

    cache_path = output_dir / "patient_role_cache.json"

    role_cache = load_json(cache_path, {})

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

    face_detector = get_face_detector(
        model_path=args.yunet_model,
        score_threshold=args.yunet_score_threshold,
        nms_threshold=args.yunet_nms_threshold,
        top_k=args.yunet_top_k,
    )

    print(
        f"Face detector: YuNet ({args.yunet_model})",
        flush=True,
    )

    qwen = QwenRunner(args.model_id)

    print(f"Segments in run: {len(segments)}", flush=True)
    print(f"Sampling: {args.sample_fps} fps", flush=True)
    print(
        f"Patient selection: {args.patient_selection}",
        flush=True,
    )
    print(
        f"Patient crop width: <= {args.frame_width}px",
        flush=True,
    )
    print(
        f"Video total-pixel budget: {total_pixels:,} "
        f"({args.video_token_budget} x 32 x 32)",
        flush=True,
    )
    print(
        "Pipeline: face tracks -> patient ROI -> literal observer -> "
        "text-only WD_P/CF_P judge",
        flush=True,
    )

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
                faces = detect_faces(
                    frame,
                    face_detector,
                    scale_factor=args.face_scale_factor,
                    min_neighbors=args.face_min_neighbors,
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
                    "No persistent face tracks were detected. "
                    "This segment needs a manual ROI or a stronger detector."
                )

            for t in tracks:
                print(
                    f"  TRACK {t['track_id']}: "
                    f"detections={t['detection_count']}, "
                    f"x={t['median_center_x_norm']:.2f}, "
                    f"face={t['median_face_size_px']:.1f}px",
                    flush=True,
                )

            track_preview_path = (
                track_preview_dir
                / f"segment_{segment_idx:03d}_tracks.jpg"
            )

            annotated = annotate_candidate_tracks(
                full_frames,
                timestamps,
                tracks,
                track_preview_path,
                max_frames=8,
            )

            patient_track_id, selection_method, role_conf, role_reason = (
                select_patient_track(
                    args=args,
                    qwen=qwen,
                    segment_idx=segment_idx,
                    tracks=tracks,
                    annotated_frames=annotated,
                    preview_path=track_preview_path,
                    cache=role_cache,
                )
            )

            patient_track = next(
                t
                for t in tracks
                if t["track_id"] == patient_track_id
            )

            role_cache[str(segment_idx)] = {
                "patient_track_id": patient_track_id,
                "selection_method": selection_method,
                "confidence": role_conf,
                "reason": role_reason,
                "video": getattr(row, "video", ""),
                "segment_path": str(segment_path),
            }
            save_json(cache_path, role_cache)

            patient_roi = face_to_person_roi(
                patient_track["median_face_bbox"],
                frame_width=fw,
                frame_height=fh,
                width_mult=args.roi_width_face_mult,
                top_mult=args.roi_top_face_mult,
                bottom_mult=args.roi_bottom_face_mult,
            )

            print(
                f"Selected patient TRACK {patient_track_id} "
                f"via {selection_method} "
                f"(confidence={role_conf:.2f})",
                flush=True,
            )
            print(
                f"Patient ROI: "
                f"x={patient_roi['x1']}:{patient_roi['x2']} "
                f"y={patient_roi['y1']}:{patient_roi['y2']}",
                flush=True,
            )

            patient_frames = []

            for frame, ts in zip(full_frames, timestamps):
                crop = crop_image(frame, patient_roi)
                crop = resize_keep_aspect(
                    crop,
                    args.frame_width,
                )
                crop = add_timestamp(crop, ts)
                patient_frames.append(crop)

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
            print(
                f"Frames sent to observer: {len(patient_frames)}",
                flush=True,
            )

            # -----------------------------
            # STAGE A: literal observer
            # -----------------------------
            observer_raw, observer_sec, peak_alloc, peak_reserved = (
                qwen.generate_from_messages(
                    observer_messages(
                        patient_frames,
                        args.sample_fps,
                        total_pixels,
                    ),
                    max_new_tokens=args.observer_max_new_tokens,
                )
            )

            observer_result = extract_json(observer_raw)

            # -----------------------------
            # STAGE B: text-only judge
            # -----------------------------
            judge_raw, judge_sec, _, _ = qwen.generate_from_messages(
                judge_messages(observer_result),
                max_new_tokens=args.judge_max_new_tokens,
            )

            judge_result = validate_judge(
                extract_json(judge_raw)
            )

            elapsed = time.time() - full_started

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
                "patient_track_id": patient_track_id,
                "patient_selection_method": selection_method,
                "patient_role_confidence": role_conf,
                "patient_role_reason": role_reason,
                "face_candidate_count": len(tracks),
                "patient_roi": json.dumps(patient_roi),
                "primary_label": judge_result["primary_label"],
                "wd_p_present": judge_result["wd_p_present"],
                "cf_p_present": judge_result["cf_p_present"],
                "confidence": judge_result["confidence"],
                "observer_patient_visibility": observer_result.get(
                    "patient_visibility",
                    "",
                ),
                "observer_windows": json.dumps(
                    observer_result.get("windows", []),
                    ensure_ascii=False,
                ),
                "observer_events": json.dumps(
                    observer_result.get("events", []),
                    ensure_ascii=False,
                ),
                "observer_behavior_flags": json.dumps(
                    observer_result.get("behavior_flags", {}),
                    ensure_ascii=False,
                ),
                "observer_summary": observer_result.get(
                    "overall_literal_summary",
                    "",
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
                "frames_sent": len(patient_frames),
                "sample_fps": args.sample_fps,
                "frame_width": args.frame_width,
                "video_token_budget": args.video_token_budget,
                "peak_allocated_gb": (
                    round(peak_alloc, 3)
                    if peak_alloc is not None
                    else None
                ),
                "peak_reserved_gb": (
                    round(peak_reserved, 3)
                    if peak_reserved is not None
                    else None
                ),
                "observer_inference_sec": round(observer_sec, 3),
                "judge_inference_sec": round(judge_sec, 3),
                "elapsed_sec": round(elapsed, 3),
            }

            rows = [
                x
                for x in rows
                if int(float(x["segment_idx"])) != segment_idx
            ]
            rows.append(output_row)
            save_rows(rows, csv_path)

            detail = {
                "segment_idx": segment_idx,
                "segment_path": str(segment_path),
                "patient_tracking": {
                    "candidates": tracks,
                    "selected_track_id": patient_track_id,
                    "selection_method": selection_method,
                    "role_confidence": role_conf,
                    "role_reason": role_reason,
                    "patient_roi": patient_roi,
                    "track_preview": str(track_preview_path),
                    "patient_preview": str(patient_preview_path),
                },
                "settings": {
                    "model_id": args.model_id,
                    "sample_fps": args.sample_fps,
                    "frame_width": args.frame_width,
                    "video_token_budget": args.video_token_budget,
                    "total_pixels": total_pixels,
                    "patient_selection": args.patient_selection,
                },
                "timestamps_sec": timestamps,
                "observer": {
                    "parsed": observer_result,
                    "raw": observer_raw,
                    "inference_sec": observer_sec,
                },
                "judge": {
                    "parsed": judge_result,
                    "raw": judge_raw,
                    "inference_sec": judge_sec,
                },
                "peak_allocated_gb": peak_alloc,
                "peak_reserved_gb": peak_reserved,
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
                f"CF_P={judge_result['cf_p_present']} | "
                f"confidence={judge_result['confidence']:.2f}",
                flush=True,
            )

            print(
                f"Observer: {observer_sec:.1f}s | "
                f"Judge: {judge_sec:.1f}s | "
                f"total: {elapsed:.1f}s",
                flush=True,
            )

            if peak_alloc is not None:
                print(
                    f"Peak allocated VRAM: {peak_alloc:.2f} GiB",
                    flush=True,
                )
                print(
                    f"Peak reserved VRAM: {peak_reserved:.2f} GiB",
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
                "elapsed_sec": round(elapsed, 3),
            }

            rows = [
                x
                for x in rows
                if int(float(x["segment_idx"])) != segment_idx
            ]
            rows.append(output_row)
            save_rows(rows, csv_path)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nFinished", flush=True)
    print(f"CSV: {csv_path}", flush=True)
    print(f"JSONL: {jsonl_path}", flush=True)
    print(f"Role cache: {cache_path}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Visual Experiment V4.2: YuNet/LFS face-track patient ROI + "
            "literal visual behavior observer + text-only WD_P/CF_P judge."
        )
    )

    parser.add_argument("--segments-csv", required=True)

    parser.add_argument(
        "--output-dir",
        default="./output/qwen3vl_visual_experiment_v4_patient_tracking",
    )

    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--segment-indices",
        default=None,
        help="Comma-separated indices, e.g. 4,10,16,63,65",
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
        help="Width of final tight patient crop.",
    )

    parser.add_argument(
        "--video-token-budget",
        type=int,
        default=6144,
    )

    # OpenCV 5 face detector: YuNet / FaceDetectorYN.
    parser.add_argument(
        "--yunet-model",
        default="./models/face_detection_yunet_2026may.onnx",
        help=(
            "Path to the OpenCV Zoo YuNet ONNX model. "
            "Downloaded automatically if missing."
        ),
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

    # Tracking / compatibility parameters.
    # The next two options are retained from V4 but are unused by YuNet.
    parser.add_argument(
        "--face-scale-factor",
        type=float,
        default=1.08,
    )
    parser.add_argument(
        "--face-min-neighbors",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--min-face-px",
        type=int,
        default=28,
    )
    parser.add_argument(
        "--track-center-threshold",
        type=float,
        default=0.14,
    )
    parser.add_argument(
        "--min-track-detections",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--min-track-fraction",
        type=float,
        default=0.08,
    )

    # Patient upper-body ROI around median face.
    parser.add_argument(
        "--roi-width-face-mult",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--roi-top-face-mult",
        type=float,
        default=1.2,
    )
    parser.add_argument(
        "--roi-bottom-face-mult",
        type=float,
        default=5.5,
    )

    parser.add_argument(
        "--patient-selection",
        choices=[
            "vlm",
            "interactive",
            "leftmost",
            "rightmost",
            "largest",
        ],
        default="vlm",
        help=(
            "Default 'vlm': Qwen selects patient from numbered persistent "
            "face tracks. If uncertain, script asks you interactively."
        ),
    )

    parser.add_argument(
        "--role-confidence-threshold",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--observer-max-new-tokens",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--judge-max-new-tokens",
        type=int,
        default=320,
    )

    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())