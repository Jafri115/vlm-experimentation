#!/usr/bin/env python
r"""
Create ten 1-minute, patient-face-cropped NO-RUPTURE videos from the middle of sessions.

Definition of strict negative:
For a segment with >=2 independent coders, EVERY coder has:
    WD_P < 3
    WD_T < 3
    CF_P < 3
    CF_T < 3

The script:
1. Reads the merged 3RS coder CSV.
2. Finds strict negative segments.
3. Requires an existing patient-side cache (left/right) so the patient is
   selected consistently.
4. Picks 5 segments from 5 different patients.
5. Resolves the full-session videos recursively under VIDEO_ROOT.
6. Extracts the labelled 1-minute interval.
7. Detects the patient face with YuNet, smooths the crop over time, and writes
   an audio-free MP4.
8. Saves selected_negative_segments.csv with the chosen labels.

Run from:
    C:\Data\Sequence_model\VLM_experiments

Example:
    & ".\.venv_molmo2\Scripts\python.exe" -u `
      ".\scripts\make_5_negative_face_crops.py" `
      --labels-csv ".\completed_segments_merged (1).csv"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


DEFAULT_VIDEO_ROOT = r"C:\Data\Sequence_model\Memopsy_videos\CONVERTED"
DEFAULT_ROLE_CACHE = r".\output\qwen3vl_visual_experiment_v5\patient_role_cache.json"
DEFAULT_YUNET = (
    r".\models\face_detection_yunet\face_detection_yunet_2026may.onnx"
)
DEFAULT_OUTPUT = r".\output\manual_review_negative_face_crops_middle10"


def normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return str(value).strip()


def parse_time(value) -> float:
    if pd.isna(value):
        raise ValueError("Missing time value.")

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()

    try:
        return float(text)
    except ValueError:
        pass

    parts = text.split(":")

    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)

    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)

    raise ValueError(f"Unsupported time format: {value!r}")


def load_role_cache(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Role cache not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def patient_side_for_video(cache: dict, video_name: str):
    """
    Supports role-cache keys such as:
      401043_S5.mp4
      401043_S5
    and case-insensitive matching.
    """
    by_lower = {str(k).lower(): v for k, v in cache.items()}

    candidates = [
        video_name,
        Path(video_name).name,
        Path(video_name).stem,
    ]

    for key in candidates:
        value = cache.get(key)
        if value is None:
            value = by_lower.get(str(key).lower())

        if isinstance(value, dict):
            side = value.get("patient_side")
            if side in {"left", "right"}:
                return side

    return None


def index_videos(video_root: Path):
    index = {}

    for pattern in ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.mkv", "*.MKV"):
        for path in video_root.rglob(pattern):
            index.setdefault(path.name.lower(), path)

    return index


def aggregate_strict_negatives(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)

    score_cols = ["WD_P", "WD_T", "CF_P", "CF_T"]

    required = [
        "video",
        "patient_id",
        "session_id",
        "coder",
        "segment_id",
        "segment_start",
        "segment_end",
        *score_cols,
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    df = df.dropna(subset=score_cols).copy()

    df["patient_id"] = df["patient_id"].map(normalize_id)
    df["session_id"] = df["session_id"].map(normalize_id)
    df["video"] = df["video"].astype(str).str.strip()

    # Each coder must rate every rupture dimension below 3.
    df["coder_no_rupture"] = (df[score_cols] < 3).all(axis=1)

    keys = [
        "video",
        "patient_id",
        "session_id",
        "segment_id",
        "segment_start",
        "segment_end",
    ]

    g = (
        df.groupby(keys, dropna=False)
        .agg(
            n_coders=("coder", "nunique"),
            all_coders_no_rupture=("coder_no_rupture", "all"),
            WD_P_max=("WD_P", "max"),
            WD_T_max=("WD_T", "max"),
            CF_P_max=("CF_P", "max"),
            CF_T_max=("CF_T", "max"),
            WD_P_mean=("WD_P", "mean"),
            WD_T_mean=("WD_T", "mean"),
            CF_P_mean=("CF_P", "mean"),
            CF_T_mean=("CF_T", "mean"),
            coders=("coder", lambda x: "|".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )

    neg = g[
        (g["n_coders"] >= 2)
        & (g["all_coders_no_rupture"])
    ].copy()

    # Prefer very clean negatives (all max ratings close to 1).
    neg["strictness"] = (
        neg["WD_P_max"]
        + neg["WD_T_max"]
        + neg["CF_P_max"]
        + neg["CF_T_max"]
    )

    return neg.sort_values(
        ["strictness", "patient_id", "video", "segment_id"]
    ).reset_index(drop=True)


def get_video_duration(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return float("nan")

    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps and fps > 0 and count and count > 0:
        return float(count / fps)

    return float("nan")


def select_middle_segments(
    neg: pd.DataFrame,
    video_index: dict,
    role_cache: dict,
    n: int = 10,
    middle_start: float = 0.33,
    middle_end: float = 0.67,
):
    """
    Select strict negatives whose 1-minute segment midpoint falls in the
    middle portion of the full therapy-session video.

    Default:
        0.33 <= segment_midpoint / session_duration <= 0.67

    Preference order:
      1. strictest negative ratings
      2. closest to exact session midpoint (0.50)
      3. one segment per patient
    """
    candidates = []

    duration_cache = {}

    for row in neg.itertuples(index=False):
        video_name = Path(str(row.video)).name

        video_path = video_index.get(video_name.lower())
        if video_path is None:
            continue

        side = patient_side_for_video(role_cache, video_name)
        if side not in {"left", "right"}:
            continue

        if str(video_path) not in duration_cache:
            duration_cache[str(video_path)] = get_video_duration(video_path)

        duration = duration_cache[str(video_path)]

        if not np.isfinite(duration) or duration <= 0:
            continue

        start_sec = parse_time(row.segment_start)
        end_sec = parse_time(row.segment_end) + 1.0

        if end_sec <= start_sec:
            end_sec = start_sec + 60.0

        midpoint_sec = (start_sec + end_sec) / 2.0
        relative_position = midpoint_sec / duration

        if not (middle_start <= relative_position <= middle_end):
            continue

        d = row._asdict()
        d["video_path"] = str(video_path)
        d["patient_side"] = side
        d["session_duration_sec"] = float(duration)
        d["segment_midpoint_sec"] = float(midpoint_sec)
        d["relative_session_position"] = float(relative_position)
        d["distance_from_session_middle"] = abs(relative_position - 0.50)

        candidates.append(d)

    if not candidates:
        raise RuntimeError(
            "No strict negative segments were found in the requested "
            "middle-session range with both a resolvable video and cached "
            "patient side."
        )

    c = pd.DataFrame(candidates)

    # First prefer the cleanest negatives, then the segments closest to
    # the exact session midpoint.
    c = c.sort_values(
        [
            "strictness",
            "distance_from_session_middle",
            "patient_id",
            "video",
            "segment_id",
        ]
    )

    # Prefer different patients to make the manual comparison more diverse.
    selected = (
        c.groupby("patient_id", group_keys=False)
        .head(1)
        .head(n)
        .reset_index(drop=True)
    )

    if len(selected) < n:
        # If fewer than n unique patients are available, fill the remainder
        # with additional middle-session negatives from already represented
        # patients, without duplicating the exact same segment.
        selected_keys = set(
            zip(
                selected["video"].astype(str),
                selected["segment_id"].astype(str),
            )
        )

        extras = c[
            ~c.apply(
                lambda r: (str(r["video"]), str(r["segment_id"])) in selected_keys,
                axis=1,
            )
        ]

        need = n - len(selected)
        selected = pd.concat(
            [selected, extras.head(need)],
            ignore_index=True,
        )

    if len(selected) < n:
        raise RuntimeError(
            f"Only {len(selected)} usable strict-negative middle-session "
            f"segments found; requested {n}."
        )

    return selected.head(n).reset_index(drop=True)


def create_detector(yunet_model: Path):
    if not yunet_model.exists():
        raise FileNotFoundError(
            f"YuNet model not found: {yunet_model}"
        )

    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError(
            f"OpenCV {cv2.__version__} does not expose FaceDetectorYN."
        )

    return cv2.FaceDetectorYN.create(
        model=str(yunet_model),
        config="",
        input_size=(320, 320),
        score_threshold=0.55,
        nms_threshold=0.30,
        top_k=5000,
    )


def detect_faces(frame_bgr, detector):
    h, w = frame_bgr.shape[:2]
    detector.setInputSize((w, h))

    _, faces = detector.detect(frame_bgr)

    if faces is None:
        return []

    out = []

    for face in faces:
        x, y, fw, fh = map(float, face[:4])

        if min(fw, fh) < 20:
            continue

        score = float(face[-1])
        out.append((x, y, fw, fh, score))

    return out


def pick_patient_face(faces, frame_width: int, side: str):
    if not faces:
        return None

    side_faces = []

    for face in faces:
        x, y, w, h, score = face
        cx = x + w / 2.0
        this_side = "left" if cx < frame_width / 2.0 else "right"

        if this_side == side:
            side_faces.append(face)

    if not side_faces:
        return None

    # Prefer large/high-confidence face on the cached patient side.
    return max(
        side_faces,
        key=lambda f: (f[2] * f[3], f[4]),
    )


def face_crop_box(
    face,
    frame_width: int,
    frame_height: int,
    margin: float = 1.20,
):
    """
    Square-ish crop centered on the face, with margin for small head movement.

    margin=1.20 means ~2.4 face widths total around center.
    """
    x, y, w, h, _ = face

    cx = x + w / 2.0
    cy = y + h / 2.0

    size = max(w, h) * (1.0 + 2.0 * margin)

    x1 = cx - size / 2.0
    y1 = cy - size / 2.0
    x2 = cx + size / 2.0
    y2 = cy + size / 2.0

    # Clamp while trying to preserve square size.
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > frame_width:
        shift = x2 - frame_width
        x1 -= shift
        x2 = frame_width
    if y2 > frame_height:
        shift = y2 - frame_height
        y1 -= shift
        y2 = frame_height

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width, x2)
    y2 = min(frame_height, y2)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def fallback_side_crop(frame_width: int, frame_height: int, side: str):
    """
    Used only before the first reliable face detection.
    """
    if side == "left":
        x1, x2 = 0, int(frame_width * 0.58)
    else:
        x1, x2 = int(frame_width * 0.42), frame_width

    return np.array(
        [x1, 0, x2, frame_height],
        dtype=np.float32,
    )


def crop_and_resize(frame, box, output_size: int):
    h, w = frame.shape[:2]

    x1, y1, x2, y2 = box
    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(x1 + 1, min(w, round(x2))))
    y2 = int(max(y1 + 1, min(h, round(y2))))

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        raise RuntimeError("Empty crop produced.")

    return cv2.resize(
        crop,
        (output_size, output_size),
        interpolation=cv2.INTER_AREA,
    )


def extract_face_video(
    video_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    patient_side: str,
    detector,
    output_size: int = 384,
    output_fps: float = 25.0,
    detect_every: int = 3,
    smoothing: float = 0.80,
    face_margin: float = 1.20,
):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = 25.0

    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = (
        source_frames / source_fps
        if source_frames > 0
        else float("inf")
    )

    start_sec = max(0.0, float(start_sec))
    end_sec = min(float(end_sec), duration)

    if end_sec <= start_sec:
        cap.release()
        raise RuntimeError(
            f"Invalid time interval {start_sec}-{end_sec} "
            f"for {video_path.name}"
        )

    start_frame = int(round(start_sec * source_fps))
    end_frame = int(round(end_sec * source_fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        float(output_fps),
        (output_size, output_size),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    # Resampling from source FPS to requested output FPS.
    sample_step = source_fps / output_fps
    next_output_source_pos = float(start_frame)

    current_box = None
    last_face_box = None
    decoded_idx = start_frame
    written = 0
    missed_detections = 0

    while decoded_idx < end_frame:
        ok, frame = cap.read()

        if not ok:
            break

        if decoded_idx + 1e-6 >= next_output_source_pos:
            h, w = frame.shape[:2]

            if written % detect_every == 0 or last_face_box is None:
                faces = detect_faces(frame, detector)
                patient_face = pick_patient_face(
                    faces,
                    frame_width=w,
                    side=patient_side,
                )

                if patient_face is not None:
                    detected_box = face_crop_box(
                        patient_face,
                        frame_width=w,
                        frame_height=h,
                        margin=face_margin,
                    )
                    last_face_box = detected_box

                    if current_box is None:
                        current_box = detected_box
                    else:
                        # Exponential smoothing avoids jitter.
                        current_box = (
                            smoothing * current_box
                            + (1.0 - smoothing) * detected_box
                        )
                else:
                    missed_detections += 1

            if current_box is None:
                current_box = fallback_side_crop(
                    frame_width=w,
                    frame_height=h,
                    side=patient_side,
                )

            cropped = crop_and_resize(
                frame,
                current_box,
                output_size=output_size,
            )

            writer.write(cropped)
            written += 1
            next_output_source_pos += sample_step

        decoded_idx += 1

    cap.release()
    writer.release()

    if written == 0:
        raise RuntimeError(f"No frames written for {video_path.name}")

    return {
        "frames_written": written,
        "output_fps": output_fps,
        "duration_written_sec": written / output_fps,
        "missed_face_detection_checks": missed_detections,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--role-cache", default=DEFAULT_ROLE_CACHE)
    parser.add_argument("--yunet-model", default=DEFAULT_YUNET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)

    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--middle-start", type=float, default=0.33,
                        help="Lower relative session position for segment midpoint.")
    parser.add_argument("--middle-end", type=float, default=0.67,
                        help="Upper relative session position for segment midpoint.")
    parser.add_argument("--output-size", type=int, default=384)
    parser.add_argument("--output-fps", type=float, default=25.0)
    parser.add_argument("--face-margin", type=float, default=1.20)

    args = parser.parse_args()

    labels_csv = Path(args.labels_csv)
    video_root = Path(args.video_root)
    role_cache_path = Path(args.role_cache)
    yunet_model = Path(args.yunet_model)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("10 STRICT NEGATIVE MIDDLE-SESSION FACE-CROP VIDEOS")
    print("=" * 72)
    print("Labels:", labels_csv)
    print("Videos:", video_root)
    print("Output:", output_dir)
    print()

    neg = aggregate_strict_negatives(labels_csv)
    print("Strict double-rated negative segments:", len(neg))
    print(
        f"Middle-session filter: {args.middle_start:.2f} - "
        f"{args.middle_end:.2f} of full session duration"
    )

    role_cache = load_role_cache(role_cache_path)
    video_index = index_videos(video_root)

    selected = select_middle_segments(
        neg,
        video_index=video_index,
        role_cache=role_cache,
        n=args.n,
        middle_start=args.middle_start,
        middle_end=args.middle_end,
    )

    detector = create_detector(yunet_model)

    output_rows = []

    for i, row in enumerate(selected.itertuples(index=False), start=1):
        start_sec = parse_time(row.segment_start)

        # Annotation endpoint is typically XX:XX:59; treat it as inclusive.
        end_sec = parse_time(row.segment_end) + 1.0

        # Ensure at most one labelled minute.
        end_sec = min(end_sec, start_sec + 60.0)

        video_path = Path(row.video_path)

        filename = (
            f"NEG_MID_{i:02d}_"
            f"{video_path.stem}_"
            f"seg{int(row.segment_id):03d}_"
            f"{int(start_sec):05d}-{int(end_sec):05d}s_"
            f"FACE.mp4"
        )

        output_path = output_dir / filename

        print(
            f"[{i}/{len(selected)}] "
            f"{video_path.name} | segment {row.segment_id} | "
            f"{start_sec:.1f}-{end_sec:.1f}s | "
            f"patient={row.patient_side}"
        )

        # Set margin globally for this extraction by wrapping the function's
        # face_crop_box behavior through a temporary local replacement would
        # be overkill; default 1.20 is recommended for manual face review.
        stats = extract_face_video(
            video_path=video_path,
            output_path=output_path,
            start_sec=start_sec,
            end_sec=end_sec,
            patient_side=row.patient_side,
            detector=detector,
            output_size=args.output_size,
            output_fps=args.output_fps,
            face_margin=args.face_margin,
        )

        print(
            f"    saved {output_path.name} | "
            f"{stats['duration_written_sec']:.1f}s"
        )

        d = row._asdict()
        d.update(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "cropped_video": str(output_path),
                **stats,
            }
        )
        output_rows.append(d)

    manifest = pd.DataFrame(output_rows)

    manifest_path = output_dir / "selected_negative_middle_segments.csv"
    manifest.to_csv(manifest_path, index=False)

    print()
    print("DONE")
    print("=" * 72)
    print("Videos:", output_dir)
    print("Manifest:", manifest_path)
    print()
    print(
        manifest[
            [
                "video",
                "patient_id",
                "session_id",
                "segment_id",
                "segment_start",
                "segment_end",
                "WD_P_max",
                "WD_T_max",
                "CF_P_max",
                "CF_T_max",
                "patient_side",
                "relative_session_position",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()