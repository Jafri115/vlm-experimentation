#!/usr/bin/env python3
"""
Build the V7 patient-role cache directly from your existing patient face crops.

REFERENCE INPUT
---------------
A folder like:

    reference_crops/
        401001.jpg
        401003.jpg
        401006.jpg
        ...

The filename stem is the patient_id.

METHOD
------
1. InsightFace embeds each reference patient crop once.
2. For each V7 session, sample frames from its 60-s segment clips.
3. Detect all faces with InsightFace.
4. Compare every detected face with the expected patient's reference crop.
5. Prefer frames where the expected patient also ranks above all OTHER patient
   reference crops and clearly beats the other face in the same frame.
6. Vote whether that matched patient is the LEFTMOST or RIGHTMOST face.
7. Write a V5/V7-compatible role cache:

       {"401043_S5.mp4": {"patient_side": "left", ...}}

No rupture labels are loaded or used.

This does NOT replace V7. It only removes the manual patient-track prompt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def configure_windows_cuda_dlls() -> None:
    if os.name != "nt":
        return

    candidates = []
    try:
        import torch
        candidates.append(Path(torch.__file__).resolve().parent / "lib")
    except Exception:
        pass

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")

    for version in ("v13.1", "v13.0", "v12.8", "v12.6", "v12.4"):
        candidates.append(
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
            / version
            / "bin"
        )

    for directory in candidates:
        if not directory.is_dir():
            continue
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(directory))
            except OSError:
                pass


def norm_embedding(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x))
    if n <= 1e-12:
        raise ValueError("zero-length face embedding")
    return x / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def patient_id_from_row(row: pd.Series) -> str:
    value = str(row.get("patient_id", "") or "").strip()
    if value and value.lower() not in {"nan", "none"}:
        return value

    for field in ("video", "segment_path", "segment_id"):
        text = str(row.get(field, "") or "")
        m = re.search(r"(?<!\d)(\d{6})_S\d+", text, re.I)
        if m:
            return m.group(1)

    raise ValueError(f"Cannot determine patient_id from row: {row.to_dict()}")


def cache_key_from_row(row: pd.Series) -> str:
    video = str(row.get("video", "") or "").strip()
    if video and video.lower() not in {"nan", "none"}:
        return Path(video.replace("\\", "/")).name

    text = str(row.get("segment_path", "") or "")
    name = Path(text.replace("\\", "/")).name
    m = re.search(r"(?P<pid>\d{6})_S0*(?P<sid>\d+)", name, re.I)
    if not m:
        raise ValueError(f"Cannot make V7 cache key from {text!r}")
    return f"{m.group('pid')}_S{int(m.group('sid'))}.mp4"


def upscale_reference(image: np.ndarray, minimum_side: int = 256) -> np.ndarray:
    h, w = image.shape[:2]
    short = min(h, w)
    if short >= minimum_side:
        return image
    scale = minimum_side / max(1, short)
    return cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_CUBIC,
    )


def choose_reference_face(faces: list[Any], image_shape) -> Any | None:
    if not faces:
        return None
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    def key(face):
        box = np.asarray(face.bbox, dtype=float)
        fx = (box[0] + box[2]) / 2.0
        fy = (box[1] + box[3]) / 2.0
        area = max(1.0, box[2] - box[0]) * max(1.0, box[3] - box[1])
        center_dist = math.hypot((fx - cx) / max(w, 1), (fy - cy) / max(h, 1))
        return (area, -center_dist, float(getattr(face, "det_score", 0.0)))

    return max(faces, key=key)


def build_reference_embeddings(app, crop_dir: Path, output_dir: Path):
    records = []
    embeddings = {}

    crop_paths = sorted(
        p for p in crop_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not crop_paths:
        raise FileNotFoundError(f"No reference images found in {crop_dir}")

    print(f"Reference crops found: {len(crop_paths)}", flush=True)

    for path in crop_paths:
        patient_id = path.stem.strip()
        image = cv2.imread(str(path))
        if image is None:
            records.append({
                "patient_id": patient_id,
                "reference_path": str(path),
                "status": "READ_FAILED",
            })
            continue

        working = upscale_reference(image)
        faces = app.get(working)
        face = choose_reference_face(faces, working.shape)

        if face is None or getattr(face, "normed_embedding", None) is None:
            records.append({
                "patient_id": patient_id,
                "reference_path": str(path),
                "status": "FACE_NOT_DETECTED",
                "original_width": image.shape[1],
                "original_height": image.shape[0],
            })
            print(f"  {patient_id}: FACE NOT DETECTED", flush=True)
            continue

        emb = norm_embedding(face.normed_embedding)
        embeddings[patient_id] = emb
        records.append({
            "patient_id": patient_id,
            "reference_path": str(path),
            "status": "OK",
            "original_width": image.shape[1],
            "original_height": image.shape[0],
            "det_score": float(getattr(face, "det_score", math.nan)),
        })
        print(f"  {patient_id}: OK", flush=True)

    ref_df = pd.DataFrame(records)
    ref_df.to_csv(
        output_dir / "reference_crop_embedding_status.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not embeddings:
        raise RuntimeError("No usable patient reference embeddings were created.")

    ids = sorted(embeddings)
    matrix = np.stack([embeddings[x] for x in ids], axis=0)
    np.savez_compressed(
        output_dir / "reference_crop_embeddings.npz",
        patient_ids=np.asarray(ids),
        embeddings=matrix,
    )

    return embeddings


def sample_video_frames(video_path: Path, samples: int) -> list[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 60.0

    # Avoid exact first/last frames, which are often less useful.
    times = np.linspace(
        max(0.25, duration * 0.05),
        max(0.5, duration * 0.95),
        samples,
    )

    output = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            output.append((float(t), frame))

    cap.release()
    return output


def side_of_face(face: Any, faces: list[Any]) -> str | None:
    """Return whether selected face is leftmost or rightmost among visible faces."""
    if len(faces) < 2:
        return None

    centers = []
    for f in faces:
        box = np.asarray(f.bbox, dtype=float)
        centers.append(((box[0] + box[2]) / 2.0, f))
    centers.sort(key=lambda x: x[0])

    selected_center = (float(face.bbox[0]) + float(face.bbox[2])) / 2.0
    left_center = centers[0][0]
    right_center = centers[-1][0]

    if abs(selected_center - left_center) <= abs(selected_center - right_center):
        return "left"
    return "right"


def evaluate_frame(
    app,
    frame: np.ndarray,
    patient_id: str,
    reference_embeddings: dict[str, np.ndarray],
    face_margin_threshold: float,
):
    faces = app.get(frame)
    if not faces:
        return None

    expected = reference_embeddings[patient_id]
    all_ids = list(reference_embeddings.keys())

    face_rows = []
    for face in faces:
        if getattr(face, "normed_embedding", None) is None:
            continue
        emb = norm_embedding(face.normed_embedding)
        expected_similarity = cosine(emb, expected)

        ref_scores = {
            pid: cosine(emb, reference_embeddings[pid])
            for pid in all_ids
        }
        ranked_refs = sorted(ref_scores.items(), key=lambda kv: kv[1], reverse=True)
        expected_rank = 1 + next(
            i for i, (pid, _) in enumerate(ranked_refs)
            if pid == patient_id
        )
        best_other_ref = max(
            (score for pid, score in ref_scores.items() if pid != patient_id),
            default=-1.0,
        )

        face_rows.append({
            "face": face,
            "embedding": emb,
            "expected_similarity": expected_similarity,
            "expected_rank": expected_rank,
            "reference_margin": expected_similarity - best_other_ref,
        })

    if not face_rows:
        return None

    face_rows.sort(
        key=lambda x: (
            -x["expected_rank"],
            x["expected_similarity"],
            x["reference_margin"],
        ),
        reverse=True,
    )
    best = face_rows[0]

    second_face_similarity = (
        max(x["expected_similarity"] for x in face_rows[1:])
        if len(face_rows) > 1
        else -1.0
    )
    face_margin = best["expected_similarity"] - second_face_similarity
    side = side_of_face(best["face"], [x["face"] for x in face_rows])

    strong = (
        side is not None
        and best["expected_rank"] == 1
        and face_margin >= face_margin_threshold
    )

    box = np.asarray(best["face"].bbox, dtype=float)
    return {
        "face_count": len(face_rows),
        "side": side,
        "expected_similarity": float(best["expected_similarity"]),
        "expected_reference_rank": int(best["expected_rank"]),
        "reference_margin": float(best["reference_margin"]),
        "face_margin": float(face_margin),
        "strong_vote": int(strong),
        "bbox_x1": float(box[0]),
        "bbox_y1": float(box[1]),
        "bbox_x2": float(box[2]),
        "bbox_y2": float(box[3]),
    }


def choose_session_side(rows: list[dict[str, Any]], min_strong_votes: int):
    side_rows = [x for x in rows if x.get("side") in {"left", "right"}]
    strong_rows = [x for x in side_rows if int(x.get("strong_vote", 0)) == 1]

    source_rows = strong_rows if len(strong_rows) >= min_strong_votes else side_rows
    if not source_rows:
        return None

    left = sum(x["side"] == "left" for x in source_rows)
    right = sum(x["side"] == "right" for x in source_rows)
    side = "left" if left >= right else "right"
    votes_for_side = max(left, right)
    consistency = votes_for_side / max(1, len(source_rows))

    return {
        "patient_side": side,
        "vote_source": "strong" if source_rows is strong_rows else "all_matched_frames",
        "votes_used": len(source_rows),
        "left_votes": left,
        "right_votes": right,
        "side_consistency": consistency,
        "strong_vote_count": len(strong_rows),
        "median_expected_similarity": float(np.median([
            x["expected_similarity"] for x in source_rows
        ])),
        "median_face_margin": float(np.median([
            x["face_margin"] for x in source_rows
        ])),
        "median_reference_margin": float(np.median([
            x["reference_margin"] for x in source_rows
        ])),
    }


def run(args):
    configure_windows_cuda_dlls()

    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:
        raise RuntimeError(
            "InsightFace is not available. Use the Python environment where "
            "your prescreen identity code already worked, or install insightface + onnxruntime."
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.provider == "cuda"
        else ["CPUExecutionProvider"]
    )
    ctx_id = 0 if args.provider == "cuda" else -1

    app = FaceAnalysis(
        name="buffalo_l",
        root=str(args.insightface_root.expanduser().resolve()),
        providers=providers,
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(
        ctx_id=ctx_id,
        det_size=(args.det_size, args.det_size),
    )

    print("V7 AUTO PATIENT IDENTITY FROM REFERENCE CROPS", flush=True)
    print(f"Reference crops: {args.reference_crops_dir}", flush=True)
    print(f"Provider: {','.join(app.models['recognition'].session.get_providers())}", flush=True)
    print("Rupture labels used: NO", flush=True)

    references = build_reference_embeddings(
        app,
        args.reference_crops_dir,
        args.output_dir,
    )

    manifest = pd.read_csv(args.segments_csv)

    forbidden = {
        "human_label", "human_binary", "WD_P", "WD_T", "CF_P", "CF_T",
        "WD_P_mean", "WD_T_mean", "CF_P_mean", "CF_T_mean",
    }
    leakage = forbidden.intersection(manifest.columns)
    if leakage:
        raise ValueError(
            "Identity stage must remain rupture-blind. Remove label columns: "
            f"{sorted(leakage)}"
        )

    manifest["_patient_id"] = manifest.apply(patient_id_from_row, axis=1)
    manifest["_cache_key"] = manifest.apply(cache_key_from_row, axis=1)

    missing_refs = sorted(
        set(manifest["_patient_id"].astype(str)) - set(references)
    )
    if missing_refs:
        raise ValueError(
            "No usable reference crop embedding for patient(s): "
            + ", ".join(missing_refs)
        )

    cache = {}
    audit_rows = []
    review_rows = []
    frame_rows = []

    groups = list(manifest.groupby("_cache_key", sort=False))
    print(f"Sessions to identify: {len(groups)}", flush=True)

    for i, (cache_key, group) in enumerate(groups, start=1):
        patient_id = str(group.iloc[0]["_patient_id"])
        print(f"\n[{i}/{len(groups)}] {cache_key} | patient {patient_id}", flush=True)

        all_frame_results = []
        segment_rows = group.head(args.max_segments_per_session)

        for _, row in segment_rows.iterrows():
            video_path = Path(str(row["segment_path"]))
            segment_idx = int(row["segment_idx"])

            try:
                samples = sample_video_frames(video_path, args.samples_per_segment)
            except Exception as exc:
                print(f"  segment {segment_idx}: cannot sample: {exc}", flush=True)
                continue

            for timestamp, frame in samples:
                result = evaluate_frame(
                    app,
                    frame,
                    patient_id,
                    references,
                    args.face_margin_threshold,
                )
                if result is None:
                    continue

                result.update({
                    "cache_key": cache_key,
                    "patient_id": patient_id,
                    "segment_idx": segment_idx,
                    "timestamp_sec": timestamp,
                })
                all_frame_results.append(result)
                frame_rows.append(result.copy())

        decision = choose_session_side(
            all_frame_results,
            args.min_strong_votes,
        )

        if decision is None:
            raise RuntimeError(
                f"Could not determine patient side for {cache_key}. "
                "No frames contained enough matched faces."
            )

        confidence = (
            "HIGH"
            if decision["strong_vote_count"] >= args.min_strong_votes
            and decision["side_consistency"] >= args.high_confidence_consistency
            else "REVIEW"
        )

        cache[cache_key] = {
            "patient_side": decision["patient_side"],
            "source": "insightface_reference_crop",
            "identity_confidence": confidence,
            "expected_patient_id": patient_id,
            **decision,
        }

        audit = {
            "cache_key": cache_key,
            "patient_id": patient_id,
            "patient_side": decision["patient_side"],
            "identity_confidence": confidence,
            **decision,
        }
        audit_rows.append(audit)

        if confidence != "HIGH":
            review_rows.append(audit.copy())

        print(
            f"  -> {decision['patient_side'].upper()} | {confidence} | "
            f"strong={decision['strong_vote_count']} | "
            f"votes L/R={decision['left_votes']}/{decision['right_votes']} | "
            f"consistency={decision['side_consistency']:.2f} | "
            f"median face margin={decision['median_face_margin']:.3f}",
            flush=True,
        )

    args.role_cache_output.parent.mkdir(parents=True, exist_ok=True)
    args.role_cache_output.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pd.DataFrame(audit_rows).to_csv(
        args.output_dir / "identity_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(review_rows).to_csv(
        args.output_dir / "identity_review_queue.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(frame_rows).to_csv(
        args.output_dir / "identity_frame_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFinished.", flush=True)
    print(f"Role cache: {args.role_cache_output}", flush=True)
    print(f"Audit: {args.output_dir / 'identity_audit.csv'}", flush=True)
    print(f"Review queue: {args.output_dir / 'identity_review_queue.csv'}", flush=True)
    print(f"Cache entries: {len(cache)}", flush=True)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segments-csv", type=Path, required=True)
    p.add_argument("--reference-crops-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--role-cache-output", type=Path, required=True)

    p.add_argument(
        "--insightface-root",
        type=Path,
        default=Path.home() / ".insightface",
    )
    p.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--det-size", type=int, default=640)

    p.add_argument("--samples-per-segment", type=int, default=12)
    p.add_argument("--max-segments-per-session", type=int, default=2)
    p.add_argument("--face-margin-threshold", type=float, default=0.05)
    p.add_argument("--min-strong-votes", type=int, default=2)
    p.add_argument("--high-confidence-consistency", type=float, default=0.75)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
