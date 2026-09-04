#!/usr/bin/env python
"""
InternVL3.5-14B — literal behavior detection benchmark
======================================================
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


warnings.filterwarnings(
    "ignore",
    message=r"MatMul8bitLt: inputs will be cast.*",
)

# Existing benchmark preprocessing.
import run_qwen3vl_visual_experiment_v5 as v5

# Reuse the exact benchmark vocabulary/prompt/parser from the Molmo run.
from run_molmo2_8b_behavior_benchmark import (
    append_jsonl,
    limit_frames_and_times,
    load_completed,
    merge_intervals,
    observer_prompt,
    parse_pipe_output,
    therapist_side_from_tracks,
    write_detections_csv,
)


DEFAULT_MODEL = "OpenGVLab/InternVL3_5-14B"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(
            lambda img: img.convert("RGB")
            if img.mode != "RGB"
            else img
        ),
        T.Resize(
            (input_size, input_size),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.ToTensor(),
        T.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def frames_to_pixel_values(frames, input_size=448):
    """
    Convert one patient-cropped PIL frame to one InternVL image patch.

    max_num=1 in InternVL's official video example is effectively one
    448x448 visual tile per frame. Keeping one tile/frame prevents the
    model-specific tiler from changing the effective number of frames.
    """
    transform = build_transform(input_size)

    pixel_values = []

    for frame in frames:
        frame = frame.convert("RGB")
        pixel_values.append(transform(frame))

    if not pixel_values:
        raise ValueError("No frames supplied to InternVL.")

    pixel_values = torch.stack(pixel_values, dim=0)

    # Exactly one visual patch group for each sampled video frame.
    num_patches_list = [1] * len(pixel_values)

    return pixel_values, num_patches_list


class InternVL35Runner:
    def __init__(
        self,
        model_id,
        input_size=448,
        use_flash_attn=False,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "InternVL3.5 benchmark requires CUDA."
            )

        self.device = torch.device("cuda:0")
        self.input_size = int(input_size)

        print(
            f"Loading {model_id} in bitsandbytes INT8",
            flush=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )

        # Do not call .cuda() or .to() on an 8-bit model after loading.
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=bool(use_flash_attn),
            trust_remote_code=True,
        ).eval()
        print(
            "InternVL3.5 model loaded.",
            flush=True,
        )

        if hasattr(self.model, "hf_device_map"):
            print(
                f"Device map: {self.model.hf_device_map}",
                flush=True,
            )

        allocated = (
            torch.cuda.memory_allocated()
            / (1024 ** 3)
        )
        reserved = (
            torch.cuda.memory_reserved()
            / (1024 ** 3)
        )

        print(
            f"After load: allocated={allocated:.1f} GB | "
            f"reserved={reserved:.1f} GB",
            flush=True,
        )

    def generate(
        self,
        frames,
        prompt,
        max_new_tokens=1200,
    ):
        pixel_values, num_patches_list = (
            frames_to_pixel_values(
                frames,
                input_size=self.input_size,
            )
        )

        pixel_values = pixel_values.to(
            device=self.device,
            dtype=torch.float16,
            non_blocking=True,
        )

        # InternVL's native video convention.
        video_prefix = "".join(
            f"Frame{i + 1}: <image>\n"
            for i in range(len(num_patches_list))
        )

        question = video_prefix + prompt

        generation_config = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
        }

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        started = time.time()

        with torch.inference_mode():
            raw = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            )

        torch.cuda.synchronize()

        elapsed = time.time() - started

        peak_alloc = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved()
            / (1024 ** 3)
        )

        del pixel_values

        return (
            str(raw),
            elapsed,
            peak_alloc,
            peak_reserved,
        )


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    details_jsonl = (
        out_dir
        / "internvl35_14b_behavior_details.jsonl"
    )
    detections_csv = (
        out_dir
        / "internvl35_14b_behavior_detections.csv"
    )
    rejected_csv = (
        out_dir
        / "internvl35_14b_rejected_rows.csv"
    )

    role_cache_path = Path(args.role_cache)
    role_cache = v5.load_role_cache(
        role_cache_path
    )

    segments = pd.read_csv(args.segments_csv)

    segments["segment_idx"] = pd.to_numeric(
        segments["segment_idx"],
        errors="raise",
    ).astype(int)

    requested = v5.parse_segment_indices(
        args.segment_indices
    )

    if requested is not None:
        segments = segments[
            segments["segment_idx"].isin(
                requested
            )
        ].copy()

    done = load_completed(details_jsonl)

    face_detector = v5.get_face_detector(
        model_path=args.yunet_model,
        score_threshold=args.yunet_score_threshold,
        nms_threshold=args.yunet_nms_threshold,
        top_k=args.yunet_top_k,
    )

    print("")
    print(
        "INTERNVL3.5-14B INT8 "
        "BEHAVIOR DETECTION BENCHMARK"
    )
    print("=" * 65)
    print(f"Model: {args.model_id}")
    print("Quantization: bitsandbytes INT8")
    print(f"Segments: {len(segments)}")
    print(f"Sampling: {args.sample_fps} FPS")
    print(
        f"Window: {args.window_seconds}s"
    )
    print(
        "Max frames/window: "
        f"{args.window_max_frames}"
    )
    print(
        f"InternVL input size: "
        f"{args.internvl_input_size}"
    )
    print("NO rupture classification.")
    print("NO audio or transcript.")
    print(
        "NO engineered behavior features."
    )
    print("")

    model = InternVL35Runner(
        model_id=args.model_id,
        input_size=args.internvl_input_size,
        use_flash_attn=args.use_flash_attn,
    )

    all_rejected = []

    for pos, row in enumerate(
        segments.itertuples(index=False),
        start=1,
    ):
        idx = int(row.segment_idx)

        if idx in done:
            print(
                f"[{pos}/{len(segments)}] "
                f"segment {idx}: already done",
                flush=True,
            )
            continue

        segment_path = Path(row.segment_path)

        print(
            f"\n[{pos}/{len(segments)}] "
            f"segment {idx}: "
            f"{segment_path.name}",
            flush=True,
        )

        segment_started = time.time()

        try:
            (
                full_frames,
                timestamps,
                duration,
            ) = v5.sample_full_frames(
                segment_path,
                sample_fps=args.sample_fps,
                max_duration=args.max_duration,
            )

            if not full_frames:
                raise RuntimeError(
                    "No frames sampled."
                )

            fw, fh = full_frames[0].size

            detections = []

            for frame_idx, (frame, ts) in enumerate(
                zip(
                    full_frames,
                    timestamps,
                )
            ):
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
                center_threshold=(
                    args.track_center_threshold
                ),
            )

            tracks = v5.filter_candidate_tracks(
                tracks,
                total_frames=len(full_frames),
                min_detection_count=(
                    args.min_track_detections
                ),
                min_detection_fraction=(
                    args.min_track_fraction
                ),
            )

            if not tracks:
                raise RuntimeError(
                    "No persistent face candidate."
                )

            preview_path = (
                out_dir
                / "track_previews"
                / f"segment_{idx:03d}_tracks.jpg"
            )

            v5.annotate_tracks(
                full_frames,
                timestamps,
                tracks,
                preview_path,
                max_frames=4,
            )

            (
                selected,
                selection_method,
            ) = v5.select_patient_track(
                tracks=tracks,
                row=row,
                preview_path=preview_path,
                cache=role_cache,
                cache_path=role_cache_path,
                mode=args.patient_selection,
                forced_side=args.patient_side,
            )

            therapist_side = (
                therapist_side_from_tracks(
                    tracks,
                    selected,
                )
            )

            patient_roi = v5.face_to_person_roi(
                selected["median_face_bbox"],
                frame_width=fw,
                frame_height=fh,
                width_mult=(
                    args.roi_width_face_mult
                ),
                top_mult=(
                    args.roi_top_face_mult
                ),
                bottom_mult=(
                    args.roi_bottom_face_mult
                ),
            )

            patient_frames = (
                v5.crop_patient_frames(
                    full_frames,
                    timestamps,
                    patient_roi,
                    frame_width=args.frame_width,
                )
            )

            windows = v5.split_windows(
                patient_frames,
                timestamps,
                window_seconds=(
                    args.window_seconds
                ),
                max_duration=min(
                    args.max_duration,
                    duration,
                ),
            )

            segment_rows = []

            detail = {
                "segment_idx": idx,
                "status": "ok",
                "segment_path": str(
                    segment_path
                ),
                "model": args.model_id,
                "quantization": "bnb_int8",
                "patient_track_id": (
                    selected["track_id"]
                ),
                "patient_selection_method": (
                    selection_method
                ),
                "therapist_side": (
                    therapist_side
                ),
                "windows": [],
            }

            for w_no, w in enumerate(
                windows,
                start=1,
            ):
                frames, times = (
                    limit_frames_and_times(
                        w["frames"],
                        w["timestamps"],
                        args.window_max_frames,
                    )
                )

                print(
                    f"  window "
                    f"{w_no}/{len(windows)} "
                    f"{w['start']:.0f}-"
                    f"{w['end']:.0f}s | "
                    f"{len(frames)} frames",
                    flush=True,
                )

                prompt = observer_prompt(
                    w["start"],
                    w["end"],
                    therapist_side,
                )

                (
                    raw,
                    sec,
                    peak_alloc,
                    peak_reserved,
                ) = model.generate(
                    frames,
                    prompt,
                    max_new_tokens=(
                        args.max_new_tokens
                    ),
                )

                parsed = parse_pipe_output(
                    raw,
                    w["start"],
                    w["end"],
                )

                detected_codes = [
                    x["behavior_code"]
                    for x
                    in parsed["detections"]
                ]

                unique_codes = list(
                    dict.fromkeys(
                        detected_codes
                    )
                )

                print(
                    "    "
                    + (
                        ", ".join(unique_codes)
                        if unique_codes
                        else "none"
                    )
                    + f" | {sec:.1f}s"
                    + f" | peak "
                    f"{peak_alloc:.1f} GB",
                    flush=True,
                )

                segment_rows.extend(
                    parsed["detections"]
                )

                for reject in (
                    parsed["rejected"]
                ):
                    all_rejected.append({
                        "segment_idx": idx,
                        "window_start": (
                            w["start"]
                        ),
                        "window_end": (
                            w["end"]
                        ),
                        **reject,
                    })

                detail["windows"].append({
                    "window_start": (
                        w["start"]
                    ),
                    "window_end": (
                        w["end"]
                    ),
                    "frame_timestamps": (
                        times
                    ),
                    "raw": raw,
                    "parsed": parsed,
                    "inference_sec": sec,
                    "peak_allocated_gb": (
                        peak_alloc
                    ),
                    "peak_reserved_gb": (
                        peak_reserved
                    ),
                })

            segment_rows = merge_intervals(
                segment_rows,
                gap=args.merge_gap_sec,
            )

            flat = []

            for x in segment_rows:
                flat.append({
                    "segment_idx": idx,
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
                    "kind": x["kind"],
                    "behavior_code": (
                        x["behavior_code"]
                    ),
                    "start_sec": (
                        x["start_sec"]
                    ),
                    "end_sec": (
                        x["end_sec"]
                    ),
                    "certainty": (
                        x["certainty"]
                    ),
                    "description": (
                        x["description"]
                    ),
                })

            detail["merged_detections"] = (
                flat
            )

            detail["elapsed_sec"] = (
                time.time()
                - segment_started
            )

            append_jsonl(
                details_jsonl,
                detail,
            )

            done.add(idx)

            write_detections_csv(
                details_jsonl,
                detections_csv,
            )

            if all_rejected:
                pd.DataFrame(
                    all_rejected
                ).to_csv(
                    rejected_csv,
                    index=False,
                    encoding="utf-8-sig",
                )

            print(
                f"  complete: "
                f"{len(flat)} detections | "
                f"{time.time() - segment_started:.1f}s",
                flush=True,
            )

            torch.cuda.empty_cache()

        except Exception as exc:
            traceback.print_exc()

            append_jsonl(
                details_jsonl,
                {
                    "segment_idx": idx,
                    "status": "error",
                    "segment_path": str(
                        segment_path
                    ),
                    "model": args.model_id,
                    "quantization": (
                        "bnb_int8"
                    ),
                    "error": repr(exc),
                },
            )

            print(
                f"ERROR segment "
                f"{idx}: {exc}",
                flush=True,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_detections_csv(
        details_jsonl,
        detections_csv,
    )

    if all_rejected:
        pd.DataFrame(
            all_rejected
        ).to_csv(
            rejected_csv,
            index=False,
            encoding="utf-8-sig",
        )

    print("")
    print("Finished.")
    print(
        f"Detections: "
        f"{detections_csv}"
    )
    print(
        f"Details: "
        f"{details_jsonl}"
    )
    print(
        f"Rejected rows: "
        f"{rejected_csv}"
    )


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--segments-csv",
        required=True,
    )

    p.add_argument(
        "--output-dir",
        default=(
            "./output/"
            "internvl35_14b_behavior_benchmark"
        ),
    )

    p.add_argument(
        "--model-id",
        default=DEFAULT_MODEL,
    )

    p.add_argument(
        "--segment-indices",
        default="4,10,16,63,65",
    )

    p.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
    )

    p.add_argument(
        "--window-seconds",
        type=float,
        default=15.0,
    )

    p.add_argument(
        "--window-max-frames",
        type=int,
        default=30,
    )

    p.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
    )

    p.add_argument(
        "--frame-width",
        type=int,
        default=320,
    )

    p.add_argument(
        "--internvl-input-size",
        type=int,
        default=448,
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=1200,
    )

    p.add_argument(
        "--merge-gap-sec",
        type=float,
        default=1.1,
    )

    p.add_argument(
        "--role-cache",
        default=(
            "./output/"
            "qwen3vl_visual_experiment_v5/"
            "patient_role_cache.json"
        ),
    )

    p.add_argument(
        "--yunet-model",
        default=(
            "./models/"
            "face_detection_yunet_2026may.onnx"
        ),
    )

    p.add_argument(
        "--yunet-score-threshold",
        type=float,
        default=0.60,
    )

    p.add_argument(
        "--yunet-nms-threshold",
        type=float,
        default=0.30,
    )

    p.add_argument(
        "--yunet-top-k",
        type=int,
        default=5000,
    )

    p.add_argument(
        "--min-face-px",
        type=int,
        default=24,
    )

    p.add_argument(
        "--track-center-threshold",
        type=float,
        default=0.14,
    )

    p.add_argument(
        "--min-track-detections",
        type=int,
        default=6,
    )

    p.add_argument(
        "--min-track-fraction",
        type=float,
        default=0.08,
    )

    p.add_argument(
        "--roi-width-face-mult",
        type=float,
        default=4.5,
    )

    p.add_argument(
        "--roi-top-face-mult",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--roi-bottom-face-mult",
        type=float,
        default=5.0,
    )

    p.add_argument(
        "--patient-selection",
        choices=[
            "interactive",
            "leftmost",
            "rightmost",
            "largest",
        ],
        default="interactive",
    )

    p.add_argument(
        "--patient-side",
        choices=["left", "right"],
        default=None,
    )

    p.add_argument(
        "--use-flash-attn",
        action="store_true",
    )

    return p


if __name__ == "__main__":
    run(
        build_parser().parse_args()
    )