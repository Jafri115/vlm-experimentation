#!/usr/bin/env python
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_llavavideo_open_description_v1_standalone as bench

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token


SEGMENT_IDX = 4
MODEL_ID = "lmms-lab/LLaVA-Video-7B-Qwen2"


def generate_freeform(runner, frames, prompt, max_new_tokens=300):
    video_np = np.stack(
        [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames],
        axis=0,
    )

    video_tensor = runner.image_processor.preprocess(
        video_np,
        return_tensors="pt",
    )["pixel_values"].to(
        device=runner.device,
        dtype=runner.dtype,
    )

    video = [video_tensor]

    question = DEFAULT_IMAGE_TOKEN + "\n" + prompt

    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)

    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_question,
        runner.tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(runner.device)

    # Clean deterministic generation config so old defaults do not warn.
    gen_config = copy.deepcopy(runner.model.generation_config)
    gen_config.do_sample = False
    gen_config.temperature = None
    gen_config.top_p = None
    gen_config.top_k = None

    with torch.inference_mode():
        output_ids = runner.model.generate(
            input_ids,
            images=video,
            modalities=["video"],
            generation_config=gen_config,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    if (
        output_ids.ndim == 2
        and output_ids.shape[1] >= input_ids.shape[1]
        and torch.equal(
            output_ids[0, :input_ids.shape[1]],
            input_ids[0],
        )
    ):
        decode_ids = output_ids[:, input_ids.shape[1]:]
    else:
        decode_ids = output_ids

    return runner.tokenizer.batch_decode(
        decode_ids,
        skip_special_tokens=True,
    )[0].strip()


def main():
    root = Path.cwd()
    manifest_path = root / "output/visual_pilot_100/segments_manifest.csv"
    role_cache_path = root / "output/qwen3vl_visual_experiment_v5/patient_role_cache.json"
    out_dir = root / "output/llavavideo_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    row_df = manifest[manifest["segment_idx"].astype(int) == SEGMENT_IDX]

    if row_df.empty:
        raise RuntimeError(f"Segment {SEGMENT_IDX} not found in manifest.")

    row = next(row_df.itertuples(index=False))
    segment_path = Path(row.segment_path)

    role_cache = bench.load_role_cache(role_cache_path)

    detector = bench.get_face_detector(
        model_path=root / "models/face_detection_yunet/face_detection_yunet_2026may.onnx",
        score_threshold=0.60,
        nms_threshold=0.30,
        top_k=5000,
    )

    frames, timestamps, duration = bench.sample_full_frames(
        segment_path,
        sample_fps=2.0,
        max_duration=60.0,
    )

    fw, fh = frames[0].size
    detections = []

    for frame_idx, (frame, ts) in enumerate(zip(frames, timestamps)):
        for face in bench.detect_faces(frame, detector, min_face_px=24):
            detections.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp": float(ts),
                    "bbox": face,
                }
            )

    tracks = bench.cluster_static_faces(
        detections,
        frame_width=fw,
        frame_height=fh,
        center_threshold=0.14,
    )
    tracks = bench.filter_candidate_tracks(
        tracks,
        total_frames=len(frames),
        min_detection_count=6,
        min_detection_fraction=0.08,
    )

    selected, method = bench.select_patient_track(
        tracks=tracks,
        row=row,
        preview_path=out_dir / "track_preview.jpg",
        cache=role_cache,
        cache_path=role_cache_path,
        mode="interactive",
        forced_side=None,
    )

    roi = bench.face_to_person_roi(
        selected["median_face_bbox"],
        frame_width=fw,
        frame_height=fh,
        width_mult=4.5,
        top_mult=1.0,
        bottom_mult=5.0,
    )

    patient_frames = bench.crop_patient_frames(
        frames,
        timestamps,
        roi,
        frame_width=320,
    )

    windows = bench.split_windows(
        patient_frames,
        timestamps,
        window_seconds=15.0,
        max_duration=min(60.0, duration),
    )

    window = windows[0]
    input_frames, input_times = bench.limit_frames_and_times(
        window["frames"],
        window["timestamps"],
        16,
    )

    bench.make_contact_sheet(
        input_frames,
        out_dir / "segment4_window1_actual_model_input.jpg",
        cols=4,
        max_frames=16,
        thumb_width=240,
    )

    print()
    print("LLaVA-VIDEO VISUAL PIPELINE DIAGNOSTIC")
    print("=" * 64)
    print("Segment:", SEGMENT_IDX)
    print("Patient selection:", method)
    print("Frames:", len(input_frames))
    print("Timestamps:", [round(float(x), 2) for x in input_times])
    print(
        "Saved actual input contact sheet:",
        out_dir / "segment4_window1_actual_model_input.jpg",
    )

    runner = bench.LlavaVideoRunner(
        model_id=MODEL_ID,
        max_new_tokens=300,
    )

    tests = [
        (
            "TEST 1 - basic visibility",
            "Is a person visible in this video? Answer yes or no, then briefly describe what is visibly present.",
        ),
        (
            "TEST 2 - free visual description",
            (
                "Describe only what you can directly see the person doing in these video frames. "
                "Mention visible posture, head position, gaze if visible, hands, arms, mouth, "
                "and any visible movement. Do not infer emotion or intention. Use plain text, not JSON."
            ),
        ),
        (
            "TEST 3 - movement check",
            (
                "Look across the frames in chronological order. "
                "Does any body part visibly change position? "
                "If yes, describe the visible change. If no, describe the stable visible posture."
            ),
        ),
    ]

    results = []

    for title, prompt in tests:
        print()
        print(title)
        print("-" * 64)
        answer = generate_freeform(
            runner,
            input_frames,
            prompt,
            max_new_tokens=300,
        )
        print(answer)
        results.append(
            {
                "title": title,
                "prompt": prompt,
                "answer": answer,
            }
        )

    (out_dir / "diagnostic_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("Diagnostic complete.")
    print("Results:", out_dir / "diagnostic_results.json")


if __name__ == "__main__":
    main()