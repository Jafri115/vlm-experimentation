#!/usr/bin/env python
from __future__ import annotations

import argparse, copy, json, re, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Existing project helper: same patient tracking/crop used by the frozen benchmark.
import run_qwen3vl_visual_experiment_v5 as v5

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
    role_cache = v5.load_role_cache(role_cache_path)

    segments = pd.read_csv(args.segments_csv)
    segments["segment_idx"] = pd.to_numeric(segments["segment_idx"], errors="raise").astype(int)
    wanted = v5.parse_segment_indices(args.segment_indices)
    if wanted is not None:
        segments = segments[segments["segment_idx"].isin(wanted)].copy()

    latest = read_latest_details(details_path)
    completed = {i for i, r in latest.items() if r.get("status") == "ok"}

    face_detector = v5.get_face_detector(
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
            full_frames, timestamps, duration = v5.sample_full_frames(path, sample_fps=args.sample_fps, max_duration=args.max_duration)
            fw, fh = full_frames[0].size
            detections = []
            for frame_idx, (frame, ts) in enumerate(zip(full_frames, timestamps)):
                for face in v5.detect_faces(frame, face_detector, min_face_px=args.min_face_px):
                    detections.append({"frame_idx": frame_idx, "timestamp": float(ts), "bbox": face})
            tracks = v5.cluster_static_faces(detections, frame_width=fw, frame_height=fh, center_threshold=args.track_center_threshold)
            tracks = v5.filter_candidate_tracks(
                tracks, total_frames=len(full_frames),
                min_detection_count=args.min_track_detections,
                min_detection_fraction=args.min_track_fraction,
            )
            if not tracks:
                raise RuntimeError("No persistent face candidate")

            preview = out / "track_previews" / f"segment_{idx:03d}_tracks.jpg"
            v5.annotate_tracks(full_frames, timestamps, tracks, preview, max_frames=4)
            selected, method = v5.select_patient_track(
                tracks=tracks, row=row, preview_path=preview,
                cache=role_cache, cache_path=role_cache_path,
                mode=args.patient_selection, forced_side=args.patient_side,
            )
            roi = v5.face_to_person_roi(
                selected["median_face_bbox"], frame_width=fw, frame_height=fh,
                width_mult=args.roi_width_face_mult,
                top_mult=args.roi_top_face_mult,
                bottom_mult=args.roi_bottom_face_mult,
            )
            patient_frames = v5.crop_patient_frames(full_frames, timestamps, roi, frame_width=args.frame_width)
            windows = v5.split_windows(
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