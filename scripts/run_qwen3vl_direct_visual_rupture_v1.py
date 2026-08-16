import argparse, json, re, tempfile, time, traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

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
- an expression of disgust directed toward the other person
  when the interactional target is visually clear

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
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < 0:
        raise ValueError(f"No JSON object in model output:\n{text}")
    return json.loads(text[a:b+1])


def parse_indices(value):
    return None if not value else {int(x.strip()) for x in value.split(",") if x.strip()}


def video_duration(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count / fps if count > 0 else 60.0


def resize_frame(frame, max_width):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    nw = max(32, int(round(max_width / 32)) * 32)
    nh = max(32, int(round((h * scale) / 32)) * 32)
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def sample_chunk(path, start, end, fps, width, out_dir):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    n = max(2, int(round((end - start) * fps)))
    times = np.linspace(start, max(start, end - 0.05), n)
    files = []
    for i, ts in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = resize_frame(frame, width)
        fp = Path(out_dir) / f"f_{i:03d}.jpg"
        if cv2.imwrite(str(fp), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            files.append(fp)
    cap.release()
    if len(files) < 2:
        raise RuntimeError(f"Too few frames for {start:.1f}-{end:.1f}s")
    return files


def build_chunks(path, chunk_sec, sample_fps, width, temp_dir):
    duration = min(60.0, video_duration(path))
    chunks, start, idx = [], 0.0, 0
    while start < duration - 0.01:
        end = min(duration, start + chunk_sec)
        cdir = Path(temp_dir) / f"chunk_{idx:02d}"
        cdir.mkdir(parents=True, exist_ok=True)
        files = sample_chunk(path, start, end, sample_fps, width, cdir)
        chunks.append({"index": idx, "start": start, "end": end, "files": files})
        start, idx = end, idx + 1
    return chunks


def build_messages(chunks, roles, sample_fps):
    content = [{"type": "text", "text": (
        "The following visual chunks are consecutive parts of ONE 60-second psychotherapy segment. "
        "Inspect all chunks first and make ONE final binary decision for the complete minute."
    )}]

    for c in chunks:
        content.append({"type": "text", "text": f"Chunk {c['index']+1}: {c['start']:.1f}-{c['end']:.1f}s"})
        content.append({
            "type": "video",
            "video": [str(p.resolve()) for p in c["files"]],
            "sample_fps": float(sample_fps),
        })

    prompt = f"""
ROLE LAYOUT:
{roles}

{VISUAL_DEFINITION}

Return JSON only:
{{
  "label": "RUPTURE",
  "confidence": 0.00,
  "visual_type": "withdrawal|confrontation|mixed|unclear|none",
  "actor": "patient|therapist|both|unclear|none",
  "evidence_windows": [
    {{"start_sec": 0.0, "end_sec": 10.0, "visual_evidence": "literal visible evidence"}}
  ],
  "counterevidence": ["visible behavior arguing against rupture"],
  "reason": "one short visual-only explanation"
}}

Rules:
- label must be exactly RUPTURE or NO_RUPTURE.
- If NO_RUPTURE, visual_type must be none or unclear and actor must be none or unclear.
- confidence = confidence in the binary label, 0.0 to 1.0.
- Evidence must be directly visible. Never invent speech content or tone.
""".strip()
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def validate(result):
    label = str(result.get("label", "NO_RUPTURE")).upper().strip()
    if label not in {"RUPTURE", "NO_RUPTURE"}:
        label = "NO_RUPTURE"
    try:
        conf = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    except Exception:
        conf = 0.0
    typ = str(result.get("visual_type", "none")).lower().strip()
    actor = str(result.get("actor", "none")).lower().strip()
    if typ not in {"withdrawal", "confrontation", "mixed", "unclear", "none"}:
        typ = "unclear"
    if actor not in {"patient", "therapist", "both", "unclear", "none"}:
        actor = "unclear"
    if label == "NO_RUPTURE":
        typ = typ if typ in {"none", "unclear"} else "unclear"
        actor = actor if actor in {"none", "unclear"} else "unclear"
    result.update(label=label, confidence=conf, visual_type=typ, actor=actor)
    result.setdefault("evidence_windows", [])
    result.setdefault("counterevidence", [])
    result.setdefault("reason", "")
    return result


class Classifier:
    def __init__(self, model_id):
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("Model:", model_id, flush=True)
        print("CUDA available:", torch.cuda.is_available(), flush=True)
        print("dtype:", dtype, flush=True)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, device_map="auto", attn_implementation="sdpa"
        )
        self.model.eval()
        print("Model device:", next(self.model.parameters()).device, flush=True)

    def predict(self, chunks, roles, sample_fps, max_new_tokens):
        messages = build_messages(chunks, roles, sample_fps)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, metadata = zip(*videos)
            videos, metadata = list(videos), list(metadata)
        else:
            metadata = None
        inputs = self.processor(
            text=[text], images=images, videos=videos, video_metadata=metadata,
            padding=True, return_tensors="pt", do_resize=False, **video_kwargs
        )
        inputs = inputs.to(next(self.model.parameters()).device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
            )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        raw = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
        return validate(extract_json(raw)), raw, peak


def save_rows(rows, path):
    df = pd.DataFrame(rows)
    if not df.empty:
        df["segment_idx"] = pd.to_numeric(df["segment_idx"], errors="coerce")
        df = df.dropna(subset=["segment_idx"]).sort_values("segment_idx").drop_duplicates("segment_idx", keep="last")
    df.to_csv(path, index=False, encoding="utf-8-sig")


def run(args):
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "direct_visual_rupture_predictions.csv"
    jsonl_path = outdir / "direct_visual_rupture_details.jsonl"

    seg = pd.read_csv(args.segments_csv)
    if not {"segment_idx", "segment_path"}.issubset(seg.columns):
        raise ValueError("segments CSV needs segment_idx and segment_path columns")
    seg["segment_idx"] = pd.to_numeric(seg["segment_idx"], errors="raise").astype(int)
    wanted = parse_indices(args.segment_indices)
    if wanted is not None:
        seg = seg[seg["segment_idx"].isin(wanted)].copy()
    if args.max_segments is not None:
        seg = seg.head(args.max_segments)

    rows, completed = [], set()
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        rows = old.to_dict("records")
        if "status" in old.columns:
            completed = set(pd.to_numeric(old.loc[old.status == "ok", "segment_idx"], errors="coerce").dropna().astype(int))

    clf = Classifier(args.model_id)
    print(f"Segments in run: {len(seg)}", flush=True)
    print(f"Chunks: {args.chunk_sec}s | sampling: {args.sample_fps} fps | width: {args.frame_width}px", flush=True)

    for pos, row in enumerate(seg.itertuples(index=False), 1):
        idx, path = int(row.segment_idx), Path(row.segment_path)
        if idx in completed:
            print(f"[{pos}/{len(seg)}] segment {idx}: already done", flush=True)
            continue
        print(f"\n[{pos}/{len(seg)}] segment {idx}: {path.name}", flush=True)
        started = time.time()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                chunks = build_chunks(path, args.chunk_sec, args.sample_fps, args.frame_width, tmp)
                nframes = sum(len(c["files"]) for c in chunks)
                print(f"Visual chunks: {len(chunks)} | frames sent: {nframes}", flush=True)
                result, raw, peak = clf.predict(chunks, args.role_description, args.sample_fps, args.max_new_tokens)
            elapsed = time.time() - started
            outrow = {
                "segment_idx": idx, "segment_path": str(path), "status": "ok", "error": "",
                "label": result["label"], "confidence": result["confidence"],
                "visual_type": result["visual_type"], "actor": result["actor"],
                "evidence_windows": json.dumps(result["evidence_windows"], ensure_ascii=False),
                "counterevidence": json.dumps(result["counterevidence"], ensure_ascii=False),
                "reason": result["reason"], "num_chunks": len(chunks), "frames_sent": nframes,
                "sample_fps": args.sample_fps, "frame_width": args.frame_width,
                "peak_vram_gb": round(peak, 3) if peak is not None else None,
                "elapsed_sec": round(elapsed, 3),
            }
            rows = [r for r in rows if int(float(r["segment_idx"])) != idx] + [outrow]
            save_rows(rows, csv_path)
            detail = {
                "segment_idx": idx, "segment_path": str(path), "visual_definition": VISUAL_DEFINITION,
                "settings": {"model_id": args.model_id, "chunk_sec": args.chunk_sec,
                             "sample_fps": args.sample_fps, "frame_width": args.frame_width,
                             "role_description": args.role_description},
                "chunks": [{"index": c["index"], "start": c["start"], "end": c["end"], "num_frames": len(c["files"])} for c in chunks],
                "prediction": result, "raw_model_output": raw, "peak_vram_gb": peak,
                "elapsed_sec": elapsed,
            }
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")
            completed.add(idx)
            print(f"Prediction: {result['label']} | confidence={result['confidence']:.2f}", flush=True)
            print(f"Type: {result['visual_type']} | actor: {result['actor']}", flush=True)
            if peak is not None:
                print(f"Peak allocated VRAM: {peak:.2f} GiB", flush=True)
            print(f"Finished in {elapsed:.1f}s", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            elapsed = time.time() - started
            print(f"ERROR segment {idx}: {exc}", flush=True)
            traceback.print_exc()
            err = {"segment_idx": idx, "segment_path": str(path), "status": "error", "error": repr(exc), "elapsed_sec": round(elapsed, 3)}
            rows = [r for r in rows if int(float(r["segment_idx"])) != idx] + [err]
            save_rows(rows, csv_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nFinished", flush=True)
    print("CSV:", csv_path, flush=True)
    print("JSONL:", jsonl_path, flush=True)


def parser():
    p = argparse.ArgumentParser(description="Direct visual-only binary rupture classifier. No RAG, audio, transcript, or caption stage.")
    p.add_argument("--segments-csv", required=True)
    p.add_argument("--output-dir", default="./qwen3vl_direct_visual_baseline")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--segment-indices", default=None)
    p.add_argument("--max-segments", type=int, default=None)
    p.add_argument("--chunk-sec", type=float, default=10.0)
    p.add_argument("--sample-fps", type=float, default=2.0)
    p.add_argument("--frame-width", type=int, default=384)
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--role-description", default="Patient is the person on the LEFT side of the video. Therapist is the person on the RIGHT side of the video.")
    return p


if __name__ == "__main__":
    run(parser().parse_args())