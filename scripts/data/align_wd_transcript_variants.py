"""Force-align staged transcript wording on the audio machine and slice minute inputs.

Uses stable-ts word alignment (German), with no new ASR transcription. Produces
speaker-free LLM inputs for the controlled wording comparison. Manual timestamp
review is still necessary; automated alignment checks are not ground truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path

VARIANTS = ("amberscript_cohere_ft", "cohere_ft", "cohere_base")


def read_rows(path):
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def wording(text):
    # Same explicit transformation for every provider; original text remains staged.
    return re.sub(r"\s+", " ", re.sub(r"(?im)^\s*(?:T|P\d*|UNKNOWN|SPEAKER[_-]?\d+)\s*:\s*", "", text)).strip()


def tokens(text):
    return re.findall(r"\w+", text.casefold())


def check_words(raw, expected):
    words = [word for segment in raw.get("segments", []) for word in segment.get("words", [])]
    issues = []
    previous = 0.0
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        if not math.isfinite(start + end) or start < 0 or end <= start or start + 0.05 < previous:
            issues.append("invalid_or_nonmonotonic_word_timestamps")
        if end - start > 5:
            issues.append("word_longer_than_5_seconds")
        previous = start
    if not words:
        issues.append("no_aligned_words")
    if tokens(" ".join(str(w["word"]) for w in words)) != tokens(expected):
        issues.append("alignment_dropped_or_changed_words")
    return words, sorted(set(issues))


def build_datasets(args, source_index):
    aligned = {}
    for key, source in source_index.items():
        path = args.output / "alignment" / key[1] / (key[0] + ".json")
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("status") == "automated_checks_passed" and result.get("source_sha256") == source["transcript_sha256"] and result.get("aligner_model") == args.model:
                aligned[key] = result
    counts = {}
    for cohort in ("available_coverage", "paired_control"):
        per_variant = {}
        for variant in VARIANTS:
            rows = read_rows(args.variants_root / variant / cohort / "segment_labels.jsonl")
            prepared = {}
            for row in rows:
                key = (row["session_uid"], row["transcript_provider"])
                result = aligned.get(key)
                if result is None:
                    continue
                selected = [w for w in result["words"] if float(row["start_sec"]) <= (float(w["start"]) + float(w["end"])) / 2 < float(row["end_sec"])]
                text = " ".join(str(w["word"]).strip() for w in selected).strip()
                if not text:
                    continue
                new = dict(row, transcript_text=text, transcript_text_plain=text, llm_ready=True,
                    transcript_status="ALIGNED_AUTOMATED_CHECKS_PASSED", input_format="speaker_free_wording_control",
                    alignment_method="stable_ts_forced_word_alignment", alignment_reviewed=False,
                    speaker_roles_validated=False, timing_status="forced_word_alignment_unreviewed",
                    alignment_audio_sha256=result["audio_sha256"], alignment_model=result["aligner_model"],
                    alignment_word_count=len(selected), review_flags="TIMING_REQUIRES_MANUAL_SPOT_CHECK;NO_SPEAKER_ROLES")
                prepared[row["segment_uid"]] = new
            per_variant[variant] = prepared
        common = set.intersection(*(set(rows) for rows in per_variant.values()))
        for variant, rows in per_variant.items():
            root = args.output / variant / cohort
            common_rows = [rows[uid] for uid in sorted(common)]
            save_rows(root / "segment_manifest.jsonl", common_rows)
            fold_paths = sorted((args.variants_root / variant / cohort).glob("fold_*/alignment_pending.jsonl"))
            for path in fold_paths:
                original = read_rows(path)
                folded = [dict(rows[r["segment_uid"]], split=r["split"]) for r in original if r["segment_uid"] in common]
                # Original fold patient assignments are retained verbatim.
                patients = {}
                for row in folded:
                    patient = str(row["patient_id"])
                    if patient in patients and patients[patient] != row["split"]:
                        raise ValueError(f"Patient leakage in {path}: {patient}")
                    patients[patient] = row["split"]
                save_rows(root / path.parent.name / "master_manifest.jsonl", folded)
            counts[f"{variant}/{cohort}"] = {"ready_common_segments": len(common), "patients": len({r['patient_id'] for r in common_rows}),
                "strict_asr_held_out_segments": sum(r["strict_asr_held_out"] for r in common_rows),
                "nonempty_aligned_before_common_filter": len(rows)}
    save(args.output / "dataset_summary.json", {"datasets": counts, "input_format": "speaker_free_wording_control", "alignment_reviewed": False, "partial_run": args.max_sessions is not None})
    print(json.dumps(counts, indent=2))


def main(args):
    session_sources = {}
    required = set()
    for variant in VARIANTS:
        for row in read_rows(args.variants_root / variant / "session_transcripts.jsonl"):
            session_sources[(row["session_uid"], row["transcript_model"])] = row
        for row in read_rows(args.variants_root / variant / "available_coverage/segment_labels.jsonl"):
            required.add((row["session_uid"], row["transcript_provider"]))
    requests = { (r["session_uid"], r["model"]): r for r in read_rows(args.variants_root / "alignment_requests.jsonl") }
    override = {}
    if args.audio_map:
        with args.audio_map.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["session_uid"] in override:
                    raise ValueError("Duplicate session_uid in audio map")
                override[row["session_uid"]] = row["audio_path"]
    media_index = {}
    if args.media_root:
        for path in args.media_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac", ".mp4", ".m4a"}:
                media_index.setdefault(path.stem, []).append(path)
    planned = []
    for sid, provider in sorted(required):
        candidates = media_index.get(sid, [])
        if sid in override:
            path = Path(override[sid])
        elif args.media_root:
            if len(candidates) > 1:
                raise ValueError(f"Multiple audio/video matches for {sid}; supply --audio-map")
            path = candidates[0] if candidates else None
        else:
            path = Path(requests[(sid, provider)]["media_path"])
        planned.append({"session_uid": sid, "provider": provider, "audio_path": str(path) if path else None,
            "audio_available": bool(path and path.is_file())})
    if args.max_sessions is not None:
        keep = set(sorted({p['session_uid'] for p in planned})[:args.max_sessions])
        planned = [p for p in planned if p['session_uid'] in keep]
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / "alignment_plan.json", planned)
    missing = [p for p in planned if not p["audio_available"]]
    print(f"Plan: {len(planned)} unique transcript/audio alignments; missing audio: {len(missing)}", flush=True)
    if args.plan_only:
        return
    if missing:
        raise FileNotFoundError("Audio missing; inspect alignment_plan.json. Use --media-root or --audio-map.")
    # Exclusive lock prevents two launchers from occupying the GPU for this job.
    lock = args.output / "alignment.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(__import__('os').getpid()))
    try:
        import stable_whisper
        import torch
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; use --device cpu explicitly if intended")
        model = None
        audio_hashes = {}
        failures = []
        for index, job in enumerate(planned, 1):
            sid, provider = job["session_uid"], job["provider"]
            source = session_sources[(sid, provider)]
            path = Path(job["audio_path"])
            if str(path) not in audio_hashes:
                audio_hashes[str(path)] = digest(path)
            audio_hash = audio_hashes[str(path)]
            output = args.output / "alignment" / provider / f"{sid}.json"
            if output.exists():
                previous = json.loads(output.read_text(encoding="utf-8"))
                if previous.get("status") == "automated_checks_passed" and previous.get("source_sha256") == source["transcript_sha256"] and previous.get("audio_sha256") == audio_hash and previous.get("aligner_model") == args.model:
                    print(f"[{index}/{len(planned)}] SKIP {provider}/{sid}", flush=True)
                    continue
            print(f"[{index}/{len(planned)}] ALIGN {provider}/{sid}", flush=True)
            try:
                if model is None:
                    model = stable_whisper.load_model(args.model, device=args.device)
                text = wording(source["transcript_text_original"])
                if not text:
                    raise ValueError("Empty source transcript")
                start = time.monotonic()
                result = model.align(str(path), text, language="de")
                if result is None:
                    raise ValueError("Aligner returned no result")
                words, issues = check_words(result.to_dict(), text)
                record = {"session_uid": sid, "provider": provider, "source_sha256": source["transcript_sha256"],
                    "audio_sha256": audio_hash, "audio_path": str(path), "aligner_model": args.model,
                    "alignment_seconds": time.monotonic() - start, "issues": issues,
                    "status": "review_required" if issues else "automated_checks_passed", "words": words}
                save(output, record)
                if issues:
                    failures.append({"session_uid": sid, "provider": provider, "issues": issues})
            except Exception as error:
                failures.append({"session_uid": sid, "provider": provider, "error": str(error)})
                save(output.with_suffix(".error.json"), failures[-1])
        save(args.output / "alignment_failures.json", failures)
        build_datasets(args, session_sources)
        if failures:
            raise RuntimeError(f"{len(failures)} alignments require review; see alignment_failures.json")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants-root", type=Path, default=Path("output/wd_transcript_variants_v1"))
    parser.add_argument("--output", type=Path, default=Path("output/wd_transcript_variants_aligned_v1"))
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--audio-map", type=Path, help="CSV with session_uid,audio_path; overrides archived paths")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--plan-only", action="store_true")
    main(parser.parse_args())
