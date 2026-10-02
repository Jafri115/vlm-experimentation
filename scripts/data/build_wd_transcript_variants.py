"""Stage provenance-preserving transcript variants for the WD ASR experiment.

Session text is never divided proportionally across minute-level labels. Pending
fold manifests deliberately have llm_ready=False until alignment is supplied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


VARIANTS = {
    "amberscript_cohere_ft": ("amberscript", "cohere_finetuned"),
    "cohere_ft": ("cohere_finetuned",),
    "cohere_base": ("cohere_base",),
}


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def session_id(row):
    return Path(row["video"]).stem


def extract(path):
    raw = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".txt":
        return raw.strip(), "full_session_untimed"
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        if "prediction" not in data:
            raise ValueError(f"JSON is metadata rather than a transcript: {path}")
        return str(data["prediction"]).strip(), "chunk_timing_unvalidated"
    if path.suffix.lower() == ".csv":
        rows = csv_rows(path)
        if rows and "TEXT" not in rows[0]:
            raise ValueError(f"Unsupported transcript CSV schema: {path}")
        return "\n".join(r["TEXT"] for r in rows if r.get("SPEAKER") != "CAVE"), "csv_timing_unvalidated"
    if path.suffix.lower() == ".srt":
        blocks = re.split(r"\r?\n\s*\r?\n", raw.strip())
        text = []
        durations = []
        for block in blocks:
            lines = block.splitlines()
            idx = next((i for i, line in enumerate(lines) if "-->" in line), None)
            if idx is None:
                continue
            text.append(" ".join(lines[idx + 1:]))
            numbers = re.findall(r"(\d+):(\d+):(\d+)[,.](\d+)", lines[idx])
            if len(numbers) == 2:
                times = [int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000 for h,m,s,ms in numbers]
                durations.append(times[1] - times[0])
        return "\n".join(text), "coarse_chunks_unvalidated" if durations and max(durations) > 60 else "cue_timing_unvalidated"
    raise ValueError(f"Unsupported transcript format: {path}")


def main(args):
    archive = args.archive.resolve()
    release = json.loads((archive / "release.json").read_text(encoding="utf-8"))
    if not release.get("complete"):
        raise ValueError("Archive export is incomplete")
    manifest = csv_rows(archive / "manifest.csv")
    sessions = {r["sample_id"]: r for r in csv_rows(archive / "sessions.csv")}
    sources = {}
    for item in manifest:
        if item["artifact_role"] != "scoring_source":
            continue
        key = (item["sample_id"], item["model"])
        if key in sources:
            raise ValueError(f"Ambiguous scoring source: {key}")
        path = (archive / item["archive_path"]).resolve()
        if not path.is_relative_to(archive):
            raise ValueError(f"Archive path outside root: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Hash mismatch: {path}")
        text, timing = extract(path)
        sources[key] = {
            "transcript_text_original": text,
            "transcript_model": item["model"],
            "transcript_source": item["archive_path"],
            "transcript_sha256": item["sha256"],
            "timing_status": timing,
            "speaker_roles_validated": False,
        }
    with (args.master_root / "fold_1/master_manifest.jsonl").open(encoding="utf-8-sig") as handle:
        labels = [json.loads(line) for line in handle if line.strip()]
    if len({r["segment_uid"] for r in labels}) != len(labels):
        raise ValueError("Duplicate segment IDs in master")
    # Preserve exact original per-fold assignments, rather than generating new ones.
    folds = {}
    for path in sorted(args.master_root.glob("fold_*/master_manifest.jsonl")):
        with path.open(encoding="utf-8-sig") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if {r["segment_uid"] for r in rows} != {r["segment_uid"] for r in labels}:
            raise ValueError(f"Fold cohort differs: {path}")
        folds[path.parent.name] = {r["segment_uid"]: r["split"] for r in rows}
    if not folds:
        raise ValueError("No original folds found")
    paired_sessions = {session_id(r) for r in labels if all((session_id(r), model) in sources for model in ("amberscript", "cohere_finetuned", "cohere_base"))}
    shared_rows = [r for r in labels if session_id(r) in paired_sessions]
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"archive_release": release, "master_root": str(args.master_root.resolve()), "master_segments": len(labels), "master_patients": len({str(r['patient_id']) for r in labels}), "paired_three_source_sessions": len(paired_sessions), "paired_three_source_segments": len(shared_rows), "variants": {}}
    for name, priorities in VARIANTS.items():
        root = args.output / name
        session_rows = []
        for sid, meta in sorted(sessions.items()):
            source = next((sources[(sid, model)] for model in priorities if (sid, model) in sources), None)
            if source is None:
                continue
            session_rows.append({"session_uid": sid, "patient_id": meta["patient_id"], **source,
                "asr_evaluation_role": meta.get("evaluation_role", "unknown"),
                "strict_asr_held_out": meta.get("evaluation_role") == "not_training_or_validation_patient",
                "llm_ready": False,
                "status": "PENDING_AUDIO_ALIGNMENT_AND_SPEAKER_VALIDATION",
                "session_wer_percent": meta.get("cohere_ft_wer_percent" if source["transcript_model"] == "cohere_finetuned" else "cohere_base_wer_percent", "") if source["transcript_model"] != "amberscript" else None})
        selected = {r["session_uid"]: r for r in session_rows}
        jsonl(root / "session_transcripts.jsonl", session_rows)
        for cohort_name, cohort in (("available_coverage", labels), ("paired_control", shared_rows)):
            pending = []
            for old in cohort:
                sid = session_id(old)
                source = selected.get(sid)
                if source is None:
                    continue
                row = {k: v for k,v in old.items() if not k.startswith("transcript") and k not in ("llm_ready", "review_flags", "paired_ready")}
                row.update({"transcript_text": "", "transcript_text_plain": "", "llm_ready": False,
                    "transcript_status": "PENDING_AUDIO_ALIGNMENT_AND_SPEAKER_VALIDATION",
                    "transcript_provider": source["transcript_model"],
                    "transcript_source": source["transcript_source"], "transcript_sha256": source["transcript_sha256"],
                    "timing_status": source["timing_status"], "asr_evaluation_role": source["asr_evaluation_role"],
                    "strict_asr_held_out": source["strict_asr_held_out"], "session_uid": sid})
                pending.append(row)
            for fold, assignments in folds.items():
                jsonl(root / cohort_name / fold / "alignment_pending.jsonl", [dict(r, split=assignments[r["segment_uid"]]) for r in pending])
            jsonl(root / cohort_name / "segment_labels.jsonl", [dict(r, split=None) for r in pending])
        summary["variants"][name] = {"sessions": len(session_rows), "sessions_in_master": len({session_id(r) for r in labels} & set(selected)),
            "master_segments_with_source": sum(session_id(r) in selected for r in labels),
            "providers": dict(Counter(r["transcript_model"] for r in session_rows)),
            "strict_asr_held_out_sessions": sum(r["strict_asr_held_out"] for r in session_rows),
            "training_ready_segments": 0}
    jsonl(args.output / "alignment_requests.jsonl", [{"session_uid": sid, "model": model, "source": source["transcript_source"], "sha256": source["transcript_sha256"], "timing_status": source["timing_status"], "media_path": sessions[sid].get("media_path"), "required": "forced word alignment to original audio; shared validated P/T diarization"} for (sid,model), source in sorted(sources.items()) if model in {"amberscript", "cohere_base", "cohere_finetuned"}])
    (args.output / "build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("artifacts/transcription_artifacts_v1"))
    parser.add_argument("--master-root", type=Path, default=Path("output/wd_nested_fusion_transfer/wd_multimodal_master_expanded"))
    parser.add_argument("--output", type=Path, default=Path("output/wd_transcript_variants_v1"))
    main(parser.parse_args())
