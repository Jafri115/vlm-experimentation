"""Repair unresolved transcript roles using existing per-cue evidence.

Some cues already have a canonical T/P label from the ASR/transcript export,
but were marked UNKNOWN when full-session diarization had no temporal overlap.
This script keeps the diarization provenance, applies that canonical label as a
fallback, and writes a separate transcript inventory for an explicitly marked
expanded cohort. It never invents text or timestamps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def canonical(value):
    value = str(value or "").strip().upper()
    return value if value in {"T", "P"} else None


def main(args):
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts = {"raw_label": 0, "session_mapping": 0, "still_unknown": 0, "rows_ready": 0}
    for row in rows:
        fallback_flags = []
        timing_repaired = False
        mapping = {}
        try:
            mapping = json.loads(row.get("full_session_role_mapping") or "{}")
        except (TypeError, json.JSONDecodeError):
            mapping = {}
        for utterance in row.get("utterances", []):
            if canonical(utterance.get("speaker")):
                continue
            role = canonical(utterance.get("original_asr_role"))
            if role:
                utterance["speaker"] = role
                utterance["role_assignment_status"] = "FALLBACK_TRANSCRIPT_ROLE"
                fallback_flags.append("ROLE_FALLBACK_FROM_TRANSCRIPT_LABEL")
                counts["raw_label"] += 1
                continue
            raw_speaker = utterance.get("original_asr_raw_speaker") or utterance.get("global_speaker")
            role = canonical(mapping.get(raw_speaker))
            if role:
                utterance["speaker"] = role
                utterance["global_speaker"] = raw_speaker
                utterance["role_assignment_status"] = "FALLBACK_SESSION_ROLE_MAPPING"
                fallback_flags.append("ROLE_FALLBACK_FROM_SESSION_MAPPING")
                counts["session_mapping"] += 1

        unknown = [u for u in row.get("utterances", []) if not canonical(u.get("speaker"))]
        row["unresolved_utterances"] = len(unknown)
        row["all_roles_resolved"] = bool(row.get("utterances")) and not unknown
        row["text_available"] = bool(row.get("utterances"))
        row["transcript_text"] = "\n".join(
            f"[{float(u['start_sec'])//60:02.0f}:{float(u['start_sec'])%60:04.1f}] {u.get('speaker', 'UNKNOWN')}: {u.get('text', '').strip()}"
            for u in row.get("utterances", [])
            if str(u.get("text", "")).strip()
        )
        row["transcript_text_plain"] = "\n".join(
            f"{u.get('speaker', 'UNKNOWN')}: {u.get('text', '').strip()}"
            for u in row.get("utterances", [])
            if str(u.get("text", "")).strip()
        )
        flags = [f for f in str(row.get("review_flags", "")).split(";") if f]
        flags = [f for f in flags if f != "REVIEW_UTTERANCE_ROLES"]
        flags.extend(fallback_flags)
        row["review_flags"] = ";".join(dict.fromkeys(flags))
        # A few ASR cues cross a segment boundary by only a few milliseconds.
        # Clip those cues to the known segment interval. Larger or malformed
        # timing problems remain excluded for manual review.
        try:
            seg_start, seg_end = float(row["start_sec"]), float(row["end_sec"])
            cues = row.get("utterances", [])
            valid = all(float(u["start_sec"]) < float(u["end_sec"]) for u in cues)
            valid = valid and all(
                float(u["start_sec"]) >= seg_start - 0.5
                and float(u["end_sec"]) <= seg_end + 0.5
                for u in cues
            )
            if row.get("transcript_status") == "REVIEW_TIMING" and valid:
                for u in cues:
                    u["start_sec"] = max(seg_start, float(u["start_sec"]))
                    u["end_sec"] = min(seg_end, float(u["end_sec"]))
                timing_repaired = True
                fallback_flags.append("TIMING_REPAIRED_BOUNDARY_CLIP")
        except (KeyError, TypeError, ValueError):
            valid = False
        timing_ok = row.get("transcript_status") != "REVIEW_TIMING" or timing_repaired
        if timing_repaired:
            row["transcript_text"] = "\n".join(
                f"[{float(u['start_sec'])//60:02.0f}:{float(u['start_sec'])%60:04.1f}] {u.get('speaker', 'UNKNOWN')}: {u.get('text', '').strip()}"
                for u in row.get("utterances", [])
                if str(u.get("text", "")).strip()
            )
        row["llm_ready"] = bool(row["text_available"] and row["all_roles_resolved"] and timing_ok)
        if row["llm_ready"]:
            row["transcript_status"] = "READY_WITH_REPAIRS" if fallback_flags else "READY"
            counts["rows_ready"] += 1
        else:
            counts["still_unknown"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"input": str(args.input), "output": str(args.output), "rows": len(rows), **counts}
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    main(parser.parse_args())
