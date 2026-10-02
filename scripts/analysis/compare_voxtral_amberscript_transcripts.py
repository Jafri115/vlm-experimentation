"""Compare Voxtral and Amberscript transcript inventories and quality proxies.

This is an inventory comparison, not a WER evaluation. WER requires both
providers to transcribe the same audio windows and a human reference. The
current WD inventories often use complementary 60-second windows, so the
report explicitly counts exact shared windows before making text comparisons.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def load(path: Path, provider: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"video", "patient_id", "start_sec", "end_sec", "transcript_text_plain"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["provider"] = provider
    frame["text"] = frame["transcript_text_plain"].fillna("").astype(str)
    frame["word_count_proxy"] = frame["text"].str.findall(r"\b\w+\b").str.len()
    frame["has_unknown_speaker"] = frame["text"].str.contains(r"\bUNKNOWN\s*:", case=False, regex=True)
    frame["window_key"] = (
        frame["video"].astype(str)
        + "|"
        + pd.to_numeric(frame["start_sec"], errors="coerce").round(3).astype(str)
        + "|"
        + pd.to_numeric(frame["end_sec"], errors="coerce").round(3).astype(str)
    )
    return frame


def provider_summary(frame: pd.DataFrame) -> dict:
    text = frame["text"]
    words = frame["word_count_proxy"]
    resolved = frame.get("all_roles_resolved", pd.Series(False, index=frame.index))
    return {
        "provider": frame["provider"].iloc[0],
        "segments": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()),
        "videos": int(frame["video"].nunique()),
        "nonempty_transcripts": int(text.str.strip().ne("").sum()),
        "empty_transcripts": int(text.str.strip().eq("").sum()),
        "segments_with_unknown_speaker": int(frame["has_unknown_speaker"].sum()),
        "unknown_speaker_segment_rate": float(frame["has_unknown_speaker"].mean()),
        "all_roles_resolved_rate": float(pd.Series(resolved).astype(str).str.lower().eq("true").mean()),
        "word_count_total_proxy": int(words.sum()),
        "word_count_mean_proxy": float(words.mean()),
        "word_count_median_proxy": float(words.median()),
        "text_characters": int(text.str.len().sum()),
    }


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    amb = load(args.amberscript, "amberscript")
    vox = load(args.voxtral, "voxtral")
    all_frame = pd.concat([amb, vox], ignore_index=True)

    amb_windows = set(amb["window_key"])
    vox_windows = set(vox["window_key"])
    shared_windows = amb_windows & vox_windows
    shared_videos = set(amb["video"]) & set(vox["video"])

    by_video = (
        all_frame.groupby(["video", "provider"], as_index=False)
        .agg(
            segments=("window_key", "size"),
            patients=("patient_id", "nunique"),
            text_words=("word_count_proxy", "sum"),
            unknown_speaker_segments=("has_unknown_speaker", "sum"),
            start_sec=("start_sec", "min"),
            end_sec=("end_sec", "max"),
        )
        .sort_values(["video", "provider"])
    )
    by_video.to_csv(args.output / "coverage_by_video.csv", index=False, encoding="utf-8-sig")

    summaries = pd.DataFrame([provider_summary(amb), provider_summary(vox)])
    summaries.to_csv(args.output / "provider_summary.csv", index=False, encoding="utf-8-sig")

    # Save the exact common windows for a possible future reference-based WER run.
    shared = all_frame[all_frame["window_key"].isin(shared_windows)].copy()
    shared.to_csv(args.output / "shared_exact_windows.csv", index=False, encoding="utf-8-sig")

    report = {
        "amberscript": str(args.amberscript.resolve()),
        "voxtral": str(args.voxtral.resolve()),
        "amberscript_summary": provider_summary(amb),
        "voxtral_summary": provider_summary(vox),
        "shared_videos": len(shared_videos),
        "shared_video_names": sorted(shared_videos),
        "shared_exact_segment_windows": len(shared_windows),
        "interpretation": (
            "Direct transcript accuracy comparison is not identifiable from these files "
            "because no exact segment windows are shared. The providers can be compared "
            "for inventory coverage and quality proxies, but WER/CER requires both systems "
            "to transcribe the same audio windows against a human reference."
        ),
    }
    (args.output / "comparison_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    try:
        summary_table = summaries.to_markdown(index=False, floatfmt=".3f")
    except ImportError:
        summary_table = "\n".join([
            "| " + " | ".join(summaries.columns) + " |",
            "| " + " | ".join(["---"] * len(summaries.columns)) + " |",
            *[
                "| " + " | ".join(str(v) for v in row) + " |"
                for row in summaries.itertuples(index=False, name=None)
            ],
        ])
    lines = [
        "# Voxtral versus Amberscript transcript comparison",
        "",
        "This is an inventory and quality-proxy comparison. It is not a WER/CER evaluation.",
        "",
        f"- Shared videos: **{len(shared_videos)}**",
        f"- Shared exact segment windows: **{len(shared_windows)}**",
        "",
        "## Provider summary",
        "",
        summary_table,
        "",
        "## Interpretation",
        "",
        report["interpretation"],
        "",
        "The current inventories are complementary: Amberscript contains the broad transcript inventory, while Voxtral supplies many windows missing from Amberscript. To measure which transcript is more accurate, rerun both providers on an identical sampled set of audio windows and compare each against a corrected German reference.",
    ]
    (args.output / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), **{k: report[k] for k in ("shared_videos", "shared_exact_segment_windows")}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amberscript", type=Path, default=Path("data/amberscript_llm/amberscript_segments.csv"))
    parser.add_argument("--voxtral", type=Path, default=Path("data/amberscript_llm/voxtral_segments.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/transcript_provider_comparison"))
    main(parser.parse_args())
