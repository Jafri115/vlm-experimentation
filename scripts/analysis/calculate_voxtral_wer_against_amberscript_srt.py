"""Score Voxtral windows against full-session Amberscript SRT captions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


TIMING = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")


def seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(path: Path) -> list[tuple[float, float, str]]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    cues = []
    i = 0
    while i < len(lines):
        match = TIMING.search(lines[i])
        if not match:
            i += 1
            continue
        start = seconds(*match.groups()[:4]); end = seconds(*match.groups()[4:])
        i += 1; text = []
        while i < len(lines) and lines[i].strip():
            text.append(lines[i].strip()); i += 1
        cues.append((start, end, " ".join(text)))
        i += 1
    return cues


def tokens(value: object) -> list[str]:
    text = "" if value is None else str(value)
    text = re.sub(r"\b(?:T|P\d*|UNKNOWN|SPEAKER[_-]?\d+)\s*:\s*", " ", text, flags=re.I)
    text = text.lower().replace("ß", "ss")
    return re.findall(r"[\wäöüÄÖÜ]+", text, flags=re.UNICODE)


def edit_distance(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    n, m = len(ref), len(hyp)
    d = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): d[i][0] = (i, i, 0, 0)
    for j in range(1, m + 1): d[0][j] = (j, 0, j, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]: d[i][j] = d[i - 1][j - 1]; continue
            d[i][j] = min(
                (d[i - 1][j][0] + 1, d[i - 1][j][1] + 1, d[i - 1][j][2], d[i - 1][j][3]),
                (d[i][j - 1][0] + 1, d[i][j - 1][1], d[i][j - 1][2] + 1, d[i][j - 1][3]),
                (d[i - 1][j - 1][0] + 1, d[i - 1][j - 1][1], d[i - 1][j - 1][2], d[i - 1][j - 1][3] + 1),
                key=lambda x: x[0],
            )
    return d[n][m]


def main(args: argparse.Namespace) -> None:
    vox = pd.read_csv(args.voxtral, encoding="utf-8-sig", low_memory=False)
    required = {"video", "start_sec", "end_sec", "transcript_text_plain"}
    missing = required.difference(vox.columns)
    if missing: raise ValueError(f"Voxtral input missing columns: {sorted(missing)}")
    rows = []
    for video, group in vox.groupby("video", sort=True):
        srt = args.srt_dir / f"{video}_transcript.srt"
        if not srt.exists():
            rows.append({"video": video, "status": "missing_srt", "voxtral_windows": len(group)})
            continue
        cues = parse_srt(srt); totals = [0, 0, 0, 0, 0]
        for _, row in group.iterrows():
            start, end = float(row.start_sec), float(row.end_sec)
            ref_text = " ".join(text for a, b, text in cues if a < end and b > start)
            ref, hyp = tokens(ref_text), tokens(row.transcript_text_plain)
            e, d, i, s = edit_distance(ref, hyp)
            totals[0] += 1; totals[1] += len(ref); totals[2] += e; totals[3] += d; totals[4] += i
        rows.append({"video": video, "status": "scored", "voxtral_windows": totals[0], "reference_words": totals[1], "errors": totals[2], "deletions": totals[3], "insertions": totals[4], "WER": totals[2] / totals[1] if totals[1] else None})
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    summary = {"voxtral_rows": len(vox), "videos": int(vox.video.nunique()), "scored_videos": int((result.status == "scored").sum()), "missing_srt_videos": int((result.status == "missing_srt").sum()), "overall_reference_words": int(result.reference_words.fillna(0).sum()), "overall_errors": int(result.errors.fillna(0).sum()), "overall_WER": float(result.errors.sum() / result.reference_words.sum()) if result.reference_words.sum() else None, "output": str(args.output.resolve())}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voxtral", type=Path, default=Path("data/amberscript_llm/voxtral_segments.csv"))
    parser.add_argument("--srt-dir", type=Path, default=Path("data/transcripts/srt_captions"))
    parser.add_argument("--output", type=Path, default=Path("output/transcript_provider_comparison/voxtral_wer_against_amberscript_srt_by_video.csv"))
    main(parser.parse_args())
