"""Calculate Voxtral WER against Amberscript on exactly shared audio windows.

Amberscript is treated as the provisional reference. This intentionally refuses
to score Voxtral-only windows as errors; missing coverage is reported separately.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def normalize(text: object) -> list[str]:
    value = "" if text is None else str(text)
    value = re.sub(r"\[\d{1,3}:\d{2}(?:\.\d+)?\]\s*", " ", value)
    value = re.sub(r"\b(?:UNKNOWN|SPEAKER[_-]?\d+|[TP])\s*:\s*", " ", value, flags=re.I)
    value = value.lower().replace("ß", "ss")
    return re.findall(r"[\wäöüÄÖÜ]+", value, flags=re.UNICODE)


def distance(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int, int]:
    n, m = len(reference), len(hypothesis)
    table = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): table[i][0] = (i, i, 0, 0)
    for j in range(1, m + 1): table[0][j] = (j, 0, j, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                table[i][j] = table[i - 1][j - 1]
                continue
            candidates = [
                (table[i - 1][j][0] + 1, table[i - 1][j][1] + 1, table[i - 1][j][2], table[i - 1][j][3]),
                (table[i][j - 1][0] + 1, table[i][j - 1][1], table[i][j - 1][2] + 1, table[i][j - 1][3]),
                (table[i - 1][j - 1][0] + 1, table[i - 1][j - 1][1], table[i - 1][j - 1][2], table[i - 1][j - 1][3] + 1),
            ]
            table[i][j] = min(candidates, key=lambda x: x[0])
    return table[n][m]


def main(args: argparse.Namespace) -> None:
    amb = pd.read_csv(args.amberscript, encoding="utf-8-sig", low_memory=False)
    vox = pd.read_csv(args.voxtral, encoding="utf-8-sig", low_memory=False)
    text_col = args.text_column
    for frame, label in ((amb, "Amberscript"), (vox, "Voxtral")):
        for col in ("video", "start_sec", "end_sec", text_col):
            if col not in frame.columns:
                raise ValueError(f"{label} input is missing {col!r}")
        frame["window_key"] = frame["video"].astype(str) + "|" + frame["start_sec"].astype(str) + "|" + frame["end_sec"].astype(str)
    joined = amb.merge(vox, on="window_key", suffixes=("_amberscript", "_voxtral"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for video, group in joined.groupby("video_amberscript", dropna=False):
        edits = subs = dels = ins = ref_words = hyp_words = 0
        for _, row in group.iterrows():
            ref = normalize(row[f"{text_col}_amberscript"])
            hyp = normalize(row[f"{text_col}_voxtral"])
            e, d, i, s = distance(ref, hyp)
            edits += e; dels += d; ins += i; subs += s
            ref_words += len(ref); hyp_words += len(hyp)
        rows.append({"video": video, "shared_windows": len(group), "reference_words": ref_words, "hypothesis_words": hyp_words, "errors": edits, "substitutions": subs, "deletions": dels, "insertions": ins, "WER": edits / ref_words if ref_words else None})
    result = pd.DataFrame(rows).sort_values("video") if rows else pd.DataFrame(columns=["video", "shared_windows", "reference_words", "hypothesis_words", "errors", "substitutions", "deletions", "insertions", "WER"])
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    summary = {"amberscript_rows": len(amb), "voxtral_rows": len(vox), "shared_windows": len(joined), "shared_videos": int(result.video.nunique()), "coverage_warning": len(joined) == 0, "output": str(args.output.resolve())}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amberscript", type=Path, default=Path("data/amberscript_llm/amberscript_segments.csv"))
    parser.add_argument("--voxtral", type=Path, default=Path("data/amberscript_llm/voxtral_segments.csv"))
    parser.add_argument("--text-column", default="transcript_text_plain")
    parser.add_argument("--output", type=Path, default=Path("output/transcript_provider_comparison/voxtral_wer_by_video.csv"))
    main(parser.parse_args())
