#!/usr/bin/env python
"""Combine fold_N/test_predictions.csv files into one out-of-fold table."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def fold_number(path: Path) -> int:
    match = re.fullmatch(r"fold_(\d+)", path.parent.name)
    if not match:
        raise ValueError(f"Expected fold_N parent directory: {path}")
    return int(match.group(1))


def main(args) -> None:
    paths = sorted(args.fold_root.glob(f"fold_*/{args.filename}"), key=fold_number)
    if not paths:
        raise SystemExit(f"No fold_N/{args.filename} under {args.fold_root}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["outer_fold"] = fold_number(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    identifier = "segment_uid" if "segment_uid" in combined else "sample_id"
    if identifier not in combined:
        raise ValueError("Predictions need segment_uid or sample_id")
    duplicates = combined[combined[identifier].astype(str).duplicated(keep=False)]
    if not duplicates.empty:
        examples = duplicates[identifier].astype(str).unique()[:10].tolist()
        raise ValueError(f"Segments occur in multiple test folds: {examples}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(
        {
            "fold_root": str(args.fold_root.resolve()),
            "folds": len(paths),
            "rows": len(combined),
            "identifier": identifier,
            "output": str(args.output.resolve()),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--filename", default="test_predictions.csv")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
