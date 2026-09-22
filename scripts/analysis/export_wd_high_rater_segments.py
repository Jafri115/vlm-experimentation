"""Export transcripts where BOTH human WD_P ratings are strictly above a threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def export_segments(input_path: Path, output_path: Path, threshold: float = 3,
                    split: str | None = None) -> dict:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output must differ from the input file.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Cohort file not found: {input_path}")
    if input_path.suffix.lower() == ".jsonl":
        frame = pd.read_json(input_path, lines=True, encoding="utf-8-sig", dtype=False)
    else:
        frame = pd.read_csv(input_path, encoding="utf-8-sig", dtype=str,
                            keep_default_na=False, low_memory=False)
    required = {"WD_P_rater1", "WD_P_rater2", "transcript_text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Use a master cohort manifest containing transcripts.")
    identifier = "sample_id" if "sample_id" in frame else "segment_uid"
    if identifier not in frame:
        raise ValueError("Input must contain sample_id or segment_uid.")
    ids = frame[identifier].fillna("").astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Input contains empty or duplicate segment identifiers. Use one master file or one fold manifest.")
    input_rows = len(frame)
    if split:
        if "split" not in frame:
            raise ValueError("--split requires a fold manifest with a split column.")
        frame = frame.loc[frame["split"].astype(str).str.lower().eq(split)].copy()
    ratings = frame[["WD_P_rater1", "WD_P_rater2"]].apply(pd.to_numeric, errors="coerce")
    for column in ratings:
        supplied = frame[column].fillna("").astype(str).str.strip().ne("")
        invalid = supplied & (~ratings[column].between(1, 5) | ratings[column].isna())
        if invalid.any():
            raise ValueError(f"Invalid or out-of-range ratings in {column}: {int(invalid.sum())} rows.")
    selected = frame.loc[ratings.gt(threshold).all(axis=1)].copy()
    selected["WD_P_rater1"] = ratings.loc[selected.index, "WD_P_rater1"]
    selected["WD_P_rater2"] = ratings.loc[selected.index, "WD_P_rater2"]
    selected["human_exact_agreement"] = selected.WD_P_rater1.eq(selected.WD_P_rater2)
    selected = selected.sort_values(identifier)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve every source column, including timing, provider, roles, and multiline transcripts.
    selected.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "input": str(input_path.resolve()), "output": str(output_path.resolve()),
        "rule": f"WD_P_rater1 > {threshold:g} AND WD_P_rater2 > {threshold:g}",
        "split": split, "input_rows": input_rows, "rows_after_split_filter": len(frame),
        "selected_segments": len(selected),
        "patients": int(selected.patient_id.nunique()) if "patient_id" in selected else None,
        "exact_agreement_segments": int(selected.human_exact_agreement.sum()),
        "selected_without_transcript": int(selected.transcript_text.fillna("").astype(str).str.strip().eq("").sum()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("output/wd_multimodal_master_expanded/paired_master_soft.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/wd_high_rater_segments/both_raters_above_3.csv"))
    parser.add_argument("--threshold", type=float, default=3,
                        help="Strict lower bound for EACH rating (default: 3, selecting 4/5).")
    parser.add_argument("--split", choices=["train", "val", "test"],
                        help="Optional split filter; use a specific fold master manifest.")
    args = parser.parse_args()
    print(json.dumps(export_segments(args.input, args.output, args.threshold, args.split), indent=2))
