#!/usr/bin/env python
"""Audit saved VLM manifests before constructing aligned transcript datasets.

This script only reads CSV/JSON artifacts. It does not load a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MANIFEST_NAMES = {
    "pilot_manifest.csv",
    "manifest_with_binary_targets.csv",
    "cv_manifest.csv",
}


def compact_counts(series: pd.Series) -> str:
    counts = series.fillna("<NA>").astype(str).value_counts(dropna=False)
    return ";".join(f"{key}:{int(value)}" for key, value in counts.items())


def first_existing(frame: pd.DataFrame, names: list[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def artifact_exists_near(path: Path, name: str) -> bool:
    """Look in the manifest directory, its folds, and one parent directory."""
    roots = [path.parent, path.parent.parent]
    for root in roots:
        if root.exists() and any(root.rglob(name)):
            return True
    return False


def audit_one(path: Path, output_root: Path) -> dict:
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # preserve failures in the audit instead of stopping
        return {
            "path": str(path.relative_to(output_root.parent)),
            "filename": path.name,
            "read_error": f"{type(exc).__name__}: {exc}",
        }

    split_col = first_existing(frame, ["split", "Split", "data_split"])
    patient_col = first_existing(frame, ["patient_id", "patient", "Patient_ID"])
    consensus_col = first_existing(frame, ["WD_consensus", "wd_consensus"])
    soft_col = first_existing(frame, ["WD_soft", "wd_soft"])
    mean_col = first_existing(
        frame,
        ["WD_P_mean_from_raters", "WD_P_mean", "wd_p_mean", "WD_mean"],
    )

    consensus = (
        pd.to_numeric(frame[consensus_col], errors="coerce")
        if consensus_col
        else pd.Series(dtype=float)
    )
    soft = (
        pd.to_numeric(frame[soft_col], errors="coerce")
        if soft_col
        else pd.Series(dtype=float)
    )

    split_counts = compact_counts(frame[split_col]) if split_col else ""
    split_values = (
        set(frame[split_col].dropna().astype(str).str.lower().str.strip())
        if split_col
        else set()
    )
    consensus_rows = int(consensus.notna().sum()) if consensus_col else None
    soft_rows = int(soft.notna().sum()) if soft_col else None

    if soft_col:
        soft_counts = ";".join(
            f"{float(key):g}:{int(value)}"
            for key, value in soft.dropna().value_counts().sort_index().items()
        )
    else:
        soft_counts = ""

    notes: list[str] = []
    if len(frame) == 2512:
        notes.append("slide total=2512")
    if consensus_rows == 1777:
        notes.append("slide consensus=1777")
    if len(frame) == 2512 and consensus_rows == 1777:
        notes.append("MATCHES SOFT-LABEL SLIDE COHORT")
    if split_values == {"train", "val", "test"}:
        notes.append("usable fixed split")
    elif split_col:
        notes.append("does not contain all train/val/test splits")
    else:
        notes.append("no split column")
    if "smoke" in str(path).lower() or "prepare" in path.parts:
        notes.append("smoke/prepare artifact")

    return {
        "path": str(path.relative_to(output_root.parent)),
        "filename": path.name,
        "rows": len(frame),
        "patients": int(frame[patient_col].astype(str).nunique()) if patient_col else None,
        "split_counts": split_counts,
        "consensus_rows": consensus_rows,
        "consensus_negative": int((consensus == 0).sum()) if consensus_col else None,
        "consensus_positive": int((consensus == 1).sum()) if consensus_col else None,
        "soft_rows": soft_rows,
        "soft_counts": soft_counts,
        "has_wd_mean": bool(mean_col),
        "has_test_predictions_nearby": artifact_exists_near(path, "test_predictions.csv"),
        "has_final_summary_nearby": artifact_exists_near(path, "final_summary.json"),
        "notes": "; ".join(notes),
        "read_error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("output/vlm_wd_manifest_audit.csv"),
    )
    args = parser.parse_args()

    root = args.output_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Output directory does not exist: {root}")

    paths = sorted(
        path
        for path in root.rglob("*.csv")
        if path.name in MANIFEST_NAMES
    )
    if not paths:
        raise SystemExit(f"No supported manifest CSVs found below {root}")

    rows = [audit_one(path, root) for path in paths]
    report = pd.DataFrame(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False)

    display_columns = [
        "path",
        "rows",
        "patients",
        "split_counts",
        "consensus_rows",
        "soft_counts",
        "has_test_predictions_nearby",
        "notes",
    ]
    print(report[display_columns].to_string(index=False))
    print(f"\nFull audit written to: {args.report.resolve()}")

    exact = report[
        report["notes"].fillna("").str.contains("MATCHES SOFT-LABEL SLIDE COHORT")
    ]
    if not exact.empty:
        print("\nManifests matching the reported 2,512 / 1,777 cohort:")
        for value in exact["path"]:
            print(f"  {value}")


if __name__ == "__main__":
    main()
