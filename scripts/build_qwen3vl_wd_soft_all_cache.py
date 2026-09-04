#!/usr/bin/env python
"""
Prepare the all-target WD_P manifest and frame cache for soft-label training.

Purpose
-------
The existing consensus cache contains only binary-consensus rows. Soft-label
training also needs binary-disagreement rows (WD_soft = 0.5).

This script:
1. loads the selected-video training manifest,
2. attempts to cache frames for EVERY row,
3. reuses already-cached consensus frames automatically,
4. writes a metadata-only manifest containing only successfully cached rows,
   suitable for the patient-grouped CV script,
5. saves failures for inspection.

The input manifest may already contain WD target columns; they are removed from
the exported CV manifest because the training script reconstructs targets from
the original labels CSV to avoid duplicated/suffixed target columns.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import finetune_qwen3vl_rupture_pilot as base


TARGET_COLUMNS = {
    "coder_1",
    "coder_2",
    "WD_P_rater1",
    "WD_P_rater2",
    "WD_P_mean",
    "WD_P_mean_from_raters",
    "WD_P_min",
    "WD_P_max",
    "WD_P_absolute_difference",
    "WD_absolute_rater_difference",
    "WD_binary_rater1",
    "WD_binary_rater2",
    "WD_soft",
    "WD_hard_mean",
    "WD_consensus",
    "WD_binary_disagreement",
}


def clean_json_value(value):
    if isinstance(value, dict):
        return {str(k): clean_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build/reuse patient frame cache for all WD_P soft-label rows.",
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--frame-cache", required=True)
    p.add_argument("--yunet-model", required=True)
    p.add_argument("--output-manifest", required=True)
    p.add_argument("--failures-csv", required=True)
    p.add_argument("--summary-json", required=True)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--frame-width", type=int, default=224)
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    df = pd.read_csv(manifest_path)

    required = {
        "sample_id",
        "patient_id",
        "session_id",
        "video",
        "segment_id",
        "video_path",
        "patient_side",
        "start_sec",
        "end_sec",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Manifest is missing required columns: {missing}")

    # group-CV ignores original split, but merge helper requires it.
    if "split" not in df.columns:
        df["split"] = "train"

    cropper = base.PatientCropper(
        yunet_model=Path(args.yunet_model),
        output_width=int(args.frame_width),
    )
    cache = base.FrameCache(
        cache_root=Path(args.frame_cache),
        cropper=cropper,
        num_frames=int(args.num_frames),
    )

    print("BUILDING / REUSING ALL-TARGET PATIENT FRAME CACHE")
    print("=" * 72)
    print("Input rows:", len(df))
    print("Frame cache:", args.frame_cache)
    print("Frames per segment:", args.num_frames)
    print("Frame width:", args.frame_width)
    if "WD_soft" in df.columns:
        print(
            "Input WD_soft distribution:",
            df["WD_soft"].value_counts(dropna=False).sort_index().to_dict(),
        )

    success_indices = []
    failures = []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        try:
            frames = cache.build(row)
            if frames is None or len(frames) == 0:
                raise RuntimeError("FrameCache returned no frames.")
            success_indices.append(i - 1)
        except Exception as exc:
            failures.append(
                {
                    "sample_id": getattr(row, "sample_id", ""),
                    "patient_id": getattr(row, "patient_id", ""),
                    "video": getattr(row, "video", ""),
                    "segment_id": getattr(row, "segment_id", ""),
                    "error": repr(exc),
                }
            )

        if i == 1 or i % 50 == 0 or i == len(df):
            print(
                f"[{i:4d}/{len(df)}] "
                f"success={len(success_indices)} failures={len(failures)}"
            )

    success_df = df.iloc[success_indices].copy()

    # Export metadata only; labels are reconstructed from labels_csv later.
    metadata_cols = [
        c for c in success_df.columns
        if c not in TARGET_COLUMNS
    ]
    out_df = success_df[metadata_cols].copy()

    output_manifest = Path(args.output_manifest)
    failures_csv = Path(args.failures_csv)
    summary_json = Path(args.summary_json)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    failures_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(output_manifest, index=False)
    pd.DataFrame(failures).to_csv(failures_csv, index=False)

    summary = {
        "input_manifest": str(manifest_path),
        "input_rows": int(len(df)),
        "cached_success": int(len(success_df)),
        "cached_failures": int(len(failures)),
        "output_manifest": str(output_manifest),
        "frame_cache": str(Path(args.frame_cache)),
        "num_frames": int(args.num_frames),
        "frame_width": int(args.frame_width),
    }

    if "WD_soft" in success_df.columns:
        counts = success_df["WD_soft"].value_counts(dropna=False).sort_index()
        summary["successful_WD_soft_distribution"] = {
            str(k): int(v) for k, v in counts.items()
        }
        summary["successful_consensus_rows"] = int(
            success_df["WD_consensus"].notna().sum()
        ) if "WD_consensus" in success_df.columns else None
        summary["successful_disagreement_rows"] = int(
            (success_df["WD_soft"] == 0.5).sum()
        )

    summary_json.write_text(
        json.dumps(clean_json_value(summary), indent=2),
        encoding="utf-8",
    )

    print("\nFINAL SUMMARY")
    print("=" * 72)
    print(json.dumps(clean_json_value(summary), indent=2))
    print("\nCV MANIFEST:", output_manifest)
    print("FAILURES:", failures_csv)


if __name__ == "__main__":
    main()
