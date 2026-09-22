#!/usr/bin/env python
"""Audit the expanded joint WD_P cohort without loading model dependencies."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def read_manifest(fold_dir: Path) -> pd.DataFrame:
    for name in ("master_manifest.csv", "vlm_manifest.csv"):
        path = fold_dir / name
        if path.exists():
            return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    raise FileNotFoundError(f"No CSV manifest in {fold_dir}")


def main(args: argparse.Namespace) -> None:
    root = args.master_root.resolve()
    cache = args.frame_cache.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    summaries, assignments, exclusions = [], [], []
    reference_ids: set[str] | None = None
    test_patients_seen: list[str] = []

    for fold in range(1, args.folds + 1):
        frame = read_manifest(root / f"fold_{fold}")
        required = {
            "sample_id", "patient_id", "session_id", "split",
            "transcript_text", "start_sec", "end_sec",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"fold_{fold} missing columns: {missing}")

        ids = set(frame["sample_id"].astype(str))
        if len(ids) != len(frame):
            raise ValueError(f"fold_{fold} has duplicate sample_id values")
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f"fold_{fold} does not contain the same cohort IDs")

        text = frame["transcript_text"].fillna("").astype(str).str.strip()
        missing_text = text.eq("")
        missing_roles = ~text.str.contains(r"\b[TP]\s*:", regex=True)
        bad_time = (
            pd.to_numeric(frame["end_sec"], errors="coerce")
            <= pd.to_numeric(frame["start_sec"], errors="coerce")
        )
        missing_frames = frame["sample_id"].astype(str).map(
            lambda x: len(list((cache / re.sub(r"[^A-Za-z0-9_.-]+", "_", x)).glob("frame_*.jpg")))
            != args.num_frames
        )

        for reason, mask in {
            "missing_transcript": missing_text,
            "missing_speaker_labels": missing_roles,
            "invalid_segment_time": bad_time,
            "missing_or_wrong_frame_count": missing_frames,
        }.items():
            for row in frame.loc[mask, ["sample_id", "patient_id", "session_id", "split"]].itertuples(index=False):
                exclusions.append({"outer_fold": fold, "reason": reason, **row._asdict()})

        split_sets = {
            split: set(frame.loc[frame["split"] == split, "patient_id"].astype(str))
            for split in ("train", "val", "test")
        }
        if any(split_sets[a] & split_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError(f"fold_{fold} is not patient-disjoint")

        for split, patients in split_sets.items():
            for patient in sorted(patients):
                assignments.append({"outer_fold": fold, "split": split, "patient_id": patient})
        test_patients_seen.extend(split_sets["test"])
        summaries.append({
            "outer_fold": fold,
            "rows": len(frame),
            "patients": frame["patient_id"].astype(str).nunique(),
            "train_rows": int((frame["split"] == "train").sum()),
            "val_rows": int((frame["split"] == "val").sum()),
            "test_rows": int((frame["split"] == "test").sum()),
            "missing_transcript": int(missing_text.sum()),
            "missing_speaker_labels": int(missing_roles.sum()),
            "invalid_segment_time": int(bad_time.sum()),
            "missing_or_wrong_frame_count": int(missing_frames.sum()),
        })

    labels = pd.read_csv(root / "paired_master_soft.csv", encoding="utf-8-sig", low_memory=False)
    consensus = labels["WD_consensus"].notna() if "WD_consensus" in labels else pd.Series(False, index=labels.index)
    summary = {
        "experiment": "joint video-transcript fusion",
        "master_root": str(root),
        "frame_cache": str(cache),
        "cohort_rows": len(reference_ids or set()),
        "cohort_patients": labels["patient_id"].astype(str).nunique(),
        "consensus_evaluation_rows": int(consensus.sum()),
        "binary_disagreement_training_rows": int((~consensus).sum()),
        "num_frames": args.num_frames,
        "folds": summaries,
        "test_patient_coverage_exactly_once": len(test_patients_seen) == len(set(test_patients_seen)),
        "blocking_input_issues": len(exclusions),
    }
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(assignments).to_csv(output / "patient_splits.csv", index=False)
    pd.DataFrame(exclusions, columns=["outer_fold", "reason", "sample_id", "patient_id", "session_id", "split"]).to_csv(
        output / "input_exclusions.csv", index=False
    )
    print(json.dumps(summary, indent=2))
    if exclusions:
        raise SystemExit("Joint input audit found blocking rows; no rows were silently excluded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=16)
    main(parser.parse_args())
