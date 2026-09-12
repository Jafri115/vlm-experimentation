#!/usr/bin/env python
"""Build canonical WD_P cohorts and identical patient-grouped folds for VLM/LLM.

The cached VLM manifests define visual eligibility. Human ratings define the
targets, and the timestamped transcript inventory defines LLM eligibility.
Only the paired cohort is assigned folds, so both modalities use the same
physical segments and patients in every train/validation/test partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from build_llm_wd_aligned_dataset import (
    load_transcripts,
    norm_id,
    norm_session,
    segment_uid,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "output/qwen3vl_wd_planning196_thr2/qwen3vl_wd_planning196_thr2"
)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(clean_json(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_table(frame: pd.DataFrame, csv_path: Path) -> None:
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    jsonl_path = csv_path.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for row in frame.to_dict("records"):
            stream.write(json.dumps(clean_json(row), ensure_ascii=False) + "\n")


def normalize_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"patient_id", "session_id", "segment_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    frame = frame.copy()
    frame["patient_id"] = frame["patient_id"].map(norm_id)
    frame["session_id"] = frame["session_id"].map(norm_session)
    frame["segment_id"] = pd.to_numeric(frame["segment_id"], errors="raise").astype(int)
    frame["segment_uid"] = [
        segment_uid(p, s, n)
        for p, s, n in zip(frame.patient_id, frame.session_id, frame.segment_id)
    ]
    if frame["segment_uid"].duplicated().any():
        raise ValueError(f"Duplicate physical segments in {path}")
    return frame


def load_frozen_targets(path: Path, threshold: float) -> dict:
    """Load the saved targets that actually defined the VLM slide cohort."""
    frame = normalize_manifest(path)
    target_columns = [
        "coder_1",
        "coder_2",
        "WD_P_rater1",
        "WD_P_rater2",
        "WD_P_mean",
        "WD_soft",
        "WD_consensus",
        "WD_binary_disagreement",
    ]
    missing = set(target_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Frozen target file lacks columns: {sorted(missing)}")
    result = {}
    for row in frame.to_dict("records"):
        result[row["segment_uid"]] = {name: row.get(name) for name in target_columns}
        result[row["segment_uid"]]["WD_hard_mean"] = int(
            float(row["WD_P_mean"]) >= threshold
        )
        result[row["segment_uid"]]["WD_absolute_rater_difference"] = abs(
            float(row["WD_P_rater1"]) - float(row["WD_P_rater2"])
        )
    return result


def attach_targets_and_transcripts(
    soft_manifest: pd.DataFrame,
    consensus_manifest: pd.DataFrame,
    targets: dict,
    transcripts: dict,
) -> pd.DataFrame:
    consensus_ids = set(consensus_manifest["segment_uid"])
    rows = []
    for item in soft_manifest.to_dict("records"):
        uid = item["segment_uid"]
        if uid not in targets:
            raise ValueError(f"No exactly-two-rater target for {uid}")
        target = targets[uid]
        transcript = transcripts.get(uid, {})
        text = str(transcript.get("transcript_text", "") or "").strip()
        llm_ready = bool(transcript.get("llm_ready")) and bool(text)
        row = {
            **item,
            **target,
            "visual_ready": True,
            "is_cached_consensus_cohort": uid in consensus_ids,
            "transcript_available": bool(text),
            "llm_ready": llm_ready,
            "paired_ready": llm_ready,
            "transcript_provider": transcript.get("transcript_provider", ""),
            "transcript_status": transcript.get("transcript_status", "NO_TRANSCRIPT_ROW"),
            "review_flags": transcript.get("review_flags", ""),
            "transcript_text": text,
            "transcript_text_plain": transcript.get("transcript_text_plain", ""),
        }
        rows.append(row)
    master = pd.DataFrame(rows)

    expected_consensus = master["WD_consensus"].notna()
    cached_consensus = master["is_cached_consensus_cohort"].astype(bool)
    if not expected_consensus.equals(cached_consensus):
        mismatch = master.loc[expected_consensus != cached_consensus, "segment_uid"].tolist()
        raise ValueError(
            "Cached consensus membership differs from reconstructed threshold-2 "
            f"targets for {len(mismatch)} rows: {mismatch[:5]}"
        )
    return master


def patient_folds(frame: pd.DataFrame, n_folds: int, seed: int) -> dict[str, int]:
    consensus = frame[frame["WD_consensus"].notna()].copy()
    patients = sorted(consensus["patient_id"].astype(str).unique())
    if len(patients) < n_folds:
        raise ValueError(f"Only {len(patients)} paired patients for {n_folds} folds")
    y = consensus["WD_consensus"].astype(int).to_numpy()
    groups = consensus["patient_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    result = {}
    for fold, (_, test_indices) in enumerate(
        splitter.split(np.zeros((len(y), 1)), y, groups), start=1
    ):
        for patient in sorted(set(groups[test_indices])):
            if patient in result:
                raise RuntimeError(f"Patient {patient} assigned twice")
            result[patient] = fold
    # A patient can have only disagreement rows and therefore be absent from
    # the consensus-based stratifier. Place each such patient in the currently
    # smallest test fold so every paired row receives one outer test fold.
    all_patients = sorted(frame["patient_id"].astype(str).unique())
    row_counts = frame["patient_id"].astype(str).value_counts().to_dict()
    fold_loads = {
        fold: sum(row_counts[patient] for patient, value in result.items() if value == fold)
        for fold in range(1, n_folds + 1)
    }
    for patient in sorted(set(all_patients) - set(result)):
        chosen = min(fold_loads, key=lambda fold: (fold_loads[fold], fold))
        result[patient] = chosen
        fold_loads[chosen] += int(row_counts[patient])
    return result


def validation_patients(
    frame: pd.DataFrame,
    test_patients: list[str],
    inner_folds: int,
    seed: int,
    outer_fold: int,
) -> list[str]:
    pool = frame[
        ~frame["patient_id"].astype(str).isin(test_patients)
        & frame["WD_consensus"].notna()
    ].copy()
    groups = pool["patient_id"].astype(str).to_numpy()
    y = pool["WD_consensus"].astype(int).to_numpy()
    n_splits = min(inner_folds, len(set(groups)))
    if n_splits < 2:
        raise ValueError("Need at least two non-test patients for validation")
    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + 1000 + outer_fold,
        )
        splits = list(splitter.split(np.zeros((len(y), 1)), y, groups))
    except ValueError:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(np.zeros((len(y), 1)), y, groups))
    _, val_indices = splits[(outer_fold - 1) % len(splits)]
    return sorted(set(groups[val_indices]))


def describe(frame: pd.DataFrame) -> dict:
    consensus = frame[frame["WD_consensus"].notna()]
    return {
        "rows": len(frame),
        "patients": int(frame["patient_id"].astype(str).nunique()),
        "providers": dict(Counter(frame["transcript_provider"].astype(str))),
        "soft_negative": int((frame["WD_soft"] == 0).sum()),
        "soft_disagreement": int((frame["WD_soft"] == 0.5).sum()),
        "soft_positive": int((frame["WD_soft"] == 1).sum()),
        "consensus_rows": len(consensus),
        "consensus_negative": int((consensus["WD_consensus"] == 0).sum()),
        "consensus_positive": int((consensus["WD_consensus"] == 1).sum()),
    }


def main(args) -> None:
    soft = normalize_manifest(args.soft_manifest)
    consensus = normalize_manifest(args.consensus_manifest)
    if len(soft) != 2512 or len(consensus) != 1777:
        raise ValueError(
            f"Expected slide cohorts 2512/1777, found {len(soft)}/{len(consensus)}"
        )
    if not set(consensus.segment_uid).issubset(set(soft.segment_uid)):
        raise ValueError("Consensus cohort is not a subset of the soft cohort")

    targets = load_frozen_targets(args.frozen_targets, args.positive_threshold)
    transcripts = load_transcripts(args.transcripts)
    master = attach_targets_and_transcripts(soft, consensus, targets, transcripts)
    observed = describe(master)
    expected = {
        "soft_negative": 743,
        "soft_disagreement": 735,
        "soft_positive": 1034,
        "consensus_rows": 1777,
    }
    for key, value in expected.items():
        if observed[key] != value:
            raise ValueError(f"Slide check failed for {key}: {observed[key]} != {value}")

    args.output.mkdir(parents=True, exist_ok=True)
    write_table(master, args.output / "visual_master_soft_2512.csv")
    write_table(
        master[master["WD_consensus"].notna()].copy(),
        args.output / "visual_master_consensus_1777.csv",
    )
    frozen_long_rows = []
    for row in master.to_dict("records"):
        common = {
            "video": row.get("video", ""),
            "patient_id": row["patient_id"],
            # Keep the normalized numeric session form used in the generated
            # master manifests so the legacy VLM trainer's exact key merge
            # succeeds after pandas type inference.
            "session_id": row["session_id"],
            "segment_id": row["segment_id"],
            "segment_start": row.get("segment_start", ""),
            "segment_end": row.get("segment_end", ""),
        }
        frozen_long_rows.extend(
            [
                {**common, "coder": row["coder_1"], "WD_P": row["WD_P_rater1"]},
                {**common, "coder": row["coder_2"], "WD_P": row["WD_P_rater2"]},
            ]
        )
    pd.DataFrame(frozen_long_rows).to_csv(
        args.output / "frozen_rater_labels_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired = master[master["paired_ready"]].copy().reset_index(drop=True)
    paired_consensus = paired[paired["WD_consensus"].notna()].copy()
    if paired.empty or paired_consensus.empty:
        raise ValueError("No paired transcript-ready rows")
    write_table(paired, args.output / "paired_master_soft.csv")
    write_table(paired_consensus, args.output / "paired_master_consensus.csv")

    assignments = patient_folds(paired, args.cv_folds, args.seed)
    assignment_rows = []
    for patient, fold in sorted(assignments.items()):
        sub = paired[paired["patient_id"].astype(str) == patient]
        con = sub[sub["WD_consensus"].notna()]
        assignment_rows.append(
            {
                "patient_id": patient,
                "outer_fold": fold,
                "soft_rows": len(sub),
                "consensus_rows": len(con),
                "consensus_negative": int((con["WD_consensus"] == 0).sum()),
                "consensus_positive": int((con["WD_consensus"] == 1).sum()),
            }
        )
    pd.DataFrame(assignment_rows).to_csv(
        args.output / "paired_cv_patient_assignments.csv", index=False
    )

    fold_summaries = []
    # The legacy VLM trainer rebuilds these target columns from a long-form
    # ratings CSV. Supplying a manifest that already has them creates pandas
    # _x/_y suffixes and breaks that merge. The LLM manifest retains them.
    vlm_target_columns = {
        "coder_1", "coder_2", "WD_P_rater1", "WD_P_rater2", "WD_P_mean",
        "WD_hard_mean", "WD_soft", "WD_consensus", "WD_binary_disagreement",
        "WD_absolute_rater_difference",
    }
    all_patients = sorted(paired["patient_id"].astype(str).unique())
    for fold in range(1, args.cv_folds + 1):
        test = sorted(patient for patient, value in assignments.items() if value == fold)
        val = validation_patients(paired, test, args.inner_folds, args.seed, fold)
        train = sorted(set(all_patients) - set(test) - set(val))
        fold_frame = paired.copy()
        fold_frame["split"] = "train"
        patient_values = fold_frame["patient_id"].astype(str)
        fold_frame.loc[patient_values.isin(val), "split"] = "val"
        fold_frame.loc[patient_values.isin(test), "split"] = "test"
        fold_dir = args.output / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_table(fold_frame, fold_dir / "master_manifest.csv")
        vlm_manifest = fold_frame.drop(
            columns=[column for column in vlm_target_columns if column in fold_frame.columns]
        )
        vlm_manifest.to_csv(
            fold_dir / "vlm_manifest.csv", index=False, encoding="utf-8-sig"
        )
        write_table(
            fold_frame[fold_frame["WD_consensus"].notna()].copy(),
            fold_dir / "consensus_manifest.csv",
        )
        split_summary = {}
        for name in ("train", "val", "test"):
            subset = fold_frame[fold_frame["split"] == name]
            split_summary[name] = describe(subset)
        split_info = {
            "outer_fold": fold,
            "train_patients": train,
            "val_patients": val,
            "test_patients": test,
            "counts": split_summary,
        }
        write_json(fold_dir / "patient_split.json", split_info)
        fold_summaries.append(split_info)

    summary = {
        "purpose": "identical physical-segment cohorts and folds for VLM/LLM WD_P",
        "positive_rule": f"each rater WD_P >= {args.positive_threshold:g}",
        "visual_master": describe(master),
        "paired_master": describe(paired),
        "excluded_from_paired": {
            "rows": int((~master["paired_ready"]).sum()),
            "status_counts": dict(
                Counter(master.loc[~master["paired_ready"], "transcript_status"].astype(str))
            ),
        },
        "cv": {"outer_folds": args.cv_folds, "inner_folds": args.inner_folds, "seed": args.seed},
        "folds": fold_summaries,
        "sources": {
            "soft_manifest": str(args.soft_manifest.resolve()),
            "soft_manifest_sha256": sha256(args.soft_manifest),
            "consensus_manifest": str(args.consensus_manifest.resolve()),
            "consensus_manifest_sha256": sha256(args.consensus_manifest),
            "transcripts": str(args.transcripts.resolve()),
            "transcripts_sha256": sha256(args.transcripts),
            "frozen_targets": str(args.frozen_targets.resolve()),
            "frozen_targets_sha256": sha256(args.frozen_targets),
        },
    }
    write_json(args.output / "master_cohort_summary.json", summary)
    print(json.dumps(clean_json(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--soft-manifest",
        type=Path,
        default=DEFAULT_SOURCE / "cv_manifest_soft_all_cached.csv",
    )
    parser.add_argument(
        "--consensus-manifest",
        type=Path,
        default=DEFAULT_SOURCE / "cv_manifest_consensus_cached.csv",
    )
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=ROOT / "data/amberscript_llm/llm_segments_all.jsonl",
    )
    parser.add_argument(
        "--frozen-targets",
        type=Path,
        default=DEFAULT_SOURCE / "full_double_rated_targets.csv",
        help="Saved two-rater targets used when the VLM cohort was created.",
    )
    parser.add_argument("--positive-threshold", type=float, default=2.0)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output/wd_multimodal_master"
    )
    main(parser.parse_args())
