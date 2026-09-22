#!/usr/bin/env python
"""Compare fixed-fold WD development runs using validation results only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clean(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def validation_fingerprint(path: Path) -> tuple[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    identifier = "segment_uid" if rows and "segment_uid" in rows[0] else "sample_id"
    ordered = sorted(rows, key=lambda row: row[identifier])
    ids = "\n".join(row[identifier] for row in ordered)
    targets = "\n".join(
        f"{row[identifier]}|{row.get('WD_P_rater1', '')}|{row.get('WD_P_rater2', '')}"
        for row in ordered
    )
    return (hashlib.sha256(ids.encode()).hexdigest(),
            hashlib.sha256(targets.encode()).hexdigest())


def main(args) -> None:
    run_root = args.root / "runs"
    records = []
    for summary_path in sorted(run_root.glob("*/final_summary.json")):
        run_dir = summary_path.parent
        summary = load_json(summary_path)
        config = load_json(run_dir / "run_config.json")
        preparation = load_json(run_dir / "preparation.json")
        validation_id_hash, validation_target_hash = validation_fingerprint(run_dir / "val_predictions.csv")
        if not summary.get("test_evaluation_skipped", False):
            continue
        metrics = summary["val_metrics"]
        records.append({
            "experiment": run_dir.name,
            "mode": summary["mode"],
            "model": config.get("model"),
            "rubric": config.get("rubric"),
            "dataset": config.get("dataset"),
            "epochs": config.get("epochs"),
            "seed": config.get("seed"),
            "pooling": config.get("pooling"),
            "patient_balanced": config.get("patient_balanced"),
            "train_n": summary["row_counts"]["train"],
            "validation_n": summary["row_counts"]["val"],
            "validation_patients": ",".join(preparation["patients"]["val"]),
            "validation_id_hash": validation_id_hash,
            "validation_target_hash": validation_target_hash,
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "spearman": metrics.get("spearman"),
            "prediction_min": metrics.get("prediction_min"),
            "prediction_max": metrics.get("prediction_max"),
            "prediction_mean": metrics.get("prediction_mean"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "auroc": metrics.get("auroc"),
            "auprc": metrics.get("auprc"),
            "brier": metrics.get("brier"),
            "capped_severity_mae": metrics.get("capped_severity_mae"),
            "capped_severity_rmse": metrics.get("capped_severity_rmse"),
            "capped_severity_spearman": metrics.get("capped_severity_spearman"),
            "ge_2_balanced_accuracy": metrics.get("ge_2_consensus_balanced_accuracy"),
            "ge_2_auroc": metrics.get("ge_2_consensus_auroc"),
            "ge_3_balanced_accuracy": metrics.get("ge_3_consensus_balanced_accuracy"),
            "ge_3_auroc": metrics.get("ge_3_consensus_auroc"),
            "three_level_balanced_accuracy": metrics.get("three_level_balanced_accuracy"),
        })
    if not records:
        raise SystemExit(f"No completed validation-only runs under {run_root}")

    baseline = next((row for row in records if row["experiment"] == args.baseline), None)
    for row in records:
        comparable = bool(baseline and row["mode"] == baseline["mode"] and
                          row["validation_id_hash"] == baseline["validation_id_hash"] and
                          row["validation_target_hash"] == baseline["validation_target_hash"])
        row["comparable_to_baseline"] = comparable
        row["mae_improvement_vs_baseline"] = (
            baseline["mae"] - row["mae"] if comparable and baseline["mae"] is not None and row["mae"] is not None
            else None
        )
        row["spearman_improvement_vs_baseline"] = (
            row["spearman"] - baseline["spearman"]
            if comparable and baseline["spearman"] is not None and row["spearman"] is not None else None
        )
        row["balanced_accuracy_improvement_vs_baseline"] = (
            row["balanced_accuracy"] - baseline["balanced_accuracy"]
            if comparable and baseline["balanced_accuracy"] is not None and row["balanced_accuracy"] is not None
            else None
        )

    args.root.mkdir(parents=True, exist_ok=True)
    csv_path = args.root / "development_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    columns = ["experiment", "mode", "train_n", "validation_n", "mae", "spearman",
               "capped_severity_mae", "capped_severity_spearman",
               "balanced_accuracy", "auroc", "ge_2_balanced_accuracy", "ge_2_auroc",
               "ge_3_balanced_accuracy", "ge_3_auroc", "three_level_balanced_accuracy",
               "prediction_min", "prediction_max",
               "mae_improvement_vs_baseline", "spearman_improvement_vs_baseline",
               "balanced_accuracy_improvement_vs_baseline", "comparable_to_baseline"]
    lines = [
        "# WD fast-development comparison",
        "",
        f"Baseline: `{args.baseline}`. All values are from validation patients; held-out test inference is disabled.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in records:
        lines.append("| " + " | ".join(clean(row.get(column)) for column in columns) + " |")
    lines += [
        "",
        "Only compare rows marked `True`. Dataset changes can alter the validation cohort; such rows require a newly generated shared split or a common-ID evaluation.",
        "",
        "Promotion gate for regression: MAE improvement >= 0.02 or Spearman improvement >= 0.05, without a narrower prediction range. Confirm promoted configurations on two more folds before the final five-fold run.",
    ]
    report_path = args.root / "development_comparison.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(records), "baseline_found": baseline is not None,
                      "csv": str(csv_path.resolve()), "report": str(report_path.resolve())}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/wd_fast_dev"))
    parser.add_argument("--baseline", default="baseline_regression")
    main(parser.parse_args())
