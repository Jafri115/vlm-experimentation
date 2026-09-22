#!/usr/bin/env python
"""Summarize OOF predictions from the WD cumulative ordinal experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm.finetune_qwen3_8b_wd_text import cumulative_metrics


REQUIRED = {
    "WD_P_rater1", "WD_P_rater2", "WD_probability_ge_2",
    "WD_probability_ge_3", "patient_id", "outer_fold",
}


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("" if not np.isfinite(value) else f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def summarize(frame: pd.DataFrame) -> dict:
    return cumulative_metrics(
        frame["WD_P_rater1"], frame["WD_P_rater2"],
        frame["WD_probability_ge_2"], frame["WD_probability_ge_3"],
    )


def patient_bootstrap(frame: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """Resample patients, keeping all of each sampled patient's segments together."""
    rng = np.random.default_rng(seed)
    patients = frame["patient_id"].astype(str).unique()
    groups = {patient: frame[frame["patient_id"].astype(str) == patient] for patient in patients}
    keys = {
        "ge2_balanced_accuracy": "ge_2_consensus_balanced_accuracy",
        "ge3_balanced_accuracy": "ge_3_consensus_balanced_accuracy",
        "three_level_balanced_accuracy": "three_level_balanced_accuracy",
        "capped_severity_MAE": "capped_severity_mae",
        "capped_severity_Spearman": "capped_severity_spearman",
    }
    values = {name: [] for name in keys}
    for _ in range(replicates):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        boot = pd.concat([groups[patient] for patient in sampled], ignore_index=True)
        metrics = summarize(boot)
        for name, key in keys.items():
            value = metrics.get(key)
            if value is not None and np.isfinite(value):
                values[name].append(float(value))
    rows = []
    for name, observed_key in keys.items():
        observed = summarize(frame).get(observed_key)
        samples = values[name]
        rows.append({"metric": name, "observed": observed,
                     "bootstrap_replicates_used": len(samples),
                     "patient_bootstrap_95ci_low": np.quantile(samples, .025) if samples else np.nan,
                     "patient_bootstrap_95ci_high": np.quantile(samples, .975) if samples else np.nan})
    return pd.DataFrame(rows)


def main(args) -> None:
    frame = pd.read_csv(args.predictions, encoding="utf-8-sig", low_memory=False)
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction file lacks columns: {missing}")
    if frame["patient_id"].isna().any():
        raise ValueError("patient_id contains missing values")
    if "segment_uid" in frame and frame["segment_uid"].astype(str).duplicated().any():
        raise ValueError("OOF file contains duplicate segment_uid values")

    args.output.mkdir(parents=True, exist_ok=True)
    overall = summarize(frame)
    (args.output / "cumulative_ordinal_summary.json").write_text(
        json.dumps(overall, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    groups = [("overall", frame)]
    if "transcript_provider" in frame:
        groups.extend((f"provider:{name}", group) for name, group in frame.groupby("transcript_provider"))
    groups.extend((f"fold:{int(name)}", group) for name, group in frame.groupby("outer_fold"))
    for group_name, group in groups:
        metrics = summarize(group)
        rows.append({
            "group": group_name,
            "N": len(group),
            "ge2_consensus_N": metrics["ge_2_consensus_N"],
            "ge2_balanced_accuracy": metrics["ge_2_consensus_balanced_accuracy"],
            "ge2_AUROC": metrics["ge_2_consensus_auroc"],
            "ge3_consensus_N": metrics["ge_3_consensus_N"],
            "ge3_balanced_accuracy": metrics["ge_3_consensus_balanced_accuracy"],
            "ge3_AUROC": metrics["ge_3_consensus_auroc"],
            "three_level_balanced_accuracy": metrics["three_level_balanced_accuracy"],
            "capped_severity_MAE": metrics["capped_severity_mae"],
            "capped_severity_Spearman": metrics["capped_severity_spearman"],
            "monotonic_violations": metrics["monotonic_violations"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "cumulative_ordinal_metrics.csv", index=False, encoding="utf-8-sig")

    r1 = pd.to_numeric(frame["WD_P_rater1"])
    r2 = pd.to_numeric(frame["WD_P_rater2"])
    target_counts = pd.DataFrame([
        {"threshold": "WD >= 2", "both_below": int(((r1 < 2) & (r2 < 2)).sum()),
         "raters_disagree": int(((r1 >= 2) != (r2 >= 2)).sum()),
         "both_at_or_above": int(((r1 >= 2) & (r2 >= 2)).sum())},
        {"threshold": "WD >= 3", "both_below": int(((r1 < 3) & (r2 < 3)).sum()),
         "raters_disagree": int(((r1 >= 3) != (r2 >= 3)).sum()),
         "both_at_or_above": int(((r1 >= 3) & (r2 >= 3)).sum())},
    ])
    target_counts.to_csv(args.output / "cumulative_target_counts.csv", index=False, encoding="utf-8-sig")
    intervals = patient_bootstrap(frame, args.bootstrap, args.seed)
    intervals.to_csv(args.output / "patient_bootstrap_intervals.csv", index=False, encoding="utf-8-sig")

    report = f"""# WD cumulative ordinal experiment report

This report evaluates two ordered questions: **WD >= 2** and **WD >= 3**. The model is
constrained so `P(WD >= 3) <= P(WD >= 2)`. Exact scores 4 and 5 are not model targets.

## OOF evaluation

{markdown_table(table)}

## Human target distribution in evaluated rows

{markdown_table(target_counts)}

## Patient-bootstrap uncertainty

The interval resamples whole patients and therefore retains correlation among segments from
the same person. It uses {args.bootstrap:,} replicates.

{markdown_table(intervals)}

## Reading the results

- `ge2` measures separation of rating 1 from rating 2 or higher on rows where both humans
  agree about that threshold.
- `ge3` measures separation of ratings below 3 from rating 3 or higher on rows where both
  humans agree about that threshold. This is the main clear-withdrawal result.
- The three-level result maps the outputs to 1, 2, or 3+. It must not be described as exact
  prediction of the original five-level scale.
- Capped-severity MAE and Spearman compare `1 + P(WD>=2) + P(WD>=3)` with the human mean
  capped at 3.
- Provider and fold rows are diagnostics. The overall patient-disjoint OOF row is primary.
"""
    (args.output / "cumulative_ordinal_report.md").write_text(report, encoding="utf-8")

    if not args.no_plots:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for ax, cutoff, column in zip(
            axes, (2, 3), ("WD_probability_ge_2", "WD_probability_ge_3")
        ):
            human = ((r1 >= cutoff).astype(float) + (r2 >= cutoff).astype(float)) / 2
            for target, label, color in ((0.0, "both below", "#4477AA"),
                                         (0.5, "disagree", "#EEAA33"),
                                         (1.0, "both above", "#228833")):
                ax.hist(frame.loc[human == target, column], bins=np.linspace(0, 1, 21),
                        alpha=.55, label=label, color=color)
            ax.axvline(.5, color="black", linestyle="--", linewidth=1)
            ax.set(title=f"Predicted P(WD >= {cutoff})", xlabel="Probability", ylabel="Segments")
            ax.legend()
        fig.savefig(args.output / "cumulative_probability_distributions.png", dpi=180)
        plt.close(fig)

    print(json.dumps({"predictions": str(args.predictions.resolve()), "rows": len(frame),
                      "patients": int(frame["patient_id"].nunique()),
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    main(parser.parse_args())
