#!/usr/bin/env python
"""Compare binary OOF WD predictions with every unordered human rating pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PAIR_ORDER = [f"{low}/{high}" for low in range(1, 6) for high in range(low, 6)]
NEGATIVE = "#4477AA"
POSITIVE = "#EE7733"
HUMAN = "#228833"


def load_predictions(path: Path, probability_column: str | None, threshold: float,
                     experiment: str | None = None, model: str | None = None) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if experiment is not None:
        if "experiment" not in frame:
            raise ValueError("--experiment was supplied but the CSV has no experiment column")
        frame = frame[frame["experiment"].astype(str) == experiment].copy()
    if model is not None:
        if "model" not in frame:
            raise ValueError("--model was supplied but the CSV has no model column")
        frame = frame[frame["model"].astype(str) == model].copy()
    if "scale" in frame:
        frame = frame[frame["scale"].astype(str).str.lower() == "binary"].copy()
    aliases = {"h1": "WD_P_rater1", "h2": "WD_P_rater2", "fold": "outer_fold"}
    frame = frame.rename(columns={old: new for old, new in aliases.items()
                                  if old in frame and new not in frame})
    if probability_column is None:
        probability_column = next(
            (column for column in ("WD_probability", "ai_probability") if column in frame), None
        )
        if probability_column is None:
            raise ValueError("Could not find WD_probability or ai_probability")
    required = {"WD_P_rater1", "WD_P_rater2", probability_column, "outer_fold"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction file lacks columns: {missing}")
    if "split" in frame:
        split = frame["split"].astype(str).str.lower()
        if (split == "test").any():
            frame = frame.loc[split == "test"].copy()
        elif split.nunique() > 1:
            raise ValueError("Prediction file contains multiple splits but no test rows")
    identifier = "segment_uid" if "segment_uid" in frame else "sample_id"
    if identifier not in frame:
        raise ValueError("Predictions require segment_uid or sample_id")
    if frame[identifier].astype(str).duplicated().any():
        raise ValueError(f"OOF predictions contain duplicate {identifier} values")
    for column in ("WD_P_rater1", "WD_P_rater2"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not frame[column].between(1, 5).all() or not np.allclose(frame[column] % 1, 0):
            raise ValueError(f"{column} must contain integer ratings from 1 through 5")
    frame["model_probability"] = pd.to_numeric(frame[probability_column], errors="raise")
    if not frame["model_probability"].between(0, 1).all():
        raise ValueError("Binary probabilities must lie between zero and one")
    frame["model_binary_prediction"] = (frame["model_probability"] >= threshold).astype(int)
    frame["human_low"] = frame[["WD_P_rater1", "WD_P_rater2"]].min(axis=1).astype(int)
    frame["human_high"] = frame[["WD_P_rater1", "WD_P_rater2"]].max(axis=1).astype(int)
    frame["human_pair"] = frame["human_low"].astype(str) + "/" + frame["human_high"].astype(str)
    frame["human_binary_soft"] = (
        (frame["WD_P_rater1"] >= 2).astype(float) +
        (frame["WD_P_rater2"] >= 2).astype(float)
    ) / 2
    frame["human_binary_status"] = frame["human_binary_soft"].map(
        {0.0: "both negative", 0.5: "human disagreement", 1.0: "both positive"}
    )
    frame["agrees_with_both_humans"] = (
        ((frame["human_binary_soft"] == 0) & (frame["model_binary_prediction"] == 0)) |
        ((frame["human_binary_soft"] == 1) & (frame["model_binary_prediction"] == 1))
    )
    if frame.empty:
        raise ValueError("No rows remain after applying experiment/model/binary filters")
    return frame, probability_column


def make_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair in PAIR_ORDER:
        group = frame[frame["human_pair"] == pair]
        low, high = (int(value) for value in pair.split("/"))
        human_target = ((low >= 2) + (high >= 2)) / 2
        negative = int((group["model_binary_prediction"] == 0).sum())
        positive = int((group["model_binary_prediction"] == 1).sum())
        rows.append({
            "human_pair": pair,
            "human_binary_target": human_target,
            "human_binary_status": {0.0: "both negative", .5: "human disagreement",
                                    1.0: "both positive"}[human_target],
            "N": len(group),
            "model_negative_count": negative,
            "model_positive_count": positive,
            "model_negative_percent": 100 * negative / len(group) if len(group) else np.nan,
            "model_positive_percent": 100 * positive / len(group) if len(group) else np.nan,
            "probability_mean": group["model_probability"].mean(),
            "probability_sd": group["model_probability"].std(),
            "probability_median": group["model_probability"].median(),
            "agreement_with_both_humans_percent": (
                100 * group["agrees_with_both_humans"].mean()
                if len(group) and human_target in {0.0, 1.0} else np.nan
            ),
        })
    return pd.DataFrame(rows)


def save_stacked_counts(summary: pd.DataFrame, output: Path, title: str) -> None:
    x = np.arange(len(summary))
    negative = summary["model_negative_count"].to_numpy()
    positive = summary["model_positive_count"].to_numpy()
    fig, axis = plt.subplots(figsize=(15, 7))
    axis.bar(x, negative, color=NEGATIVE, label="LLM predicts NO WD_P")
    axis.bar(x, positive, bottom=negative, color=POSITIVE, label="LLM predicts WD_P")
    for position, total in enumerate(summary["N"]):
        axis.text(position, total, f"{int(total):,}", ha="center", va="bottom", fontsize=8)
    axis.set_xticks(x, summary["human_pair"])
    axis.set(title=title, xlabel="Unordered pair of human WD_P ratings",
             ylabel="OOF test segments")
    axis.grid(axis="y", alpha=.2)
    axis.legend(frameon=False, ncol=2)
    axis.text(.5, -0.13,
              "Binary human rule: rating 1 = negative; rating 2 or higher = positive. "
              "A 1/2 pair is human disagreement.",
              transform=axis.transAxes, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "01_human_pair_by_binary_llm_counts.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(summary: pd.DataFrame, output: Path, title: str) -> None:
    percentages = summary[["model_negative_percent", "model_positive_percent"]].to_numpy(float)
    counts = summary[["model_negative_count", "model_positive_count"]].to_numpy(int)
    shown = np.nan_to_num(percentages, nan=0.0)
    fig, axis = plt.subplots(figsize=(7.5, 10))
    image = axis.imshow(shown, cmap="YlOrBr", vmin=0, vmax=100, aspect="auto")
    for row in range(len(summary)):
        for column in range(2):
            label = "—" if summary.iloc[row]["N"] == 0 else f"{counts[row, column]}\n{shown[row, column]:.1f}%"
            axis.text(column, row, label, ha="center", va="center", fontsize=8,
                      color="white" if shown[row, column] > 62 else "#17324D")
    axis.set_xticks((0, 1), ("Predict negative", "Predict positive"))
    axis.set_yticks(range(len(summary)), summary["human_pair"])
    axis.set(title=title, xlabel="Binary LLM decision", ylabel="Human rating pair")
    fig.colorbar(image, ax=axis, label="Percentage within human-pair row")
    fig.tight_layout()
    fig.savefig(output / "02_human_pair_by_binary_llm_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_probability_plot(frame: pd.DataFrame, summary: pd.DataFrame, output: Path,
                          title: str, threshold: float) -> None:
    observed = summary[summary["N"] > 0]
    pairs = observed["human_pair"].tolist()
    distributions = [frame.loc[frame["human_pair"] == pair, "model_probability"].to_numpy()
                     for pair in pairs]
    fig, axis = plt.subplots(figsize=(15, 7))
    boxplot_options = {
        "patch_artist": True,
        "showfliers": False,
        "medianprops": {"color": "#172B4D", "linewidth": 1.5},
    }
    try:
        # Matplotlib >=3.9 renamed labels to tick_labels.
        plot = axis.boxplot(distributions, tick_labels=pairs, **boxplot_options)
    except TypeError:
        plot = axis.boxplot(distributions, labels=pairs, **boxplot_options)
    for patch in plot["boxes"]:
        patch.set(facecolor="#66CCEE", alpha=.7)
    human_targets = observed["human_binary_target"].to_numpy(float)
    axis.scatter(np.arange(1, len(pairs) + 1), human_targets, marker="D", s=38,
                 color=HUMAN, label="Human binary target (0, 0.5, or 1)", zorder=3)
    axis.axhline(threshold, color="#CC3311", linestyle="--", linewidth=1.3,
                 label=f"Decision threshold = {threshold:g}")
    axis.set_ylim(-.05, 1.05)
    axis.set(title=title, xlabel="Unordered pair of human WD_P ratings",
             ylabel="Predicted probability of WD_P")
    axis.grid(axis="y", alpha=.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "03_binary_probability_by_human_pair.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(args) -> None:
    frame, probability_column = load_predictions(
        args.predictions, args.probability_column, args.threshold, args.experiment, args.model
    )
    args.output.mkdir(parents=True, exist_ok=True)
    summary = make_summary(frame)
    frame.to_csv(args.output / "oof_test_predictions_with_human_pairs.csv", index=False,
                 encoding="utf-8-sig")
    summary.to_csv(args.output / "human_pair_binary_prediction_summary.csv", index=False,
                   encoding="utf-8-sig")
    save_stacked_counts(summary, args.output,
                        f"{args.model_label}: binary output for every human rating pair")
    save_heatmap(summary, args.output,
                 f"{args.model_label}: binary decisions within each human rating pair")
    save_probability_plot(frame, summary, args.output,
                          f"{args.model_label}: WD_P probability by human rating pair",
                          args.threshold)

    consensus = frame[frame["human_binary_soft"].isin((0.0, 1.0))]
    metadata = {
        "prediction_file": str(args.predictions.resolve()),
        "probability_column": probability_column,
        "experiment_filter": args.experiment,
        "model_filter": args.model,
        "model_label": args.model_label,
        "threshold": args.threshold,
        "rows": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()) if "patient_id" in frame else None,
        "folds": int(frame["outer_fold"].nunique()),
        "human_consensus_rows": int(len(consensus)),
        "human_disagreement_rows": int((frame["human_binary_soft"] == .5).sum()),
        "model_predicted_positive_percent": float(100 * frame["model_binary_prediction"].mean()),
        "agreement_with_human_consensus_percent": (
            float(100 * consensus["agrees_with_both_humans"].mean()) if len(consensus) else None
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**metadata, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Combined OOF test-prediction CSV.")
    parser.add_argument("--probability-column",
                        help="Defaults to WD_probability or ai_probability.")
    parser.add_argument("--experiment", help="Optional experiment filter for a combined audit CSV.")
    parser.add_argument("--model", help="Optional model filter for a combined audit CSV.")
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--model-label", default="Qwen3-8B binary LLM")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
