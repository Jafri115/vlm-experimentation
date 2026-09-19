#!/usr/bin/env python
"""Compare OOF WD regression predictions with every unordered two-rater pair."""

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
SCORE_COLORS = {1: "#4477AA", 2: "#66CCEE", 3: "#228833", 4: "#EEAA33", 5: "#CC6677"}


def find_prediction_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in frame:
            raise ValueError(f"Prediction column is missing: {requested}")
        return requested
    candidates = ("WD_prediction", "WD_P_pred", "WD_P_prediction", "prediction")
    found = [column for column in candidates if column in frame]
    if not found:
        raise ValueError(f"Could not find a prediction column; tried {list(candidates)}")
    return found[0]


def load_predictions(path: Path, prediction_column: str | None) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    column = find_prediction_column(frame, prediction_column)
    required = {"WD_P_rater1", "WD_P_rater2", column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction file lacks columns: {missing}")
    if "split" in frame:
        split = frame["split"].astype(str).str.lower()
        if (split == "test").any():
            frame = frame.loc[split == "test"].copy()
        elif split.nunique() > 1:
            raise ValueError("Prediction file has multiple splits but no test rows")
    if "outer_fold" not in frame:
        raise ValueError("Expected OOF test predictions with an outer_fold column")
    identifier = "segment_uid" if "segment_uid" in frame else "sample_id"
    if identifier not in frame:
        raise ValueError("Predictions require segment_uid or sample_id")
    if frame[identifier].astype(str).duplicated().any():
        raise ValueError(f"OOF predictions contain duplicate {identifier} values")
    for rating in ("WD_P_rater1", "WD_P_rater2"):
        frame[rating] = pd.to_numeric(frame[rating], errors="raise")
        if not frame[rating].between(1, 5).all() or not np.allclose(frame[rating] % 1, 0):
            raise ValueError(f"{rating} must contain integer ratings from 1 through 5")
    frame["model_prediction"] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame["model_prediction"]).all():
        raise ValueError("Model predictions contain non-finite values")
    frame["model_prediction_clipped"] = frame["model_prediction"].clip(1, 5)
    frame["model_prediction_rounded"] = np.floor(
        frame["model_prediction_clipped"] + 0.5
    ).astype(int).clip(1, 5)
    frame["human_low"] = frame[["WD_P_rater1", "WD_P_rater2"]].min(axis=1).astype(int)
    frame["human_high"] = frame[["WD_P_rater1", "WD_P_rater2"]].max(axis=1).astype(int)
    frame["human_pair"] = frame["human_low"].astype(str) + "/" + frame["human_high"].astype(str)
    frame["human_mean"] = (frame["WD_P_rater1"] + frame["WD_P_rater2"]) / 2
    frame["rounded_matches_either_human"] = (
        (frame["model_prediction_rounded"] == frame["WD_P_rater1"]) |
        (frame["model_prediction_rounded"] == frame["WD_P_rater2"])
    )
    frame["continuous_within_human_interval"] = (
        (frame["model_prediction_clipped"] >= frame["human_low"]) &
        (frame["model_prediction_clipped"] <= frame["human_high"])
    )
    frame["absolute_error_to_human_mean"] = (
        frame["model_prediction_clipped"] - frame["human_mean"]
    ).abs()
    return frame, column


def make_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair in PAIR_ORDER:
        group = frame[frame["human_pair"] == pair]
        low, high = (int(value) for value in pair.split("/"))
        row = {
            "human_pair": pair,
            "human_pair_mean": (low + high) / 2,
            "N": len(group),
            "model_continuous_mean": group["model_prediction_clipped"].mean(),
            "model_continuous_sd": group["model_prediction_clipped"].std(),
            "model_continuous_median": group["model_prediction_clipped"].median(),
            "MAE_to_human_mean": group["absolute_error_to_human_mean"].mean(),
            "rounded_matches_either_human_percent": 100 * group["rounded_matches_either_human"].mean(),
            "continuous_within_human_interval_percent": 100 * group["continuous_within_human_interval"].mean(),
        }
        for score in range(1, 6):
            row[f"model_rounded_{score}_count"] = int((group["model_prediction_rounded"] == score).sum())
            row[f"model_rounded_{score}_percent"] = (
                100 * row[f"model_rounded_{score}_count"] / len(group) if len(group) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def save_stacked_counts(summary: pd.DataFrame, output: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(15, 7))
    bottom = np.zeros(len(summary))
    x = np.arange(len(summary))
    for score in range(1, 6):
        values = summary[f"model_rounded_{score}_count"].to_numpy()
        axis.bar(x, values, bottom=bottom, color=SCORE_COLORS[score], label=f"LLM rounded to {score}")
        bottom += values
    for position, total in enumerate(summary["N"]):
        axis.text(position, total, f"{int(total):,}", ha="center", va="bottom", fontsize=8)
    axis.set_xticks(x, summary["human_pair"], rotation=0)
    axis.set(title=title, xlabel="Unordered pair of human WD_P ratings",
             ylabel="OOF test segments")
    axis.grid(axis="y", alpha=.2)
    axis.legend(frameon=False, ncol=5, loc="upper right")
    axis.text(.5, -0.13,
              "Example: the 1/2 bar contains all test segments where humans rated 1 and 2; "
              "colors show the rounded LLM prediction.",
              transform=axis.transAxes, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "01_human_pair_by_rounded_llm_counts.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(summary: pd.DataFrame, output: Path, title: str) -> None:
    percentages = summary[[f"model_rounded_{score}_percent" for score in range(1, 6)]].to_numpy(float)
    counts = summary[[f"model_rounded_{score}_count" for score in range(1, 6)]].to_numpy(int)
    shown = np.nan_to_num(percentages, nan=0.0)
    fig, axis = plt.subplots(figsize=(9, 10))
    image = axis.imshow(shown, cmap="YlGnBu", vmin=0, vmax=max(1, np.nanmax(percentages)))
    threshold = max(1, np.nanmax(percentages)) * .55
    for row in range(len(summary)):
        for column in range(5):
            label = "—" if summary.iloc[row]["N"] == 0 else f"{counts[row, column]}\n{shown[row, column]:.1f}%"
            axis.text(column, row, label, ha="center", va="center", fontsize=8,
                      color="white" if shown[row, column] > threshold else "#17324D")
    axis.set_xticks(range(5), range(1, 6))
    axis.set_yticks(range(len(summary)), summary["human_pair"])
    axis.set(title=title, xlabel="Rounded LLM prediction", ylabel="Human rating pair")
    fig.colorbar(image, ax=axis, label="Percentage within human-pair row")
    fig.tight_layout()
    fig.savefig(output / "02_human_pair_by_rounded_llm_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_continuous_boxplot(frame: pd.DataFrame, summary: pd.DataFrame, output: Path, title: str) -> None:
    observed = summary[summary["N"] > 0]
    pairs = observed["human_pair"].tolist()
    distributions = [frame.loc[frame["human_pair"] == pair, "model_prediction_clipped"].to_numpy()
                     for pair in pairs]
    fig, axis = plt.subplots(figsize=(15, 7))
    plot = axis.boxplot(distributions, labels=pairs, patch_artist=True, showfliers=False,
                        medianprops={"color": "#172B4D", "linewidth": 1.5})
    for patch in plot["boxes"]:
        patch.set(facecolor="#66CCEE", alpha=.7)
    human_means = [sum(int(value) for value in pair.split("/")) / 2 for pair in pairs]
    axis.scatter(np.arange(1, len(pairs) + 1), human_means, marker="D", s=38,
                 color="#CC3311", label="Mean of two human ratings", zorder=3)
    axis.set_ylim(.8, 5.2)
    axis.set_yticks(range(1, 6))
    axis.set(title=title, xlabel="Unordered pair of human WD_P ratings",
             ylabel="Continuous LLM regression prediction")
    axis.grid(axis="y", alpha=.25)
    axis.legend(handles=[Patch(facecolor="#66CCEE", alpha=.7, label="LLM prediction distribution"),
                         plt.Line2D([], [], marker="D", linestyle="", color="#CC3311",
                                    label="Mean of two human ratings")], frameon=False)
    fig.tight_layout()
    fig.savefig(output / "03_continuous_llm_predictions_by_human_pair.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(args) -> None:
    frame, prediction_column = load_predictions(args.predictions, args.prediction_column)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = make_summary(frame)
    frame.to_csv(args.output / "oof_test_predictions_with_human_pairs.csv", index=False,
                 encoding="utf-8-sig")
    summary.to_csv(args.output / "human_pair_llm_prediction_summary.csv", index=False,
                   encoding="utf-8-sig")

    name = args.model_label
    save_stacked_counts(summary, args.output,
                        f"{name}: rounded regression output for every human rating pair")
    save_heatmap(summary, args.output,
                 f"{name}: rounded regression output within each human rating pair")
    save_continuous_boxplot(frame, summary, args.output,
                            f"{name}: continuous OOF predictions by human rating pair")

    metadata = {
        "prediction_file": str(args.predictions.resolve()),
        "prediction_column": prediction_column,
        "model_label": name,
        "rows": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()) if "patient_id" in frame else None,
        "folds": int(frame["outer_fold"].nunique()),
        "rounding_rule": "clip to [1,5], then floor(value + 0.5)",
        "overall_MAE_to_human_mean": float(frame["absolute_error_to_human_mean"].mean()),
        "overall_rounded_matches_either_human_percent": float(
            100 * frame["rounded_matches_either_human"].mean()
        ),
        "overall_continuous_within_human_interval_percent": float(
            100 * frame["continuous_within_human_interval"].mean()
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**metadata, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Combined OOF test-prediction CSV.")
    parser.add_argument("--prediction-column",
                        help="Optional override; normally auto-detected as WD_prediction.")
    parser.add_argument("--model-label", default="Qwen3-8B LLM")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
