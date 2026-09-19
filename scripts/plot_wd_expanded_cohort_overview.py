#!/usr/bin/env python
"""Create presentation-ready WD_P label-distribution plots for the expanded cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAVY = "#173A5E"
BLUE = "#3D86B8"
TEAL = "#2A9D8F"
ORANGE = "#F2A62A"
GRID = "#D7E0E8"
TEXT = "#263746"


def label_bars(ax, bars, total: int, fontsize: int = 12) -> None:
    for bar in bars:
        value = int(round(bar.get_height()))
        ax.annotate(
            f"{value:,}\n({value / total:.1%})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=TEXT,
            fontweight="semibold",
        )


def plot_mean_distribution(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    bins = np.arange(1.0, 5.01, 0.5)
    counts = frame["WD_P_mean"].value_counts().reindex(bins, fill_value=0).sort_index()
    table = counts.rename_axis("mean_WD_P").reset_index(name="segments")
    table["percent"] = table["segments"] / len(frame) * 100

    fig, ax = plt.subplots(figsize=(13.333, 7.5), dpi=180)
    colors = [TEAL if score < 3 else ORANGE for score in counts.index]
    bars = ax.bar([f"{x:.1f}" for x in counts.index], counts.values, color=colors, width=0.72)
    label_bars(ax, bars, len(frame), fontsize=11)
    ax.set_title("Distribution of mean human WD_P ratings", fontsize=24, color=NAVY, pad=22, weight="bold")
    ax.set_xlabel("Mean of the two human ratings", fontsize=15)
    ax.set_ylabel("Segments", fontsize=15)
    ax.grid(axis="y", alpha=0.35, color=GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=13)
    ax.text(
        0.99,
        0.97,
        "Low ratings dominate; scores ≥4 are exceptionally rare",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13,
        color=NAVY,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#EAF7F1", edgecolor=TEAL),
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return table


def plot_binary_consensus(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    b1 = frame["WD_P_rater1"].ge(2)
    b2 = frame["WD_P_rater2"].ge(2)
    binary_consensus = b1.eq(b2)
    counts = pd.Series(
        {
            "Binary consensus": int(binary_consensus.sum()),
            "Binary disagreement": int((~binary_consensus).sum()),
        }
    )
    exact_ordinal = frame["WD_P_rater1"].eq(frame["WD_P_rater2"])
    exact_n = int(exact_ordinal.sum())

    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=180)
    wedges, _ = ax.pie(
        counts.values,
        colors=[TEAL, ORANGE],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.36, edgecolor="white", linewidth=3),
    )
    for wedge, label, value in zip(wedges, counts.index, counts.values):
        angle = (wedge.theta1 + wedge.theta2) / 2
        x, y = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        ax.annotate(
            f"{label}\n{value:,} ({value / len(frame):.1%})",
            xy=(0.82 * x, 0.82 * y),
            xytext=(1.22 * np.sign(x), 1.05 * y),
            ha="left" if x >= 0 else "right",
            va="center",
            fontsize=14,
            color=TEXT,
            fontweight="semibold",
            arrowprops=dict(arrowstyle="-", color="#73879A", lw=1.5),
        )
    ax.text(0, 0.08, "Binary WD_P", ha="center", va="center", fontsize=18, color=NAVY, weight="bold")
    ax.text(0, -0.12, "1 vs ≥2", ha="center", va="center", fontsize=16, color=TEXT)
    ax.set_title("Binary WD_P agreement between human raters", fontsize=23, color=NAVY, pad=22, weight="bold")
    ax.text(
        0.5,
        -0.09,
        f"Exact 1–5 agreement is lower: {exact_n:,}/{len(frame):,} ({exact_n / len(frame):.1%})",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        color=NAVY,
    )
    ax.text(
        0.5,
        -0.15,
        "Binary consensus = both rate 1, or both rate ≥2",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        color="#66788A",
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    table = counts.rename_axis("agreement_group").reset_index(name="segments")
    table["percent"] = table["segments"] / len(frame) * 100
    return table


def plot_combined(frame: pd.DataFrame, mean_counts: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), dpi=180, gridspec_kw={"width_ratios": [1.55, 1]})

    ax = axes[0]
    colors = [TEAL if score < 3 else ORANGE for score in mean_counts["mean_WD_P"]]
    bars = ax.bar(
        [f"{x:.1f}" for x in mean_counts["mean_WD_P"]],
        mean_counts["segments"],
        color=colors,
        width=0.72,
    )
    label_bars(ax, bars, len(frame), fontsize=9)
    ax.set_title("Mean WD_P rating distribution", fontsize=20, color=NAVY, weight="bold", pad=18)
    ax.set_xlabel("Mean of two human ratings", fontsize=13)
    ax.set_ylabel("Segments", fontsize=13)
    ax.grid(axis="y", alpha=0.32, color=GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)

    b1 = frame["WD_P_rater1"].ge(2)
    b2 = frame["WD_P_rater2"].ge(2)
    consensus = int(b1.eq(b2).sum())
    disagreement = len(frame) - consensus
    exact = int(frame["WD_P_rater1"].eq(frame["WD_P_rater2"]).sum())

    ax = axes[1]
    ax.pie(
        [consensus, disagreement],
        labels=[
            f"Binary consensus\n{consensus:,} ({consensus / len(frame):.1%})",
            f"Binary disagreement\n{disagreement:,} ({disagreement / len(frame):.1%})",
        ],
        colors=[TEAL, ORANGE],
        startangle=90,
        counterclock=False,
        textprops=dict(fontsize=12, color=TEXT, weight="semibold"),
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=3),
        labeldistance=1.12,
    )
    ax.text(0, 0.08, "Binary WD_P", ha="center", va="center", fontsize=17, color=NAVY, weight="bold")
    ax.text(0, -0.12, "1 vs ≥2", ha="center", va="center", fontsize=15, color=TEXT)
    ax.set_title("Rater agreement at the binary boundary", fontsize=20, color=NAVY, weight="bold", pad=18)
    ax.text(
        0.5,
        -0.05,
        f"Exact ordinal agreement: {exact:,} ({exact / len(frame):.1%})",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        color=NAVY,
    )
    ax.text(
        0.5,
        -0.11,
        "Binary consensus means both 1 or both ≥2",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        color="#66788A",
    )

    fig.suptitle(
        "Expanded shared cohort: label structure",
        fontsize=25,
        color=NAVY,
        weight="bold",
        y=0.995,
    )
    fig.subplots_adjust(left=0.07, right=0.96, top=0.86, bottom=0.15, wspace=0.27)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    required = {"WD_P_rater1", "WD_P_rater2", "WD_P_mean"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame = frame.dropna(subset=list(required)).copy()
    args.output.mkdir(parents=True, exist_ok=True)

    mean_counts = plot_mean_distribution(frame, args.output / "01_mean_wd_p_distribution.png")
    binary_counts = plot_binary_consensus(frame, args.output / "02_binary_consensus_vs_disagreement.png")
    plot_combined(frame, mean_counts, args.output / "03_cohort_slide_combined.png")

    mean_counts.to_csv(args.output / "mean_wd_p_distribution.csv", index=False, encoding="utf-8-sig")
    binary_counts.to_csv(args.output / "binary_consensus_counts.csv", index=False, encoding="utf-8-sig")
    exact_n = int(frame["WD_P_rater1"].eq(frame["WD_P_rater2"]).sum())
    summary = {
        "input": str(args.input.resolve()),
        "rows": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()) if "patient_id" in frame else None,
        "binary_consensus": int(binary_counts.loc[binary_counts["agreement_group"] == "Binary consensus", "segments"].iloc[0]),
        "binary_disagreement": int(binary_counts.loc[binary_counts["agreement_group"] == "Binary disagreement", "segments"].iloc[0]),
        "exact_ordinal_agreement": exact_n,
        "exact_ordinal_agreement_percent": exact_n / len(frame) * 100,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/wd_multimodal_master_expanded/candidate_label_transcript_ready.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/wd_expanded_cohort_overview"),
    )
    main(parser.parse_args())
