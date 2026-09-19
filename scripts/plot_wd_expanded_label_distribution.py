"""Plot WD_P label distributions for the expanded shared LLM/VLM cohort."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#2878B5"
TEAL = "#2AA198"
GREEN = "#2EAD5F"
ORANGE = "#F0A52B"
RED = "#D9534F"
GRAY = "#7A8797"


def add_labels(axis, bars, total=None, decimals=1):
    for bar in bars:
        value = bar.get_height()
        if total:
            label = f"{int(value):,}\n({100 * value / total:.{decimals}f}%)"
        else:
            label = f"{int(value):,}"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_figure(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def resolve_cohort(master_root: Path) -> tuple[Path, str]:
    complete = master_root / "paired_master_soft.csv"
    candidate = master_root / "candidate_label_transcript_ready.csv"
    if complete.is_file():
        return complete, "final paired master"
    if candidate.is_file():
        return candidate, "label/transcript-ready candidate (same 4,325-row cohort)"
    raise FileNotFoundError(f"Neither {complete} nor {candidate} exists")


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sample_id", "patient_id", "WD_P_rater1", "WD_P_rater2", "WD_P_mean",
        "WD_soft", "WD_consensus", "WD_binary_disagreement",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Cohort is missing columns: {sorted(missing)}")
    out = frame.copy()
    out["patient_id"] = out["patient_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    for column in required - {"sample_id", "patient_id"}:
        out[column] = pd.to_numeric(out[column], errors="raise")
    if out.sample_id.isna().any() or out.sample_id.duplicated().any():
        raise ValueError("sample_id must be complete and unique")
    if not out.WD_P_rater1.between(1, 5).all() or not out.WD_P_rater2.between(1, 5).all():
        raise ValueError("Human ratings must be within 1-5")
    return out


def discover_fold_patients(master_root: Path, preparation_root: Path | None) -> dict[int, dict[str, list[str]]]:
    folds: dict[int, dict[str, list[str]]] = {}
    for fold in range(1, 6):
        manifest = master_root / f"fold_{fold}" / "master_manifest.csv"
        if manifest.is_file():
            split = pd.read_csv(manifest, usecols=["patient_id", "split"])
            split["patient_id"] = split.patient_id.astype(str).str.replace(r"\.0$", "", regex=True)
            folds[fold] = {
                name: sorted(split.loc[split.split.astype(str).str.lower() == name, "patient_id"].unique())
                for name in ("train", "val", "test")
            }
            continue
        if preparation_root:
            candidates = list(preparation_root.rglob(f"fold_{fold}/preparation.json"))
            if candidates:
                payload = json.loads(candidates[0].read_text(encoding="utf-8"))
                folds[fold] = {
                    name: [str(value) for value in payload["patients"][name]]
                    for name in ("train", "val", "test")
                }
    return folds


def long_count(rows: list[dict], group: str, label: str, count: int, denominator: int):
    rows.append(
        {
            "distribution": group,
            "label": label,
            "count": int(count),
            "denominator": int(denominator),
            "percent": 100 * count / denominator if denominator else np.nan,
        }
    )


def plot_raw_raters(frame: pd.DataFrame, output: Path):
    scores = np.arange(1, 6)
    c1 = frame.WD_P_rater1.value_counts().reindex(scores, fill_value=0)
    c2 = frame.WD_P_rater2.value_counts().reindex(scores, fill_value=0)
    fig, axis = plt.subplots(figsize=(10, 5.6))
    width = 0.36
    b1 = axis.bar(scores - width / 2, c1, width, label="Human rater 1", color=BLUE)
    b2 = axis.bar(scores + width / 2, c2, width, label="Human rater 2", color=ORANGE)
    add_labels(axis, b1, len(frame)); add_labels(axis, b2, len(frame))
    axis.set(
        title=f"Expanded cohort: raw WD_P ratings (N={len(frame):,} segments)",
        xlabel="Human WD_P score", ylabel="Segments", xticks=scores,
    )
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=.2)
    axis.set_ylim(0, max(c1.max(), c2.max()) * 1.20)
    save_figure(fig, output / "01_raw_rater_distribution.png")


def plot_mean_and_exact(frame: pd.DataFrame, output: Path):
    means = np.arange(1, 5.01, .5)
    mean_counts = frame.WD_P_mean.value_counts().reindex(means, fill_value=0)
    exact = frame[frame.WD_P_rater1 == frame.WD_P_rater2]
    exact_counts = exact.WD_P_rater1.value_counts().reindex(range(1, 6), fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    bars = axes[0].bar([str(x).rstrip("0").rstrip(".") for x in means], mean_counts, color=TEAL)
    add_labels(axes[0], bars, len(frame))
    axes[0].set(title="Mean of the two human ratings", xlabel="WD_P mean", ylabel="Segments")
    axes[0].grid(axis="y", alpha=.2)
    axes[0].set_ylim(0, mean_counts.max() * 1.22)
    bars = axes[1].bar(range(1, 6), exact_counts, color=[GRAY, BLUE, TEAL, ORANGE, RED])
    add_labels(axes[1], bars, len(exact))
    axes[1].set(
        title=f"Exact ordinal agreement only (N={len(exact):,})",
        xlabel="Score given by both raters", ylabel="Segments", xticks=range(1, 6),
    )
    axes[1].grid(axis="y", alpha=.2)
    axes[1].set_ylim(0, max(1, exact_counts.max()) * 1.22)
    save_figure(fig, output / "02_mean_and_exact_agreement_distribution.png")


def plot_training_targets(frame: pd.DataFrame, output: Path):
    consensus = frame.dropna(subset=["WD_consensus"])
    consensus_counts = consensus.WD_consensus.value_counts().reindex([0, 1], fill_value=0)
    soft_counts = frame.WD_soft.value_counts().reindex([0, .5, 1], fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    bars = axes[0].bar(["Negative\n(both <2)", "Positive\n(both >=2)"], consensus_counts, color=[BLUE, GREEN])
    add_labels(axes[0], bars, len(consensus))
    axes[0].set(
        title=f"Consensus-binary training target (N={len(consensus):,})",
        ylabel="Segments",
    )
    axes[0].grid(axis="y", alpha=.2); axes[0].set_ylim(0, consensus_counts.max() * 1.22)
    bars = axes[1].bar(
        ["0\n(both <2)", "0.5\n(raters disagree)", "1\n(both >=2)"],
        soft_counts,
        color=[BLUE, ORANGE, GREEN],
    )
    add_labels(axes[1], bars, len(frame))
    axes[1].set(title=f"Soft-binary training target (N={len(frame):,})", ylabel="Segments")
    axes[1].grid(axis="y", alpha=.2); axes[1].set_ylim(0, soft_counts.max() * 1.22)
    fig.suptitle("The soft target preserves binary rater disagreement; it is not a soft 1-5 score", fontsize=13)
    save_figure(fig, output / "03_consensus_and_soft_training_targets.png")


def plot_heatmap(frame: pd.DataFrame, output: Path):
    matrix = pd.crosstab(frame.WD_P_rater1, frame.WD_P_rater2).reindex(index=range(1, 6), columns=range(1, 6), fill_value=0)
    fig, axis = plt.subplots(figsize=(7.2, 6.3))
    image = axis.imshow(matrix.to_numpy(), cmap="Blues")
    threshold = matrix.to_numpy().max() / 2
    for i in range(5):
        for j in range(5):
            value = int(matrix.iloc[i, j])
            axis.text(j, i, f"{value:,}", ha="center", va="center", color="white" if value > threshold else "#16324F")
    axis.set(
        title="Joint human WD_P ratings",
        xlabel="Rater 2", ylabel="Rater 1",
        xticks=range(5), yticks=range(5),
        xticklabels=range(1, 6), yticklabels=range(1, 6),
    )
    fig.colorbar(image, ax=axis, label="Segments")
    save_figure(fig, output / "04_joint_rater_heatmap.png")


def unordered_pair_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Count rating pairs after treating (a,b) and (b,a) as the same pair."""
    low = np.minimum(frame.WD_P_rater1.to_numpy(int), frame.WD_P_rater2.to_numpy(int))
    high = np.maximum(frame.WD_P_rater1.to_numpy(int), frame.WD_P_rater2.to_numpy(int))
    observed = pd.DataFrame({"lower_rating": low, "higher_rating": high}).value_counts()
    rows = []
    for first in range(1, 6):
        for second in range(first, 6):
            count = int(observed.get((first, second), 0))
            rows.append(
                {
                    "rating_pair": f"{first}/{second}",
                    "lower_rating": first,
                    "higher_rating": second,
                    "exact_agreement": first == second,
                    "count": count,
                    "percent": 100 * count / len(frame),
                }
            )
    return pd.DataFrame(rows)


def plot_rating_pairs(pairs: pd.DataFrame, output: Path):
    fig, axis = plt.subplots(figsize=(14, 6.5))
    colors = [TEAL if exact else ORANGE for exact in pairs.exact_agreement]
    bars = axis.bar(pairs.rating_pair, pairs["count"], color=colors)
    add_labels(axis, bars, int(pairs["count"].sum()), decimals=2)
    axis.set(
        title="Every two-rater WD_P score combination",
        xlabel="Unordered rating pair (1/2 includes both 1→2 and 2→1)",
        ylabel="Segments",
    )
    axis.grid(axis="y", alpha=.2)
    axis.set_ylim(0, max(1, pairs["count"].max()) * 1.22)
    from matplotlib.patches import Patch
    axis.legend(
        handles=[Patch(color=TEAL, label="Exact agreement"), Patch(color=ORANGE, label="Different ratings")],
        frameon=False,
        ncol=2,
    )
    save_figure(fig, output / "07_unordered_rating_pair_counts.png")


def high_score_rows(frame: pd.DataFrame) -> list[dict]:
    rules = [
        ("Mean human rating >=3", frame.WD_P_mean >= 3),
        ("Both raters >=3", (frame.WD_P_rater1 >= 3) & (frame.WD_P_rater2 >= 3)),
        ("Mean human rating >=3.5", frame.WD_P_mean >= 3.5),
        ("Either rater >=4", (frame.WD_P_rater1 >= 4) | (frame.WD_P_rater2 >= 4)),
        ("Mean human rating >=4", frame.WD_P_mean >= 4),
        ("Both raters >=4", (frame.WD_P_rater1 >= 4) & (frame.WD_P_rater2 >= 4)),
        ("Exact 4/4", (frame.WD_P_rater1 == 4) & (frame.WD_P_rater2 == 4)),
        ("Either rater =5", (frame.WD_P_rater1 == 5) | (frame.WD_P_rater2 == 5)),
        ("Exact 5/5", (frame.WD_P_rater1 == 5) & (frame.WD_P_rater2 == 5)),
    ]
    return [
        {
            "criterion": name,
            "segments": int(mask.sum()),
            "percent_of_cohort": 100 * float(mask.mean()),
            "patients": int(frame.loc[mask, "patient_id"].nunique()),
        }
        for name, mask in rules
    ]


def plot_high_scores(rows: list[dict], output: Path):
    data = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(11, 6.5))
    positions = np.arange(len(data))
    bars = axis.barh(positions, data.segments, color=[TEAL, TEAL, ORANGE, ORANGE, RED, RED, RED, "#8E44AD", "#8E44AD"])
    axis.set_yticks(positions, data.criterion)
    axis.invert_yaxis()
    axis.set_xscale("symlog", linthresh=1)
    axis.set_xlabel("Segments (symlog scale)")
    axis.set_title("High WD_P ratings are scarce in the expanded cohort")
    axis.grid(axis="x", alpha=.2)
    for bar, count, percent in zip(bars, data.segments, data.percent_of_cohort):
        axis.text(max(count, .7) * 1.07, bar.get_y() + bar.get_height()/2, f"{count:,} ({percent:.2f}%)", va="center", fontsize=9)
    save_figure(fig, output / "05_high_score_scarcity.png")


def fold_rows(frame: pd.DataFrame, folds: dict[int, dict[str, list[str]]]) -> pd.DataFrame:
    rows = []
    for fold, groups in sorted(folds.items()):
        train = frame[frame.patient_id.isin(groups["train"])]
        consensus = train.dropna(subset=["WD_consensus"])
        row = {"fold": fold, "train_rows": len(train), "train_patients": train.patient_id.nunique(),
               "consensus_train_rows": len(consensus)}
        for value, suffix in [(0, "negative"), (.5, "disagreement"), (1, "positive")]:
            row[f"soft_{suffix}"] = int((train.WD_soft == value).sum())
        row["consensus_negative"] = int((consensus.WD_consensus == 0).sum())
        row["consensus_positive"] = int((consensus.WD_consensus == 1).sum())
        row["mean_ge_3"] = int((train.WD_P_mean >= 3).sum())
        row["mean_ge_3_5"] = int((train.WD_P_mean >= 3.5).sum())
        row["mean_ge_4"] = int((train.WD_P_mean >= 4).sum())
        row["both_ge_4"] = int(((train.WD_P_rater1 >= 4) & (train.WD_P_rater2 >= 4)).sum())
        row["exact_5_5"] = int(((train.WD_P_rater1 == 5) & (train.WD_P_rater2 == 5)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_folds(folds: pd.DataFrame, output: Path):
    if folds.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    x = np.arange(len(folds)); width = .35
    negative = 100 * folds.consensus_negative / folds.consensus_train_rows
    positive = 100 * folds.consensus_positive / folds.consensus_train_rows
    axes[0].bar(x, negative, width, label="Negative", color=BLUE)
    axes[0].bar(x, positive, width, bottom=negative, label="Positive", color=GREEN)
    axes[0].set(title="Consensus-binary train distribution by fold", xlabel="Outer fold", ylabel="Percent", xticks=x, xticklabels=folds.fold)
    axes[0].legend(frameon=False, ncol=2); axes[0].set_ylim(0, 100)
    soft_neg = 100 * folds.soft_negative / folds.train_rows
    soft_dis = 100 * folds.soft_disagreement / folds.train_rows
    soft_pos = 100 * folds.soft_positive / folds.train_rows
    axes[1].bar(x, soft_neg, width, label="0", color=BLUE)
    axes[1].bar(x, soft_dis, width, bottom=soft_neg, label="0.5", color=ORANGE)
    axes[1].bar(x, soft_pos, width, bottom=soft_neg + soft_dis, label="1", color=GREEN)
    axes[1].set(title="Soft-binary train distribution by fold", xlabel="Outer fold", ylabel="Percent", xticks=x, xticklabels=folds.fold)
    axes[1].legend(frameon=False, ncol=3); axes[1].set_ylim(0, 100)
    save_figure(fig, output / "06_fold_training_distributions.png")


def markdown_table(frame: pd.DataFrame, decimals: int = 2) -> str:
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.{decimals}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main(args):
    source, source_description = resolve_cohort(args.master_root)
    frame = normalize(pd.read_csv(source, encoding="utf-8-sig", low_memory=False))
    args.output.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for score in range(1, 6):
        long_count(summary_rows, "rater_1", str(score), int((frame.WD_P_rater1 == score).sum()), len(frame))
        long_count(summary_rows, "rater_2", str(score), int((frame.WD_P_rater2 == score).sum()), len(frame))
    for value in np.arange(1, 5.01, .5):
        long_count(summary_rows, "mean_rating", str(value), int((frame.WD_P_mean == value).sum()), len(frame))
    consensus = frame.dropna(subset=["WD_consensus"])
    for value in (0, 1):
        long_count(summary_rows, "consensus_binary", str(value), int((consensus.WD_consensus == value).sum()), len(consensus))
    for value in (0, .5, 1):
        long_count(summary_rows, "soft_binary", str(value), int((frame.WD_soft == value).sum()), len(frame))
    pd.DataFrame(summary_rows).to_csv(args.output / "label_distribution_counts.csv", index=False, encoding="utf-8-sig")

    joint = pd.crosstab(frame.WD_P_rater1, frame.WD_P_rater2).reindex(index=range(1, 6), columns=range(1, 6), fill_value=0)
    joint.index.name = "rater1"; joint.columns.name = "rater2"
    joint.to_csv(args.output / "joint_rater_counts.csv", encoding="utf-8-sig")
    pairs = unordered_pair_counts(frame)
    pairs.to_csv(args.output / "unordered_rating_pair_counts.csv", index=False, encoding="utf-8-sig")
    if args.only_rating_pairs:
        plot_rating_pairs(pairs, args.output)
        print(json.dumps({
            "rows": len(frame),
            "rating_pairs": len(pairs),
            "output_csv": str((args.output / "unordered_rating_pair_counts.csv").resolve()),
            "output_plot": str((args.output / "07_unordered_rating_pair_counts.png").resolve()),
        }, indent=2))
        return
    high = pd.DataFrame(high_score_rows(frame))
    high.to_csv(args.output / "high_score_counts.csv", index=False, encoding="utf-8-sig")

    preparation_root = args.preparation_root if args.preparation_root and args.preparation_root.exists() else None
    folds = discover_fold_patients(args.master_root, preparation_root)
    per_fold = fold_rows(frame, folds)
    per_fold.to_csv(args.output / "fold_training_counts.csv", index=False, encoding="utf-8-sig")

    patient = frame.groupby("patient_id").agg(
        segments=("sample_id", "size"),
        mean_ge_3=("WD_P_mean", lambda x: int((x >= 3).sum())),
        mean_ge_3_5=("WD_P_mean", lambda x: int((x >= 3.5).sum())),
        mean_ge_4=("WD_P_mean", lambda x: int((x >= 4).sum())),
    ).reset_index()
    patient.to_csv(args.output / "patient_high_score_counts.csv", index=False, encoding="utf-8-sig")

    plot_raw_raters(frame, args.output)
    plot_mean_and_exact(frame, args.output)
    plot_training_targets(frame, args.output)
    plot_heatmap(frame, args.output)
    plot_rating_pairs(pairs, args.output)
    plot_high_scores(high.to_dict("records"), args.output)
    plot_folds(per_fold, args.output)

    exact = frame[frame.WD_P_rater1 == frame.WD_P_rater2]
    headline = pd.DataFrame([
        {"training_target": "Consensus binary (<2 vs >=2)", "N": len(consensus),
         "negative": int((consensus.WD_consensus == 0).sum()), "disagreement": 0,
         "positive": int((consensus.WD_consensus == 1).sum())},
        {"training_target": "Soft binary (0 / 0.5 / 1)", "N": len(frame),
         "negative": int((frame.WD_soft == 0).sum()), "disagreement": int((frame.WD_soft == .5).sum()),
         "positive": int((frame.WD_soft == 1).sum())},
    ])
    report = [
        "# Expanded-cohort WD_P label distribution", "",
        f"Source: `{source}` ({source_description}).", "",
        f"The shared cohort contains **{len(frame):,} segments from {frame.patient_id.nunique()} patients**.", "",
        "## Binary training targets", "", markdown_table(headline, 1), "",
        "The consensus target excludes the 1,299 rows where the raters fall on opposite sides of the binary boundary. "
        "Here, consensus means agreement on **rating 1 versus rating 2-5**; it does not mean exact agreement on the ordinal score.", "",
        "The soft target uses all rows: 0 when both raters score 1, 0.5 when only one rater scores at least 2, "
        "and 1 when both raters score at least 2. It is a soft **binary** target, not a continuous 1-5 target.", "",
        "## Are there enough high-score examples?", "", markdown_table(high, 3), "",
        f"Only **{int((frame.WD_P_mean >= 4).sum())} segments ({100*(frame.WD_P_mean >= 4).mean():.2f}%)** have a mean human rating of at least 4. "
        f"Only **{int(((frame.WD_P_rater1 >= 4) & (frame.WD_P_rater2 >= 4)).sum())}** have both raters at 4 or above. "
        f"There are **{int(((frame.WD_P_rater1 == 5) & (frame.WD_P_rater2 == 5)).sum())} exact 5/5 examples**.", "",
        "Therefore, the expanded cohort is adequate for the coarse binary consensus and soft-label experiments, "
        "but it is not adequate for learning ratings 4 and 5 as well-separated ordinal classes without additional data, "
        "target redesign, or explicit imbalance handling. Most of the apparent high end is score 3 rather than 4-5.", "",
        "## Exact ordinal agreement", "",
        f"The raters give the exact same 1-5 score on **{len(exact):,}/{len(frame):,} segments ({100*len(exact)/len(frame):.1f}%)**. "
        f"Exact 3/3 accounts for {int(((frame.WD_P_rater1==3)&(frame.WD_P_rater2==3)).sum())} segments; "
        f"exact 4/4 for {int(((frame.WD_P_rater1==4)&(frame.WD_P_rater2==4)).sum())}; exact 5/5 for none.", "",
    ]
    if not per_fold.empty:
        report += ["## Training-fold availability", "", markdown_table(per_fold), "",
                   "Counts above refer to the actual patient-disjoint training subset in each outer fold. "
                   "Upper-score availability is smaller than the full-cohort numbers because validation and test patients are excluded.", ""]
    else:
        report += ["## Training-fold availability", "",
                   "Fold manifests/preparation files were not available here, so the report shows the unique shared cohort. "
                   "Run this script on the training machine to add fold-specific train counts and plot 06.", ""]
    report += ["## Figures", "",
               "1. `01_raw_rater_distribution.png`", "2. `02_mean_and_exact_agreement_distribution.png`",
               "3. `03_consensus_and_soft_training_targets.png`", "4. `04_joint_rater_heatmap.png`",
               "5. `05_high_score_scarcity.png`", "6. `06_fold_training_distributions.png` when folds are available.", ""]
    report += ["7. `07_unordered_rating_pair_counts.png` combines reversed pairs, so 1/2 includes both 1→2 and 2→1.", ""]
    (args.output / "label_distribution_report.md").write_text("\n".join(report), encoding="utf-8")

    summary = {
        "source": str(source.resolve()), "rows": len(frame), "patients": frame.patient_id.nunique(),
        "consensus_rows": len(consensus), "soft_rows": len(frame),
        "mean_ge_3": int((frame.WD_P_mean >= 3).sum()),
        "mean_ge_3_5": int((frame.WD_P_mean >= 3.5).sum()),
        "mean_ge_4": int((frame.WD_P_mean >= 4).sum()),
        "both_raters_ge_4": int(((frame.WD_P_rater1 >= 4) & (frame.WD_P_rater2 >= 4)).sum()),
        "exact_5_5": int(((frame.WD_P_rater1 == 5) & (frame.WD_P_rater2 == 5)).sum()),
        "folds_found": sorted(folds), "output": str(args.output.resolve()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", type=Path, default=Path("output/wd_multimodal_master_expanded"))
    parser.add_argument(
        "--preparation-root", type=Path,
        default=Path("output/llm_wd_ordinal_qwen3_14b_expanded_cv"),
        help="Optional recursive root containing fold_N/preparation.json files",
    )
    parser.add_argument("--output", type=Path, default=Path("output/wd_expanded_label_distribution"))
    parser.add_argument("--only-rating-pairs", action="store_true", help="Write only the unordered pair CSV and plot")
    main(parser.parse_args())
