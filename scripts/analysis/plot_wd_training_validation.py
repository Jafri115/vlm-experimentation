"""Create comparable train/validation plots for the WD experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_llm(path: Path, output: Path, method: str) -> None:
    frames = []
    for fold_file in sorted(path.glob("fold_*/training_history.csv")):
        frame = pd.read_csv(fold_file)
        frame["fold"] = fold_file.parent.name
        frames.append(frame)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    x = "epoch"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for _, fold in data.groupby("fold"):
        axes[0].plot(fold[x], fold["train_loss"], alpha=.25)
        if "val_loss" in fold:
            axes[0].plot(fold[x], fold["val_loss"], alpha=.25, linestyle="--")
    mean = data.groupby(x, as_index=False).mean(numeric_only=True)
    axes[0].plot(mean[x], mean["train_loss"], linewidth=2, label="train loss")
    if "val_loss" in mean:
        axes[0].plot(mean[x], mean["val_loss"], linewidth=2, linestyle="--", label="validation loss")
    axes[0].set_title(f"LLM {method}: loss")
    axes[0].set_xlabel("Epoch"); axes[0].grid(alpha=.25); axes[0].legend()
    metric_cols = [c for c in mean.columns if c.startswith("val_") and c not in {"val_loss"}]
    for col in metric_cols:
        axes[1].plot(mean[x], mean[col], marker="o", label=col.removeprefix("val_"))
    axes[1].set_title(f"LLM {method}: validation metrics")
    axes[1].set_xlabel("Epoch"); axes[1].grid(alpha=.25); axes[1].legend()
    fig.tight_layout(); fig.savefig(output / f"llm_{method}_train_validation.png", dpi=180); plt.close(fig)


def plot_vlm(path: Path, output: Path, method: str) -> None:
    frames = []
    for fold_file in sorted(path.glob("fold_*/learning_curves.csv")):
        frame = pd.read_csv(fold_file)
        frame["fold"] = fold_file.parent.name
        frames.append(frame)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    x = "optimizer_step"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for _, fold in data.groupby("fold"):
        axes[0].plot(fold[x], fold["train_loss_recent"], alpha=.25)
    mean = data.groupby(x, as_index=False).mean(numeric_only=True)
    axes[0].plot(mean[x], mean["train_loss_recent"], linewidth=2, label="train loss")
    axes[0].set_title(f"VLM {method}: loss"); axes[0].set_xlabel("Optimizer step")
    axes[0].grid(alpha=.25); axes[0].legend()
    for col in ["val_AUPRC", "val_AUROC", "val_soft_BCE"]:
        if col in mean:
            axes[1].plot(mean[x], mean[col], marker="o", label=col.removeprefix("val_"))
    axes[1].set_title(f"VLM {method}: validation metrics")
    axes[1].set_xlabel("Optimizer step"); axes[1].grid(alpha=.25); axes[1].legend()
    fig.tight_layout(); fig.savefig(output / f"vlm_{method}_train_validation.png", dpi=180); plt.close(fig)


def main(args):
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    for method in ("regression", "consensus", "soft"):
        plot_llm(args.root / f"llm_wd_{method}{args.suffix}_cv", output, method)
    for method in ("consensus", "soft"):
        plot_vlm(args.root / f"vlm_wd_{method}{args.suffix}_paired_cv", output, method)
    print(f"Plots written to {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output"))
    parser.add_argument("--output", type=Path, default=Path("output/wd_training_plots_repaired"))
    parser.add_argument("--suffix", default="", help="Output-root suffix before _cv, e.g. _repaired")
    main(parser.parse_args())
