"""Report matched OOF ordinal-regression results against humans and baselines."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from evaluate_wd_learning import baseline_rows, bootstrap_gain, rank_corr, read_master


def markdown(frame: pd.DataFrame, digits: int = 3) -> str:
    if frame.empty:
        return "No rows."
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---:" if pd.api.types.is_numeric_dtype(frame[c]) else "---"
                                  for c in columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("NA")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.{digits}f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_prediction(path: Path, master: pd.DataFrame) -> pd.DataFrame:
    prediction = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    key = "sample_id" if "sample_id" in prediction else "segment_uid"
    required = {key, "outer_fold", "WD_prediction"}
    missing = required - set(prediction)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    if prediction[key].isna().any() or prediction[key].duplicated().any():
        raise ValueError(f"{path}: missing or duplicate IDs")
    prediction["WD_prediction"] = pd.to_numeric(prediction.WD_prediction, errors="raise")
    if not np.isfinite(prediction.WD_prediction).all() or not prediction.WD_prediction.between(1, 5).all():
        raise ValueError(f"{path}: predictions must be finite and within 1-5")
    joined = master.merge(prediction[[key, "outer_fold", "WD_prediction"]], on=key,
                          validate="one_to_one")
    if len(joined) != len(master):
        raise ValueError(f"{path}: expected {len(master)} OOF rows, found {len(joined)}")
    if not (joined.fold == pd.to_numeric(joined.outer_fold, errors="raise")).all():
        raise ValueError(f"{path}: outer folds differ from the frozen cohort")
    return joined


def centered_rank(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    z = pd.DataFrame({"patient": frame.patient_id.to_numpy(), "target": frame.target.to_numpy(),
                      "prediction": prediction})
    z[["target", "prediction"]] -= z.groupby("patient")[["target", "prediction"]].transform("mean")
    return rank_corr(z.target, z.prediction)


def model_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    y = frame.target.to_numpy(float)
    p = np.asarray(prediction, float)
    rounded = np.clip(np.floor(p + 0.5), 1, 5).astype(int)
    same = frame.h1.to_numpy() == frame.h2.to_numpy()
    disagree = ~same
    low = np.minimum(frame.h1.to_numpy(), frame.h2.to_numpy())
    high = np.maximum(frame.h1.to_numpy(), frame.h2.to_numpy())
    kappas = [cohen_kappa_score(h, rounded, labels=[1, 2, 3, 4, 5], weights="quadratic")
              for h in (frame.h1, frame.h2)]
    return {
        "N_eval": len(frame), "MAE": np.mean(np.abs(p-y)),
        "RMSE": np.sqrt(np.mean((p-y)**2)), "Spearman": rank_corr(y, p),
        "within_patient_Spearman": centered_rank(frame, p),
        "prediction_mean": np.mean(p), "prediction_SD": np.std(p),
        "target_SD": np.std(y), "SD_ratio": np.std(p)/np.std(y),
        "within_0.5": np.mean(np.abs(p-y) <= 0.5), "within_1.0": np.mean(np.abs(p-y) <= 1.0),
        "AI_human_pairwise_MAE": np.mean((np.abs(p-frame.h1)+np.abs(p-frame.h2))/2),
        "AI_human_mean_quadratic_kappa": np.nanmean(kappas),
        "human_exact_N": int(same.sum()),
        "exact_group_rounded_match": (np.mean(rounded[same] == frame.h1.to_numpy()[same])
                                      if same.any() else np.nan),
        "exact_group_MAE": (np.mean(np.abs(p[same]-frame.h1.to_numpy()[same]))
                            if same.any() else np.nan),
        "human_disagree_N": int(disagree.sum()),
        "disagreement_between_raters": (np.mean((p[disagree] >= low[disagree]) &
                                                 (p[disagree] <= high[disagree]))
                                         if disagree.any() else np.nan),
    }


def split_counts(root: Path) -> tuple[str, str]:
    train, val = [], []
    for path in sorted(root.glob("fold_*/master_manifest.csv")):
        d = pd.read_csv(path, usecols=["split"])
        counts = d["split"].astype(str).str.lower().value_counts()
        train.append(int(counts.get("train", 0))); val.append(int(counts.get("val", 0)))
    return f"{min(train)}-{max(train)}", f"{min(val)}-{max(val)}"


def report(args) -> None:
    master = read_master(args.master_root / "paired_master_soft.csv")
    baseline, _ = baseline_rows(args.master_root, master, "regression")
    master = master.merge(baseline[["sample_id", "fold", "train_mean"]], on="sample_id",
                          validate="one_to_one")
    sources = {"Training mean": None, "Qwen3-8B ordinal": args.qwen8,
               "Qwen3-14B ordinal": args.qwen14}
    for extra in args.extra:
        name, separator, raw_path = extra.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("--extra must be NAME=PATH")
        sources[name] = Path(raw_path)
    loaded = {}
    for name, path in sources.items():
        loaded[name] = master if path is None else load_prediction(path, master)
    common = set.intersection(*(set(frame.sample_id) for frame in loaded.values()))
    if common != set(master.sample_id):
        raise ValueError("Models do not cover the full common cohort")
    train_range, val_range = split_counts(args.master_root)
    rows = []
    predictions = {}
    for name, frame in loaded.items():
        frame = frame.sort_values("sample_id").reset_index(drop=True)
        prediction = (frame.train_mean if name == "Training mean" else frame.WD_prediction).to_numpy(float)
        predictions[name] = prediction
        rows.append({"model": name, "N_train_per_fold": train_range, "N_val_per_fold": val_range,
                     **model_metrics(frame, prediction)})
    metrics = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "ordinal_regression_metrics.csv", index=False)

    human = {
        "N": len(master), "human_human_MAE": float(np.mean(np.abs(master.h1-master.h2))),
        "human_human_exact": float(np.mean(master.h1 == master.h2)),
        "human_human_within_one": float(np.mean(np.abs(master.h1-master.h2) <= 1)),
        "human_human_quadratic_kappa": float(cohen_kappa_score(
            master.h1, master.h2, labels=[1, 2, 3, 4, 5], weights="quadratic")),
    }
    pd.DataFrame([human]).to_csv(args.output / "human_reference.csv", index=False)

    comparisons = []
    rng = np.random.default_rng(args.seed)
    sorted_master = master.sort_values("sample_id").reset_index(drop=True)
    y = sorted_master.target.to_numpy(float)
    for name in predictions:
        if name == "Qwen3-14B ordinal":
            continue
        gain = np.abs(predictions[name]-y) - np.abs(predictions["Qwen3-14B ordinal"]-y)
        comparisons.append({"model": "Qwen3-14B ordinal", "baseline": name,
                            **bootstrap_gain(sorted_master, gain, args.bootstrap, rng)})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(args.output / "qwen14_paired_mae_gains.csv", index=False)

    lines = ["# Qwen3 ordinal WD_P regression", "",
             "All models use the same 20-patient, five-fold cohort. Test predictions are out of fold. "
             "The ordinal models predict probabilities for ratings 1-5 and report their expected score.", "",
             "## Main results", "", markdown(metrics[["model", "N_train_per_fold", "N_val_per_fold", "N_eval",
                 "MAE", "RMSE", "Spearman", "within_patient_Spearman", "prediction_SD", "SD_ratio"]]), "",
             "## Comparison with the two humans", "",
             markdown(metrics[["model", "AI_human_pairwise_MAE", "AI_human_mean_quadratic_kappa",
                 "human_exact_N", "exact_group_rounded_match", "exact_group_MAE",
                 "human_disagree_N", "disagreement_between_raters"]]), "",
             f"The two humans have MAE {human['human_human_MAE']:.3f}, exact agreement "
             f"{human['human_human_exact']:.3f}, within-one agreement {human['human_human_within_one']:.3f}, "
             f"and quadratic kappa {human['human_human_quadratic_kappa']:.3f}.", "",
             "AI-human agreement is descriptive: the AI was trained from these raters, so it is not an independent third human.", "",
             "## Does 14B reduce error?", "", markdown(comparison), "",
             "A positive gain means the 14B model has lower MAE. The interval resamples patients, preserving clustered segments."]
    (args.output / "ordinal_regression_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_frame = metrics[metrics.model != "Training mean"]
        axes[0].bar(plot_frame.model, plot_frame.MAE)
        axes[0].set_title("Out-of-fold MAE (lower is better)")
        axes[0].tick_params(axis="x", rotation=15)
        for i, value in enumerate(plot_frame.MAE): axes[0].text(i, value, f"{value:.3f}", ha="center", va="bottom")
        axes[1].scatter(y, predictions["Qwen3-14B ordinal"], alpha=.18, s=12)
        axes[1].plot([1, 5], [1, 5], "--", color="gray")
        axes[1].set(xlabel="Mean human rating", ylabel="Qwen3-14B expected rating",
                    xlim=(.8, 5.2), ylim=(.8, 5.2), title="Does the model cover the score range?")
        fig.tight_layout(); fig.savefig(args.output / "ordinal_regression_summary.png", dpi=180); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for label, path in [("Qwen3-8B", args.qwen8), ("Qwen3-14B", args.qwen14)]:
            histories = []
            for history_path in sorted(path.parent.glob("fold_*/training_history.csv")):
                history = pd.read_csv(history_path)
                history["fold"] = history_path.parent.name
                histories.append(history)
            if not histories:
                continue
            history = pd.concat(histories, ignore_index=True)
            for _, fold_history in history.groupby("fold"):
                axes[0].plot(fold_history.epoch, fold_history.train_loss, alpha=.15)
                axes[1].plot(fold_history.epoch, fold_history.val_mae, alpha=.15)
            mean = history.groupby("epoch", as_index=False).mean(numeric_only=True)
            axes[0].plot(mean.epoch, mean.train_loss, marker="o", linewidth=2, label=label)
            axes[1].plot(mean.epoch, mean.val_mae, marker="o", linewidth=2, label=label)
        axes[0].set(title="Ordinal cross-entropy", xlabel="Epoch", ylabel="Training loss")
        axes[1].set(title="Validation selection metric", xlabel="Epoch", ylabel="MAE")
        for axis in axes: axis.grid(alpha=.25); axis.legend()
        fig.tight_layout(); fig.savefig(args.output / "ordinal_training_validation.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", type=Path, default=Path("output/wd_multimodal_master_expanded"))
    parser.add_argument("--qwen8", type=Path,
                        default=Path("output/llm_wd_ordinal_qwen3_8b_expanded_cv/oof_predictions.csv"))
    parser.add_argument("--qwen14", type=Path,
                        default=Path("output/llm_wd_ordinal_qwen3_14b_expanded_cv/oof_predictions.csv"))
    parser.add_argument("--extra", action="append", default=[], help="Additional NAME=PATH OOF result")
    parser.add_argument("--output", type=Path, default=Path("output/wd_ordinal_regression_expanded_report"))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    report(parser.parse_args())
