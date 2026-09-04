#!/usr/bin/env python
"""
Patch finetune_qwen3vl_wd_only_large.py to add periodic train/validation
curve logging and PNG plots.

Run from:
C:\\Data\\Sequence_model\\VLM_experiments

It patches:
.\\scripts\\finetune_qwen3vl_wd_only_large.py

A backup is created before modification.
"""

from pathlib import Path
import shutil
import py_compile

TARGET = Path(r".\scripts\finetune_qwen3vl_wd_only_large.py")

if not TARGET.exists():
    raise FileNotFoundError(f"Target not found: {TARGET.resolve()}")

text = TARGET.read_text(encoding="utf-8")

if "def train(" not in text or "def make_parser()" not in text:
    raise RuntimeError("Unexpected target script structure.")

backup = TARGET.with_suffix(".py.bak_before_curves")
shutil.copy2(TARGET, backup)

# 1) matplotlib import
anchor = "import torch.nn.functional as F\n"
addition = """import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
"""
if "import matplotlib.pyplot as plt" not in text:
    if anchor not in text:
        raise RuntimeError("Could not find torch functional import.")
    text = text.replace(anchor, addition, 1)

# 2) helper functions before Evaluation section
marker = "# ---------------------------------------------------------------------------\n# Evaluation\n# ---------------------------------------------------------------------------\n"
helpers = r'''
# ---------------------------------------------------------------------------
# Learning-curve helpers
# ---------------------------------------------------------------------------

def save_learning_curves(history: List[dict], output_dir: Path) -> None:
    """Save CSV plus separate training-loss, validation-MAE and Spearman plots."""
    if not history:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(history)
    hist.to_csv(output_dir / "learning_curves.csv", index=False)

    train_hist = hist.dropna(subset=["train_loss_recent"])
    if not train_hist.empty:
        fig = plt.figure(figsize=(7, 4.5))
        ax = fig.add_subplot(111)
        ax.plot(
            train_hist["optimizer_step"],
            train_hist["train_loss_recent"],
            marker="o",
        )
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Recent mean training loss")
        ax.set_title("WD-only QLoRA training loss")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "training_loss_curve.png", dpi=180)
        plt.close(fig)

    val_hist = hist.dropna(subset=["val_WD_P_MAE"])
    if not val_hist.empty:
        fig = plt.figure(figsize=(7, 4.5))
        ax = fig.add_subplot(111)
        ax.plot(
            val_hist["optimizer_step"],
            val_hist["val_WD_P_MAE"],
            marker="o",
        )
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Validation WD_P MAE")
        ax.set_title("WD-only QLoRA validation MAE")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "validation_mae_curve.png", dpi=180)
        plt.close(fig)

        spear = val_hist.dropna(subset=["val_WD_P_spearman"])
        if not spear.empty:
            fig = plt.figure(figsize=(7, 4.5))
            ax = fig.add_subplot(111)
            ax.plot(
                spear["optimizer_step"],
                spear["val_WD_P_spearman"],
                marker="o",
            )
            ax.set_xlabel("Optimizer step")
            ax.set_ylabel("Validation WD_P Spearman")
            ax.set_title("WD-only QLoRA validation ranking")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(output_dir / "validation_spearman_curve.png", dpi=180)
            plt.close(fig)


@torch.no_grad()
def evaluate_curve_subset(
    split_df: pd.DataFrame,
    model,
    head,
    processor,
    frame_cache,
    device,
    args,
) -> dict:
    """
    Periodic validation on a deterministic fixed subset.
    This is diagnostic only; full validation still selects the checkpoint.
    """
    model.eval()
    head.eval()

    n = min(int(args.curve_val_examples), len(split_df))
    if n <= 0:
        return {}

    work = split_df.copy()
    positives = work[work["WD_P_mean"] >= args.positive_threshold]
    negatives = work[work["WD_P_mean"] < args.positive_threshold]

    desired_pos = min(len(positives), max(1, n // 3))
    selected = []

    if desired_pos > 0:
        selected.append(
            positives.sample(n=desired_pos, random_state=args.seed + 7001)
        )

    remaining = n - desired_pos
    if remaining > 0:
        neg_take = min(remaining, len(negatives))
        if neg_take > 0:
            selected.append(
                negatives.sample(n=neg_take, random_state=args.seed + 7002)
            )
        remaining -= neg_take

    if remaining > 0:
        already = (
            pd.concat(selected, ignore_index=False).index
            if selected else []
        )
        rest = work.drop(index=already, errors="ignore")
        if len(rest) > 0:
            selected.append(
                rest.sample(
                    n=min(remaining, len(rest)),
                    random_state=args.seed + 7003,
                )
            )

    eval_df = pd.concat(selected, ignore_index=False).sort_index()
    rows = []

    for row in eval_df.itertuples(index=False):
        try:
            frames = frame_cache.build(row)
            pred = predict_wd(
                model=model,
                head=head,
                processor=processor,
                frames=frames,
                device=device,
                pooling=args.pooling,
            )
            pred_value = float(pred[0, 0].detach().float().cpu())
            rows.append(
                {
                    "WD_P_true": float(row.WD_P_mean),
                    "WD_P_pred": pred_value,
                }
            )
        except Exception as exc:
            print(
                f"  CURVE-VAL ERROR {getattr(row, 'sample_id', '?')}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        torch.cuda.empty_cache()

    model.train()
    head.train()

    return compute_wd_metrics(rows, args.positive_threshold)

'''
if "def save_learning_curves(" not in text:
    if marker not in text:
        raise RuntimeError("Could not find Evaluation section marker.")
    text = text.replace(marker, helpers + "\n" + marker, 1)

# 3) initialize curve history
anchor = """    best_lora_state = None
    best_head_state = None

    optimizer_step = 0
"""
replacement = """    best_lora_state = None
    best_head_state = None

    curve_history = []
    recent_curve_losses = []

    optimizer_step = 0
"""
if "curve_history = []" not in text:
    if anchor not in text:
        raise RuntimeError("Could not find curve-history insertion point.")
    text = text.replace(anchor, replacement, 1)

# 4) collect train losses
anchor = """                running_loss += float(
                    loss.detach().cpu()
                )

                should_step = (
"""
replacement = """                detached_loss = float(
                    loss.detach().cpu()
                )
                running_loss += detached_loss
                recent_curve_losses.append(detached_loss)

                should_step = (
"""
if "recent_curve_losses.append(detached_loss)" not in text:
    if anchor not in text:
        raise RuntimeError("Could not find running-loss block.")
    text = text.replace(anchor, replacement, 1)

# 5) periodic validation after optimizer update
anchor = """                    optimizer_step += 1
                    accum_counter = 0

                pred_value = float(
"""
replacement = """                    optimizer_step += 1
                    accum_counter = 0

                    if (
                        args.eval_every_steps > 0
                        and optimizer_step % args.eval_every_steps == 0
                    ):
                        recent_train_loss = (
                            float(np.mean(recent_curve_losses))
                            if recent_curve_losses
                            else float("nan")
                        )

                        print(
                            "\\nPERIODIC CURVE VALIDATION "
                            f"@ optimizer step {optimizer_step}",
                            flush=True,
                        )

                        curve_metrics = evaluate_curve_subset(
                            split_df=val_df,
                            model=model,
                            head=head,
                            processor=processor,
                            frame_cache=frame_cache,
                            device=device,
                            args=args,
                        )

                        curve_history.append(
                            {
                                "optimizer_step": optimizer_step,
                                "epoch": epoch,
                                "train_loss_recent": recent_train_loss,
                                "val_WD_P_MAE": curve_metrics.get(
                                    "WD_P_MAE", float("nan")
                                ),
                                "val_WD_P_spearman": curve_metrics.get(
                                    "WD_P_spearman", float("nan")
                                ),
                                "val_WD_P_pred_std": curve_metrics.get(
                                    "WD_P_pred_std", float("nan")
                                ),
                            }
                        )

                        save_learning_curves(curve_history, output_dir)

                        print(
                            "  Curve point: "
                            f"train_loss={recent_train_loss:.4f} | "
                            f"val_MAE={curve_metrics.get('WD_P_MAE', float('nan')):.4f} | "
                            f"val_Spearman={curve_metrics.get('WD_P_spearman', float('nan')):.4f}",
                            flush=True,
                        )

                        recent_curve_losses = []

                pred_value = float(
"""
if "PERIODIC CURVE VALIDATION" not in text:
    if anchor not in text:
        raise RuntimeError("Could not find optimizer-step block.")
    text = text.replace(anchor, replacement, 1)

# 6) save final curve files before restoring best checkpoint
anchor = """    print(
        "\\nRESTORING BEST WD VALIDATION CHECKPOINT"
    )
"""
if anchor not in text:
    # support slightly different wording in a local version
    anchor = """    print(
        "\\nRESTORING BEST WD CHECKPOINT"
    )
"""
replacement = """    save_learning_curves(
        curve_history,
        output_dir,
    )

""" + anchor
if "save_learning_curves(\n        curve_history" not in text:
    if anchor not in text:
        raise RuntimeError("Could not find final curve-save insertion point.")
    text = text.replace(anchor, replacement, 1)

# 7) CLI args
anchor = """    p.add_argument(
        "--max-test-examples",
        type=int,
        default=0,
    )

    p.add_argument(
        "--seed",
"""
replacement = """    p.add_argument(
        "--max-test-examples",
        type=int,
        default=0,
    )

    p.add_argument(
        "--eval-every-steps",
        type=int,
        default=20,
        help=(
            "Run fixed-subset validation every N optimizer steps for "
            "learning curves. 0 disables periodic validation."
        ),
    )
    p.add_argument(
        "--curve-val-examples",
        type=int,
        default=30,
        help="Number of fixed validation examples per curve point.",
    )

    p.add_argument(
        "--seed",
"""
if '"--eval-every-steps"' not in text:
    if anchor not in text:
        raise RuntimeError("Could not find CLI insertion point.")
    text = text.replace(anchor, replacement, 1)

TARGET.write_text(text, encoding="utf-8")
py_compile.compile(str(TARGET), doraise=True)

required = [
    "def save_learning_curves(",
    "def evaluate_curve_subset(",
    '"--eval-every-steps"',
    '"--curve-val-examples"',
    "PERIODIC CURVE VALIDATION",
    "learning_curves.csv",
    "training_loss_curve.png",
    "validation_mae_curve.png",
]
missing = [x for x in required if x not in text]
if missing:
    raise RuntimeError(f"Patch validation failed: {missing}")

print("PATCH SUCCESS")
print(f"Updated: {TARGET.resolve()}")
print(f"Backup : {backup.resolve()}")
print("Syntax check: PASSED")
print("Adds periodic train/validation curve logging + PNG plots.")   