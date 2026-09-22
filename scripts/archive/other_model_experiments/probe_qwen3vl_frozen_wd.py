#!/usr/bin/env python
"""
Frozen Qwen3-VL representation probe for WD_P.

Purpose
-------
This is a diagnostic experiment, not another QLoRA run.

It asks:
    "Do frozen pretrained Qwen3-VL representations already contain
     information that predicts patient withdrawal salience (WD_P)?"

Pipeline
--------
Existing patient-disjoint manifest
    -> 16 chronological patient frames / labelled minute
    -> frozen Qwen3-VL-8B
    -> mean_all hidden representation
    -> standardization
    -> Ridge regression
    -> WD_P prediction

The script:
1. Reuses EXACTLY the train/val/test split in the existing manifest.
2. Extracts and caches one frozen VLM embedding per 1-minute segment.
3. Tunes Ridge alpha on validation MAE only.
4. Evaluates the selected Ridge model once on the held-out test patients.
5. Compares against a constant training-mean baseline.
6. Reports prediction spread and patient-wise correlations.
7. Saves diagnostic plots and CSVs.

Place this file in:
    C:\\Data\\Sequence_model\\VLM_experiments\\scripts\\

It imports preprocessing helpers from:
    finetune_qwen3vl_rupture_pilot.py
in the same scripts folder.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from PIL import Image

import torch

from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    average_precision_score,
    roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError as exc:
    raise ImportError(
        "Your transformers installation does not expose "
        "Qwen3VLForConditionalGeneration. Run this in the same "
        ".venv_molmo2 environment that successfully ran Qwen3-VL."
    ) from exc

import finetune_qwen3vl_rupture_pilot as base


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

PROBE_PROMPT = """The images are chronological frames sampled across one minute of a psychotherapy session.
Focus only on the visible patient/person being observed.

Represent the visible patient behavior in a way that preserves information potentially relevant to patient withdrawal salience.

Use only visible patient behavior. Do not use audio or transcript.
"""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(x):
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x

    clean = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            clean[k] = {
                kk: convert(vv)
                for kk, vv in v.items()
            }
        else:
            clean[k] = convert(v)

    path.write_text(
        json.dumps(clean, indent=2),
        encoding="utf-8",
    )


def safe_spearman(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 3:
        return float("nan")

    if np.allclose(y_true, y_true[0]):
        return float("nan")

    if np.allclose(y_pred, y_pred[0]):
        return float("nan")

    try:
        return float(
            spearmanr(y_true, y_pred).statistic
        )
    except Exception:
        return float("nan")


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 3.0,
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    result = {
        "n": int(len(y_true)),
        "MAE": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "Spearman": safe_spearman(
            y_true,
            y_pred,
        ),
        "true_mean": float(np.mean(y_true)),
        "true_std": float(np.std(y_true)),
        "true_min": float(np.min(y_true)),
        "true_max": float(np.max(y_true)),
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred)),
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
    }

    true_binary = (
        y_true >= threshold
    ).astype(int)

    pred_binary = (
        y_pred >= threshold
    ).astype(int)

    tp = int(
        np.sum(
            (true_binary == 1)
            & (pred_binary == 1)
        )
    )
    fp = int(
        np.sum(
            (true_binary == 0)
            & (pred_binary == 1)
        )
    )
    fn = int(
        np.sum(
            (true_binary == 1)
            & (pred_binary == 0)
        )
    )
    tn = int(
        np.sum(
            (true_binary == 0)
            & (pred_binary == 0)
        )
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    result.update(
        {
            "WD_ge3_TP": tp,
            "WD_ge3_FP": fp,
            "WD_ge3_FN": fn,
            "WD_ge3_TN": tn,
            "WD_ge3_precision": precision,
            "WD_ge3_recall": recall,
            "WD_ge3_f1": f1,
        }
    )

    if len(np.unique(true_binary)) == 2:
        try:
            result["WD_ge3_AUPRC"] = float(
                average_precision_score(
                    true_binary,
                    y_pred,
                )
            )
            result["WD_ge3_AUROC"] = float(
                roc_auc_score(
                    true_binary,
                    y_pred,
                )
            )
        except Exception:
            pass

    return result


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    print("=" * 72)

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:24s}: {v:.6f}")
        else:
            print(f"{k:24s}: {v}")


# ---------------------------------------------------------------------------
# Frozen Qwen loading
# ---------------------------------------------------------------------------

def load_frozen_qwen(args):
    print("\nLOADING FROZEN QWEN3-VL")
    print("=" * 72)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for this experiment."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    quant_config = None

    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=quant_config,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )

    processor = AutoProcessor.from_pretrained(
        args.model_id,
    )

    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Trainable Qwen parameters:", trainable)
    print("Qwen is fully frozen.")

    return model, processor


# ---------------------------------------------------------------------------
# VLM representation extraction
# ---------------------------------------------------------------------------

def prepare_inputs(
    processor,
    frames: Sequence[Image.Image],
    device: torch.device,
):
    content = [
        {
            "type": "image",
            "image": frame,
        }
        for frame in frames
    ]

    content.append(
        {
            "type": "text",
            "text": PROBE_PROMPT,
        }
    )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    moved = {}

    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue

        if key in {
            "pixel_values",
            "pixel_values_videos",
        }:
            moved[key] = value.to(
                device=device,
                dtype=torch.bfloat16,
            )
        else:
            moved[key] = value.to(
                device=device
            )

    return moved


def mean_all_pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if attention_mask is None:
        valid = torch.ones(
            hidden.shape[:2],
            device=hidden.device,
            dtype=torch.bool,
        )
    else:
        valid = attention_mask.bool()

    weights = (
        valid.unsqueeze(-1)
        .to(hidden.dtype)
    )

    pooled = (
        (hidden * weights).sum(dim=1)
        / weights.sum(dim=1).clamp_min(1.0)
    )

    return pooled


def mean_image_pool(
    model,
    hidden: torch.Tensor,
    inputs,
) -> torch.Tensor:
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get(
        "attention_mask"
    )

    if input_ids is None:
        raise RuntimeError(
            "mean_image requires input_ids."
        )

    if attention_mask is None:
        valid = torch.ones_like(
            input_ids,
            dtype=torch.bool,
        )
    else:
        valid = attention_mask.bool()

    image_token_id = None

    for obj in (
        getattr(model, "config", None),
        getattr(
            getattr(model, "model", None),
            "config",
            None,
        ),
    ):
        if (
            obj is not None
            and hasattr(
                obj,
                "image_token_id",
            )
        ):
            image_token_id = getattr(
                obj,
                "image_token_id",
            )

            if image_token_id is not None:
                break

    if image_token_id is None:
        raise RuntimeError(
            "Could not determine image_token_id."
        )

    mask = (
        (input_ids == int(image_token_id))
        & valid
    )

    if torch.any(mask.sum(dim=1) == 0):
        raise RuntimeError(
            "No image-token positions found."
        )

    weights = (
        mask.unsqueeze(-1)
        .to(hidden.dtype)
    )

    pooled = (
        (hidden * weights).sum(dim=1)
        / weights.sum(dim=1).clamp_min(1.0)
    )

    return pooled


@torch.no_grad()
def extract_embedding(
    model,
    processor,
    frames,
    device,
    pooling: str,
) -> np.ndarray:
    inputs = prepare_inputs(
        processor,
        frames,
        device,
    )

    backbone = base.get_backbone(
        model
    )

    outputs = backbone(
        **inputs,
        use_cache=False,
        return_dict=True,
    )

    hidden = outputs.last_hidden_state

    if pooling == "mean_all":
        pooled = mean_all_pool(
            hidden,
            inputs.get("attention_mask"),
        )

    elif pooling == "mean_image":
        pooled = mean_image_pool(
            model,
            hidden,
            inputs,
        )

    else:
        raise ValueError(
            f"Unknown pooling: {pooling}"
        )

    vector = (
        pooled[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return vector


def cache_filename(
    sample_id: str,
    pooling: str,
) -> str:
    digest = hashlib.sha1(
        sample_id.encode("utf-8")
    ).hexdigest()[:10]

    safe = "".join(
        c if c.isalnum() or c in "-_"
        else "_"
        for c in sample_id
    )

    return (
        f"{safe}_{pooling}_{digest}.npy"
    )


def extract_all_embeddings(
    manifest: pd.DataFrame,
    model,
    processor,
    frame_cache,
    args,
) -> Dict[str, np.ndarray]:
    device = torch.device("cuda:0")

    cache_dir = Path(
        args.embedding_cache
    )
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings = {}

    total = len(manifest)

    print("\nEXTRACTING FROZEN EMBEDDINGS")
    print("=" * 72)
    print("Segments:", total)
    print("Pooling:", args.pooling)
    print(
        "Embedding cache:",
        cache_dir,
    )

    for i, row in enumerate(
        manifest.itertuples(index=False),
        start=1,
    ):
        sample_id = str(row.sample_id)

        path = (
            cache_dir
            / cache_filename(
                sample_id,
                args.pooling,
            )
        )

        if path.exists():
            vector = np.load(path)
            status = "cached"
        else:
            try:
                frames = frame_cache.build(
                    row
                )

                vector = extract_embedding(
                    model=model,
                    processor=processor,
                    frames=frames,
                    device=device,
                    pooling=args.pooling,
                )

                np.save(
                    path,
                    vector,
                )

                status = "extracted"

            except Exception as exc:
                print(
                    f"  ERROR {sample_id}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

        embeddings[
            sample_id
        ] = vector.astype(
            np.float32,
            copy=False,
        )

        print(
            f"  [{i:04d}/{total:04d}] "
            f"{sample_id} | "
            f"{status} | "
            f"dim={len(vector)}",
            flush=True,
        )

        gc.collect()
        torch.cuda.empty_cache()

    print(
        "\nEmbeddings available:",
        len(embeddings),
        "/",
        total,
    )

    return embeddings


# ---------------------------------------------------------------------------
# Probe dataset + Ridge
# ---------------------------------------------------------------------------

def build_matrix(
    df: pd.DataFrame,
    embeddings: Dict[str, np.ndarray],
):
    rows = []

    X = []
    y = []

    for row in df.itertuples(index=False):
        sid = str(row.sample_id)

        if sid not in embeddings:
            continue

        X.append(
            embeddings[sid]
        )
        y.append(
            float(row.WD_P_mean)
        )

        rows.append(
            {
                "sample_id": sid,
                "patient_id": str(
                    row.patient_id
                ),
                "video": str(row.video),
                "segment_id": int(
                    row.segment_id
                ),
                "WD_P_true": float(
                    row.WD_P_mean
                ),
            }
        )

    if not X:
        raise RuntimeError(
            "No embeddings found for split."
        )

    return (
        np.stack(X).astype(np.float32),
        np.asarray(y, dtype=np.float32),
        pd.DataFrame(rows),
    )


def fit_probe(
    train_X,
    train_y,
    val_X,
    val_y,
    alphas: Sequence[float],
):
    scaler = StandardScaler(
        with_mean=True,
        with_std=True,
    )

    train_Z = scaler.fit_transform(
        train_X
    )

    val_Z = scaler.transform(
        val_X
    )

    search_rows = []

    best = None

    for alpha in alphas:
        model = Ridge(
            alpha=float(alpha),
            solver="lsqr",
            fit_intercept=True,
            max_iter=10000,
            tol=1e-6,
        )

        model.fit(
            train_Z,
            train_y,
        )

        pred = model.predict(
            val_Z
        )

        metrics = regression_metrics(
            val_y,
            pred,
        )

        row = {
            "alpha": float(alpha),
            "val_MAE": metrics["MAE"],
            "val_RMSE": metrics["RMSE"],
            "val_Spearman": metrics["Spearman"],
            "val_pred_std": metrics["pred_std"],
            "val_pred_min": metrics["pred_min"],
            "val_pred_max": metrics["pred_max"],
        }

        search_rows.append(row)

        if (
            best is None
            or row["val_MAE"]
            < best[0]
        ):
            best = (
                row["val_MAE"],
                float(alpha),
            )

    if best is None:
        raise RuntimeError(
            "Alpha search failed."
        )

    best_alpha = best[1]

    final_model = Ridge(
        alpha=best_alpha,
        solver="lsqr",
        fit_intercept=True,
        max_iter=10000,
        tol=1e-6,
    )

    final_model.fit(
        train_Z,
        train_y,
    )

    return (
        scaler,
        final_model,
        best_alpha,
        pd.DataFrame(search_rows),
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_alpha_search(
    search_df: pd.DataFrame,
    output_dir: Path,
):
    fig = plt.figure(
        figsize=(7, 4.5)
    )
    ax = fig.add_subplot(111)

    ax.plot(
        search_df["alpha"],
        search_df["val_MAE"],
        marker="o",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Ridge alpha")
    ax.set_ylabel("Validation WD_P MAE")
    ax.set_title(
        "Frozen-Qwen Ridge probe: validation alpha search"
    )
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "ridge_alpha_validation.png",
        dpi=180,
    )

    plt.close(fig)


def plot_true_vs_pred(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
):
    fig = plt.figure(
        figsize=(6, 6)
    )
    ax = fig.add_subplot(111)

    ax.scatter(
        df["WD_P_true"],
        df["WD_P_pred"],
        alpha=0.65,
    )

    low = min(
        1.0,
        float(df["WD_P_true"].min()),
        float(df["WD_P_pred"].min()),
    )
    high = max(
        5.0,
        float(df["WD_P_true"].max()),
        float(df["WD_P_pred"].max()),
    )

    ax.plot(
        [low, high],
        [low, high],
        linestyle="--",
    )

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("True WD_P")
    ax.set_ylabel("Predicted WD_P")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)


def plot_patient_predictions(
    test_df: pd.DataFrame,
    output_path: Path,
):
    fig = plt.figure(
        figsize=(8, 5)
    )
    ax = fig.add_subplot(111)

    patients = list(
        test_df["patient_id"]
        .astype(str)
        .unique()
    )

    data = [
        test_df.loc[
            test_df["patient_id"].astype(str)
            == pid,
            "WD_P_pred",
        ].to_numpy()
        for pid in patients
    ]

    ax.boxplot(
        data,
        tick_labels=patients,
    )

    ax.set_xlabel("Held-out patient")
    ax.set_ylabel("Predicted WD_P")
    ax.set_title(
        "Frozen-Qwen Ridge probe predictions by test patient"
    )
    ax.grid(True, alpha=0.20)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_parser():
    p = argparse.ArgumentParser(
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
        description=(
            "Frozen Qwen3-VL representation probe "
            "for patient withdrawal salience."
        ),
    )

    p.add_argument(
        "--manifest",
        required=True,
        help=(
            "Existing patient-disjoint manifest from "
            "the WD-only large-data experiment."
        ),
    )

    p.add_argument(
        "--video-root",
        default=(
            r"C:\Data\Sequence_model"
            r"\Memopsy_videos\CONVERTED"
        ),
    )

    p.add_argument(
        "--yunet-model",
        default=(
            r".\models\face_detection_yunet"
            r"\face_detection_yunet_2026may.onnx"
        ),
    )

    p.add_argument(
        "--frame-cache",
        default=(
            r".\output\qwen3vl_rupture_finetune_16f_stable"
            r"\frame_cache_16"
        ),
    )

    p.add_argument(
        "--embedding-cache",
        default=(
            r".\output\qwen3vl_frozen_wd_probe"
            r"\embedding_cache"
        ),
    )

    p.add_argument(
        "--output-dir",
        default=(
            r".\output\qwen3vl_frozen_wd_probe"
            r"\run1"
        ),
    )

    p.add_argument(
        "--model-id",
        default=MODEL_ID,
    )

    p.add_argument(
        "--num-frames",
        type=int,
        default=16,
    )

    p.add_argument(
        "--frame-width",
        type=int,
        default=224,
    )

    p.add_argument(
        "--pooling",
        choices=[
            "mean_all",
            "mean_image",
        ],
        default="mean_all",
    )

    p.add_argument(
        "--positive-threshold",
        type=float,
        default=3.0,
    )

    p.add_argument(
        "--alphas",
        default=(
            "0.01,0.1,1,10,100,1000"
        ),
        help=(
            "Comma-separated Ridge alpha values."
        ),
    )

    p.add_argument(
        "--attn-implementation",
        choices=[
            "sdpa",
            "eager",
            "flash_attention_2",
        ],
        default="sdpa",
    )

    p.add_argument(
        "--no-4bit",
        action="store_true",
    )

    p.add_argument(
        "--extract-only",
        action="store_true",
        help=(
            "Only extract/cache embeddings; "
            "do not fit Ridge."
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return p


def main():
    args = make_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = Path(
        args.manifest
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path
    )

    required = {
        "sample_id",
        "split",
        "patient_id",
        "video",
        "segment_id",
        "WD_P_mean",
        "video_path",
        "patient_side",
    }

    missing = (
        required
        - set(manifest.columns)
    )

    if missing:
        raise RuntimeError(
            f"Manifest missing columns: "
            f"{sorted(missing)}"
        )

    print(
        "FROZEN QWEN3-VL WD REPRESENTATION PROBE"
    )
    print("=" * 72)
    print("Manifest:", manifest_path)
    print("Segments:", len(manifest))
    print(
        "Patients:",
        manifest["patient_id"].nunique(),
    )
    print(
        "Pooling:",
        args.pooling,
    )
    print(
        "Frames/minute:",
        args.num_frames,
    )
    print(
        "Qwen trainable parameters: 0"
    )

    print("\nSplit summary")
    print(
        manifest.groupby("split")
        .agg(
            segments=("sample_id", "size"),
            patients=("patient_id", "nunique"),
            WD_mean=("WD_P_mean", "mean"),
            WD_std=("WD_P_mean", "std"),
        )
        .to_string()
    )

    model, processor = load_frozen_qwen(
        args
    )

    cropper = base.PatientCropper(
        yunet_model=Path(
            args.yunet_model
        ),
        output_width=args.frame_width,
    )

    frame_cache = base.FrameCache(
        cache_root=Path(
            args.frame_cache
        ),
        cropper=cropper,
        num_frames=args.num_frames,
    )

    embeddings = extract_all_embeddings(
        manifest=manifest,
        model=model,
        processor=processor,
        frame_cache=frame_cache,
        args=args,
    )

    # Qwen no longer needed after embeddings exist.
    del model
    del processor

    gc.collect()
    torch.cuda.empty_cache()

    if args.extract_only:
        print(
            "\nEXTRACT-ONLY complete."
        )
        return

    train_df = manifest[
        manifest["split"] == "train"
    ].copy()

    val_df = manifest[
        manifest["split"] == "val"
    ].copy()

    test_df = manifest[
        manifest["split"] == "test"
    ].copy()

    (
        train_X,
        train_y,
        train_meta,
    ) = build_matrix(
        train_df,
        embeddings,
    )

    (
        val_X,
        val_y,
        val_meta,
    ) = build_matrix(
        val_df,
        embeddings,
    )

    (
        test_X,
        test_y,
        test_meta,
    ) = build_matrix(
        test_df,
        embeddings,
    )

    print("\nPROBE MATRICES")
    print("=" * 72)
    print(
        "Train:",
        train_X.shape,
    )
    print(
        "Val:",
        val_X.shape,
    )
    print(
        "Test:",
        test_X.shape,
    )

    alphas = [
        float(x.strip())
        for x in args.alphas.split(",")
        if x.strip()
    ]

    (
        scaler,
        ridge,
        best_alpha,
        alpha_search,
    ) = fit_probe(
        train_X,
        train_y,
        val_X,
        val_y,
        alphas,
    )

    alpha_search.to_csv(
        output_dir
        / "ridge_alpha_validation.csv",
        index=False,
    )

    plot_alpha_search(
        alpha_search,
        output_dir,
    )

    print(
        "\nSelected Ridge alpha:",
        best_alpha,
    )

    train_Z = scaler.transform(
        train_X
    )
    val_Z = scaler.transform(
        val_X
    )
    test_Z = scaler.transform(
        test_X
    )

    train_pred = ridge.predict(
        train_Z
    )
    val_pred = ridge.predict(
        val_Z
    )
    test_pred = ridge.predict(
        test_Z
    )

    train_metrics = regression_metrics(
        train_y,
        train_pred,
        args.positive_threshold,
    )

    val_metrics = regression_metrics(
        val_y,
        val_pred,
        args.positive_threshold,
    )

    test_metrics = regression_metrics(
        test_y,
        test_pred,
        args.positive_threshold,
    )

    # Constant baseline: training-set mean only.
    train_mean = float(
        np.mean(train_y)
    )

    baseline_val = np.full_like(
        val_y,
        fill_value=train_mean,
        dtype=float,
    )

    baseline_test = np.full_like(
        test_y,
        fill_value=train_mean,
        dtype=float,
    )

    baseline_val_metrics = (
        regression_metrics(
            val_y,
            baseline_val,
            args.positive_threshold,
        )
    )

    baseline_test_metrics = (
        regression_metrics(
            test_y,
            baseline_test,
            args.positive_threshold,
        )
    )

    print_metrics(
        "TRAIN RIDGE METRICS",
        train_metrics,
    )

    print_metrics(
        "VALIDATION RIDGE METRICS",
        val_metrics,
    )

    print_metrics(
        "TEST RIDGE METRICS",
        test_metrics,
    )

    print_metrics(
        "TEST CONSTANT-TRAIN-MEAN BASELINE",
        baseline_test_metrics,
    )

    val_predictions = (
        val_meta.copy()
    )
    val_predictions[
        "WD_P_pred"
    ] = val_pred
    val_predictions[
        "baseline_train_mean"
    ] = baseline_val

    test_predictions = (
        test_meta.copy()
    )
    test_predictions[
        "WD_P_pred"
    ] = test_pred
    test_predictions[
        "baseline_train_mean"
    ] = baseline_test

    val_predictions.to_csv(
        output_dir
        / "val_predictions.csv",
        index=False,
    )

    test_predictions.to_csv(
        output_dir
        / "test_predictions.csv",
        index=False,
    )

    # Patient-wise held-out test metrics.
    patient_rows = []

    for pid, group in (
        test_predictions.groupby(
            "patient_id"
        )
    ):
        metrics = regression_metrics(
            group["WD_P_true"].to_numpy(),
            group["WD_P_pred"].to_numpy(),
            args.positive_threshold,
        )

        patient_rows.append(
            {
                "patient_id": str(pid),
                "n": len(group),
                "WD_mean": float(
                    group["WD_P_true"].mean()
                ),
                "pred_mean": float(
                    group["WD_P_pred"].mean()
                ),
                "pred_std": float(
                    group["WD_P_pred"].std(
                        ddof=0
                    )
                ),
                "MAE": metrics["MAE"],
                "Spearman": metrics["Spearman"],
            }
        )

    patient_metrics = pd.DataFrame(
        patient_rows
    )

    patient_metrics.to_csv(
        output_dir
        / "test_patient_metrics.csv",
        index=False,
    )

    valid_patient_spearman = (
        patient_metrics[
            "Spearman"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    macro_within_patient_spearman = (
        float(
            valid_patient_spearman.mean()
        )
        if len(valid_patient_spearman)
        else float("nan")
    )

    summary = {
        "model_id": args.model_id,
        "qwen_frozen": True,
        "pooling": args.pooling,
        "num_frames": args.num_frames,
        "frame_width": args.frame_width,
        "embedding_dim": int(
            train_X.shape[1]
        ),
        "train_n": int(len(train_y)),
        "val_n": int(len(val_y)),
        "test_n": int(len(test_y)),
        "selected_ridge_alpha": float(
            best_alpha
        ),
        "train_WD_mean": train_mean,
        "train_ridge": train_metrics,
        "val_ridge": val_metrics,
        "test_ridge": test_metrics,
        "val_constant_baseline": (
            baseline_val_metrics
        ),
        "test_constant_baseline": (
            baseline_test_metrics
        ),
        "macro_within_test_patient_spearman": (
            macro_within_patient_spearman
        ),
    }

    write_json(
        output_dir / "summary.json",
        summary,
    )

    plot_true_vs_pred(
        val_predictions,
        output_dir
        / "val_true_vs_pred.png",
        "Frozen Qwen Ridge probe: validation",
    )

    plot_true_vs_pred(
        test_predictions,
        output_dir
        / "test_true_vs_pred.png",
        "Frozen Qwen Ridge probe: held-out test",
    )

    plot_patient_predictions(
        test_predictions,
        output_dir
        / "test_predictions_by_patient.png",
    )

    print("\nKEY COMPARISON")
    print("=" * 72)
    print(
        f"Constant baseline test MAE : "
        f"{baseline_test_metrics['MAE']:.4f}"
    )
    print(
        f"Frozen-Qwen Ridge test MAE: "
        f"{test_metrics['MAE']:.4f}"
    )
    print(
        f"Frozen-Qwen test Spearman : "
        f"{test_metrics['Spearman']:.4f}"
    )
    print(
        f"Frozen-Qwen pred std       : "
        f"{test_metrics['pred_std']:.4f}"
    )
    print(
        f"Within-patient Spearman    : "
        f"{macro_within_patient_spearman:.4f}"
    )

    print("\nFILES")
    print("=" * 72)
    print(
        output_dir
        / "summary.json"
    )
    print(
        output_dir
        / "ridge_alpha_validation.csv"
    )
    print(
        output_dir
        / "ridge_alpha_validation.png"
    )
    print(
        output_dir
        / "val_predictions.csv"
    )
    print(
        output_dir
        / "test_predictions.csv"
    )
    print(
        output_dir
        / "test_patient_metrics.csv"
    )
    print(
        output_dir
        / "test_true_vs_pred.png"
    )
    print(
        output_dir
        / "test_predictions_by_patient.png"
    )


if __name__ == "__main__":
    main()