#!/usr/bin/env python
"""
Diagnostic for Qwen3-VL rupture fine-tuning representation collapse.

Purpose
-------
Loads the same Qwen3-VL model and the saved best LoRA adapter from the rupture
pilot, extracts hidden representations for multiple psychotherapy segments,
and compares three pooling strategies:

1. final_token  - current regression-head input
2. mean_all     - mean across all non-padding multimodal tokens
3. mean_image   - mean across image-token positions when available

This script does NOT train anything.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
from scipy.stats import spearmanr
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

# Allow import when this diagnostic script is placed in the project's scripts folder.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from finetune_qwen3vl_rupture_pilot import (
    FrameCache,
    PatientCropper,
    RuptureRegressionHead,
    get_backbone,
    prepare_multimodal_inputs,
)

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    try:
        return float(spearmanr(x, y).statistic)
    except Exception:
        return float("nan")


def pairwise_stats(name, matrix, wd_labels):
    """
    matrix: [N, D] CPU float tensor
    """
    x = F.normalize(matrix.float(), dim=1)
    sim = (x @ x.T).cpu().numpy()

    n = sim.shape[0]
    vals = []
    wd_diffs = []

    for i in range(n):
        for j in range(i + 1, n):
            vals.append(float(sim[i, j]))
            wd_diffs.append(abs(float(wd_labels[i]) - float(wd_labels[j])))

    vals = np.asarray(vals, dtype=float)
    wd_diffs = np.asarray(wd_diffs, dtype=float)
    cosine_distance = 1.0 - vals

    # Mean variance across embedding dimensions.
    dim_variance = matrix.float().var(dim=0, unbiased=False)
    mean_dim_variance = float(dim_variance.mean().item())

    return {
        "pooling": name,
        "n_samples": int(n),
        "n_pairs": int(len(vals)),
        "cosine_mean": float(vals.mean()) if len(vals) else float("nan"),
        "cosine_median": float(np.median(vals)) if len(vals) else float("nan"),
        "cosine_std": float(vals.std()) if len(vals) else float("nan"),
        "cosine_min": float(vals.min()) if len(vals) else float("nan"),
        "cosine_max": float(vals.max()) if len(vals) else float("nan"),
        "mean_dim_variance": mean_dim_variance,
        "spearman_abs_WD_difference_vs_cosine_distance": safe_spearman(
            wd_diffs, cosine_distance
        ),
    }, sim


def load_model(args):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base Qwen3-VL...", flush=True)

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=quant,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=args.attn_implementation,
    )
    base.config.use_cache = False

    print("Loading best LoRA adapter:", args.adapter_dir, flush=True)
    model = PeftModel.from_pretrained(
        base,
        args.adapter_dir,
        is_trainable=False,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_id)

    base_model = model.get_base_model()
    hidden_size = int(base_model.config.text_config.hidden_size)

    head = RuptureRegressionHead(
        hidden_size=hidden_size,
        dropout=0.10,
    ).to("cuda", dtype=torch.float32)

    state = torch.load(
        args.head_path,
        map_location="cpu",
        weights_only=True,
    )
    head.load_state_dict(state)
    head.eval()

    return model, processor, head


@torch.no_grad()
def extract_one(model, head, processor, frames, device):
    inputs = prepare_multimodal_inputs(
        processor,
        frames,
        device,
    )

    backbone = get_backbone(model)
    outputs = backbone(
        **inputs,
        use_cache=False,
        return_dict=True,
    )

    hidden = outputs.last_hidden_state.float()  # [1, T, D]
    attention_mask = inputs.get("attention_mask")

    if attention_mask is None:
        valid_mask = torch.ones(
            hidden.shape[:2],
            device=hidden.device,
            dtype=torch.bool,
        )
        last_idx = hidden.shape[1] - 1
    else:
        valid_mask = attention_mask.bool()
        last_idx = int(attention_mask.long().sum(dim=1)[0].item() - 1)

    final_token = hidden[0, last_idx, :]
    mean_all = hidden[0, valid_mask[0], :].mean(dim=0)

    image_mean = None
    image_token_count = 0

    input_ids = inputs.get("input_ids")
    image_token_id = None

    # Qwen config normally exposes image_token_id on the base model config.
    base_model = model.get_base_model()
    for obj in (
        getattr(base_model, "config", None),
        getattr(getattr(base_model, "model", None), "config", None),
    ):
        if obj is not None and hasattr(obj, "image_token_id"):
            image_token_id = getattr(obj, "image_token_id")
            if image_token_id is not None:
                break

    if input_ids is not None and image_token_id is not None:
        image_mask = (input_ids == int(image_token_id)) & valid_mask
        image_token_count = int(image_mask.sum().item())
        if image_token_count > 0:
            image_mean = hidden[0, image_mask[0], :].mean(dim=0)

    # Current prediction from the exact final-token representation used by training.
    current_pred = head(final_token.unsqueeze(0))[0]

    return {
        "final_token": final_token.detach().cpu(),
        "mean_all": mean_all.detach().cpu(),
        "mean_image": image_mean.detach().cpu() if image_mean is not None else None,
        "image_token_count": image_token_count,
        "WD_pred_current_head": float(current_pred[0].item()),
        "CF_pred_current_head": float(current_pred[1].item()),
    }


def choose_samples(df, n_samples, seed):
    """
    Use all splits when possible and include WD variation.
    Deterministic sample with broad WD range.
    """
    if len(df) <= n_samples:
        return df.copy().reset_index(drop=True)

    # Bin WD to encourage low/middle/high scores in the diagnostic.
    work = df.copy()
    work["_wd_bin"] = pd.cut(
        work["WD_P_mean"],
        bins=[-np.inf, 1.5, 2.5, np.inf],
        labels=["low", "mid", "high"],
    )

    per_bin = max(1, n_samples // 3)
    pieces = []

    for i, label in enumerate(["low", "mid", "high"]):
        part = work[work["_wd_bin"] == label]
        if len(part):
            pieces.append(
                part.sample(
                    n=min(per_bin, len(part)),
                    random_state=seed + i,
                )
            )

    selected = pd.concat(pieces).drop_duplicates(subset=["sample_id"])

    if len(selected) < n_samples:
        remaining = work.drop(index=selected.index)
        if len(remaining):
            selected = pd.concat(
                [
                    selected,
                    remaining.sample(
                        n=min(n_samples - len(selected), len(remaining)),
                        random_state=seed + 99,
                    ),
                ]
            )

    return (
        selected.head(n_samples)
        .drop(columns=["_wd_bin"], errors="ignore")
        .reset_index(drop=True)
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--head-path", required=True)
    p.add_argument("--frame-cache", required=True)
    p.add_argument("--yunet-model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--frame-width", type=int, default=224)
    p.add_argument("--n-samples", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attn-implementation", default="sdpa")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    sample_df = choose_samples(manifest, args.n_samples, args.seed)

    print("QWEN3-VL POOLING DIAGNOSTIC")
    print("=" * 72)
    print("Manifest:", args.manifest)
    print("Diagnostic samples:", len(sample_df))
    print(
        "WD label range:",
        float(sample_df["WD_P_mean"].min()),
        "->",
        float(sample_df["WD_P_mean"].max()),
    )
    print("Split counts:")
    print(sample_df["split"].value_counts().to_string())

    model, processor, head = load_model(args)

    cropper = PatientCropper(
        yunet_model=Path(args.yunet_model),
        output_width=args.frame_width,
    )
    frame_cache = FrameCache(
        cache_root=Path(args.frame_cache),
        cropper=cropper,
        num_frames=args.num_frames,
    )

    device = torch.device("cuda:0")

    embeddings = {
        "final_token": [],
        "mean_all": [],
        "mean_image": [],
    }
    rows = []

    for i, row in enumerate(sample_df.itertuples(index=False), start=1):
        try:
            frames = frame_cache.build(row)
            result = extract_one(
                model,
                head,
                processor,
                frames,
                device,
            )

            embeddings["final_token"].append(result["final_token"])
            embeddings["mean_all"].append(result["mean_all"])
            if result["mean_image"] is not None:
                embeddings["mean_image"].append(result["mean_image"])

            rows.append(
                {
                    "sample_id": row.sample_id,
                    "split": row.split,
                    "patient_id": row.patient_id,
                    "WD_P_true": float(row.WD_P_mean),
                    "CF_P_true": float(row.CF_P_mean),
                    "WD_pred_current_head": result["WD_pred_current_head"],
                    "CF_pred_current_head": result["CF_pred_current_head"],
                    "image_token_count": result["image_token_count"],
                }
            )

            print(
                f"[{i:02d}/{len(sample_df):02d}] {row.sample_id} "
                f"WD={row.WD_P_mean:.1f}->{result['WD_pred_current_head']:.3f} "
                f"CF={row.CF_P_mean:.1f}->{result['CF_pred_current_head']:.3f} "
                f"image_tokens={result['image_token_count']}",
                flush=True,
            )

        except Exception as exc:
            print(
                f"ERROR {getattr(row, 'sample_id', '?')}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

        torch.cuda.empty_cache()

    if len(rows) < 3:
        raise RuntimeError("Too few successful samples for representation diagnostic.")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "sample_predictions.csv", index=False)

    wd = result_df["WD_P_true"].to_numpy()

    summaries = []
    matrices = {}

    # final_token and mean_all always have one embedding per successful row.
    for name in ("final_token", "mean_all"):
        mat = torch.stack(embeddings[name], dim=0)
        summary, sim = pairwise_stats(name, mat, wd)
        summaries.append(summary)
        matrices[name] = sim
        np.save(out_dir / f"{name}_embeddings.npy", mat.numpy())
        pd.DataFrame(sim).to_csv(
            out_dir / f"{name}_cosine_matrix.csv",
            index=False,
        )

    # Only compare image-token pooling if it was available for every successful sample.
    if len(embeddings["mean_image"]) == len(rows):
        mat = torch.stack(embeddings["mean_image"], dim=0)
        summary, sim = pairwise_stats("mean_image", mat, wd)
        summaries.append(summary)
        matrices["mean_image"] = sim
        np.save(out_dir / "mean_image_embeddings.npy", mat.numpy())
        pd.DataFrame(sim).to_csv(
            out_dir / "mean_image_cosine_matrix.csv",
            index=False,
        )
    else:
        print(
            "\nNOTE: image-token pooling unavailable for all samples "
            f"({len(embeddings['mean_image'])}/{len(rows)} succeeded)."
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "pooling_summary.csv", index=False)

    print("\nPOOLING SUMMARY")
    print("=" * 72)
    print(summary_df.to_string(index=False))

    pred_std_wd = float(result_df["WD_pred_current_head"].std(ddof=0))
    pred_std_cf = float(result_df["CF_pred_current_head"].std(ddof=0))

    print("\nCURRENT HEAD OUTPUT SPREAD")
    print("=" * 72)
    print(f"WD prediction std: {pred_std_wd:.6f}")
    print(f"CF prediction std: {pred_std_cf:.6f}")

    interpretation = {
        "n_successful": int(len(rows)),
        "WD_prediction_std": pred_std_wd,
        "CF_prediction_std": pred_std_cf,
        "pooling_summary": summaries,
    }
    (out_dir / "diagnostic_summary.json").write_text(
        json.dumps(interpretation, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", out_dir / "pooling_summary.csv")
    print("Saved:", out_dir / "sample_predictions.csv")
    print("Saved:", out_dir / "diagnostic_summary.json")


if __name__ == "__main__":
    main()