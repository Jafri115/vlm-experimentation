#!/usr/bin/env python
"""
Qwen3-VL 4-window temporal MIL fine-tuning for psychotherapy rupture salience.

Architecture
------------
One labelled 60-second segment
    -> 4 chronological 15-second windows
    -> 4 frames per window (16 frames total by default)
    -> Qwen3-VL representation for each window
    -> learned positional embeddings + attention MIL aggregator
    -> one minute-level representation
    -> regression head
    -> WD_P_mean, CF_P_mean

Important:
- The 1-minute WD_P / CF_P label supervises ONLY the final minute prediction.
- Individual 15-second windows are NOT assigned the 1-minute label.
- Qwen LoRA remains trainable, so this is still QLoRA fine-tuning.
- The vision encoder stays frozen through the existing pilot model builder.
- This script reuses the preprocessing/model helpers from
  finetune_qwen3vl_rupture_pilot.py in the same scripts folder.
"""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import get_peft_model_state_dict, set_peft_model_state_dict

import finetune_qwen3vl_rupture_pilot as base


WINDOW_PROMPT = """The images are chronological frames sampled from one 15-second window
within a labelled 1-minute psychotherapy segment.

Focus only on the visible patient/person being observed.

Create an internal visual representation useful for estimating patient rupture-marker
salience over the FULL MINUTE:
- WD_P: patient withdrawal
- CF_P: patient confrontation

Attend to directly visible behavioral patterns and how they develop across the window,
including persistence, repetition, changes, and interaction-relevant visual behavior.

The final 3RS labels apply to the full 1-minute segment, not to this 15-second window.
Therefore:
- do not assign an independent 1-5 rupture rating to this window,
- do not assume that one isolated visual cue determines the minute-level label,
- preserve information that could help distinguish weak/ambiguous, clear, elevated,
  and dominant behavior when combined with the other windows.

Use only directly visible patient behavior.
Do not infer speech content, motivation, diagnosis, personality, hidden emotion, or
therapeutic meaning that cannot be established visually.
Do not use audio or transcript.
"""


class TemporalAttentionMIL(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_windows: int = 4,
        attention_hidden: int = 256,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.num_windows = int(num_windows)

        self.position = nn.Parameter(
            torch.zeros(1, self.num_windows, hidden_size)
        )
        nn.init.normal_(self.position, mean=0.0, std=0.02)

        self.norm = nn.LayerNorm(hidden_size)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, attention_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_hidden, 1),
        )

    def forward(self, window_hidden: torch.Tensor):
        if window_hidden.ndim != 3:
            raise ValueError(
                f"Expected [B,W,D] window tensor, got {tuple(window_hidden.shape)}"
            )

        if window_hidden.shape[1] != self.num_windows:
            raise ValueError(
                f"Expected {self.num_windows} windows, got {window_hidden.shape[1]}"
            )

        x = window_hidden.float() + self.position
        x = self.norm(x)

        logits = self.scorer(x).squeeze(-1)
        weights = torch.softmax(logits, dim=1)

        pooled = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1,
        )

        return pooled, weights


def prepare_window_inputs(
    processor,
    frames: Sequence[Image.Image],
    device: torch.device,
):
    content = [
        {"type": "image", "image": frame}
        for frame in frames
    ]
    content.append(
        {"type": "text", "text": WINDOW_PROMPT}
    )

    messages = [{"role": "user", "content": content}]

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

        if key in {"pixel_values", "pixel_values_videos"}:
            moved[key] = value.to(
                device=device,
                dtype=torch.bfloat16,
            )
        else:
            moved[key] = value.to(device=device)

    return moved


def pool_hidden_state(
    model,
    inputs,
    hidden: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    attention_mask = inputs.get("attention_mask")

    if attention_mask is None:
        valid_mask = torch.ones(
            hidden.shape[:2],
            device=hidden.device,
            dtype=torch.bool,
        )
    else:
        valid_mask = attention_mask.bool()

    if pooling == "final_token":
        if attention_mask is None:
            return hidden[:, -1, :]

        last_idx = attention_mask.long().sum(dim=1) - 1
        batch_idx = torch.arange(
            hidden.shape[0],
            device=hidden.device,
        )
        return hidden[batch_idx, last_idx, :]

    if pooling == "mean_all":
        weights = valid_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return summed / denom

    if pooling == "mean_image":
        input_ids = inputs.get("input_ids")

        if input_ids is None:
            raise RuntimeError(
                "mean_image pooling requires input_ids."
            )

        base_model = (
            model.get_base_model()
            if hasattr(model, "get_base_model")
            else model
        )

        image_token_id = None
        for obj in (
            getattr(base_model, "config", None),
            getattr(
                getattr(base_model, "model", None),
                "config",
                None,
            ),
        ):
            if obj is not None and hasattr(obj, "image_token_id"):
                image_token_id = getattr(obj, "image_token_id")
                if image_token_id is not None:
                    break

        if image_token_id is None:
            raise RuntimeError(
                "Could not determine Qwen3-VL image_token_id."
            )

        image_mask = (
            (input_ids == int(image_token_id))
            & valid_mask
        )

        counts = image_mask.sum(dim=1)
        if torch.any(counts == 0):
            raise RuntimeError(
                "No image-token positions found for a window."
            )

        weights = image_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return summed / denom

    raise ValueError(
        f"Unsupported window pooling mode: {pooling!r}"
    )


def split_frames_into_windows(
    frames: Sequence[Image.Image],
    num_windows: int,
    frames_per_window: int,
) -> List[List[Image.Image]]:
    expected = int(num_windows) * int(frames_per_window)

    if len(frames) != expected:
        raise RuntimeError(
            f"Expected exactly {expected} cached frames "
            f"({num_windows}x{frames_per_window}), got {len(frames)}."
        )

    return [
        list(
            frames[
                i * frames_per_window:
                (i + 1) * frames_per_window
            ]
        )
        for i in range(num_windows)
    ]


def predict_temporal_scores(
    model,
    aggregator,
    head,
    processor,
    frames,
    device,
    num_windows: int,
    frames_per_window: int,
    pooling: str,
):
    windows = split_frames_into_windows(
        frames,
        num_windows=num_windows,
        frames_per_window=frames_per_window,
    )

    backbone = base.get_backbone(model)
    window_embeddings = []

    for window_frames in windows:
        inputs = prepare_window_inputs(
            processor,
            window_frames,
            device,
        )

        outputs = backbone(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

        hidden = outputs.last_hidden_state

        pooled = pool_hidden_state(
            model=model,
            inputs=inputs,
            hidden=hidden,
            pooling=pooling,
        )

        window_embeddings.append(pooled)

    window_tensor = torch.stack(
        window_embeddings,
        dim=1,
    )

    minute_hidden, attention_weights = aggregator(
        window_tensor
    )

    pred = head(minute_hidden)

    return pred, attention_weights


@torch.no_grad()
def evaluate_split(
    split_df: pd.DataFrame,
    split_name: str,
    model,
    aggregator,
    head,
    processor,
    frame_cache,
    device,
    args,
    output_dir: Path,
    max_examples: Optional[int] = None,
):
    model.eval()
    aggregator.eval()
    head.eval()

    rows = []

    eval_df = split_df.copy()
    if max_examples is not None and max_examples > 0:
        eval_df = eval_df.head(max_examples)

    print(f"\nEvaluating {split_name}: {len(eval_df)} segments")

    for i, row in enumerate(eval_df.itertuples(index=False), start=1):
        try:
            frames = frame_cache.build(row)

            pred, weights = predict_temporal_scores(
                model=model,
                aggregator=aggregator,
                head=head,
                processor=processor,
                frames=frames,
                device=device,
                num_windows=args.window_count,
                frames_per_window=args.frames_per_window,
                pooling=args.pooling,
            )

            pred_np = pred[0].detach().float().cpu().numpy()
            weights_np = weights[0].detach().float().cpu().numpy()

            result = {
                "sample_id": row.sample_id,
                "patient_id": row.patient_id,
                "video": row.video,
                "segment_id": row.segment_id,
                "WD_P_true": float(row.WD_P_mean),
                "CF_P_true": float(row.CF_P_mean),
                "WD_P_pred": float(pred_np[0]),
                "CF_P_pred": float(pred_np[1]),
            }

            for w_i, value in enumerate(weights_np, start=1):
                result[f"window_weight_{w_i}"] = float(value)

            rows.append(result)

            weight_text = ",".join(
                f"{x:.2f}" for x in weights_np
            )

            print(
                f"  [{i:03d}/{len(eval_df):03d}] "
                f"{row.sample_id} | "
                f"WD {row.WD_P_mean:.1f}->{pred_np[0]:.2f} | "
                f"CF {row.CF_P_mean:.1f}->{pred_np[1]:.2f} | "
                f"W=[{weight_text}]",
                flush=True,
            )

        except Exception as exc:
            print(
                f"  ERROR {row.sample_id}: {type(exc).__name__}: {exc}",
                flush=True,
            )

        torch.cuda.empty_cache()

    pred_path = output_dir / f"{split_name}_predictions.csv"
    pd.DataFrame(rows).to_csv(pred_path, index=False)

    metrics = base.compute_metrics(
        rows,
        args.positive_threshold,
    )

    base.write_json(
        output_dir / f"{split_name}_metrics.json",
        metrics,
    )

    print(f"\n{split_name.upper()} METRICS")
    print(json.dumps(metrics, indent=2))

    return metrics


def train(args, manifest: pd.DataFrame):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base.seed_everything(args.seed)

    expected_frames = (
        args.window_count
        * args.frames_per_window
    )

    if args.num_frames != expected_frames:
        raise ValueError(
            f"--num-frames must equal --window-count * --frames-per-window. "
            f"Got {args.num_frames} vs "
            f"{args.window_count}*{args.frames_per_window}={expected_frames}."
        )

    train_df = manifest[manifest["split"] == "train"].copy()
    val_df = manifest[manifest["split"] == "val"].copy()
    test_df = manifest[manifest["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(
            "Manifest must contain non-empty train/val/test."
        )

    model, processor, head = base.build_model_and_processor(args)

    base_model = model.get_base_model()
    hidden_size = int(
        base_model.config.text_config.hidden_size
    )

    aggregator = TemporalAttentionMIL(
        hidden_size=hidden_size,
        num_windows=args.window_count,
        attention_hidden=args.temporal_attention_hidden,
        dropout=args.temporal_dropout,
    ).to("cuda", dtype=torch.float32)

    cropper = base.PatientCropper(
        yunet_model=Path(args.yunet_model),
        output_width=args.frame_width,
    )

    frame_cache = base.FrameCache(
        cache_root=Path(args.frame_cache),
        cropper=cropper,
        num_frames=args.num_frames,
    )

    device = torch.device("cuda:0")

    lora_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    temporal_params = (
        list(aggregator.parameters())
        + list(head.parameters())
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_params,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": temporal_params,
                "lr": args.head_learning_rate,
                "weight_decay": args.weight_decay,
            },
        ]
    )

    steps_per_epoch = math.ceil(
        len(train_df) / args.grad_accum_steps
    )
    planned_steps = steps_per_epoch * args.epochs

    if args.max_train_steps > 0:
        planned_steps = min(
            planned_steps,
            args.max_train_steps,
        )

    print("\nTRAINING PLAN")
    print("=" * 72)
    print("Architecture: 4-window temporal attention MIL")
    print("Model:", args.model_id)
    print("Train segments:", len(train_df))
    print("Val segments:", len(val_df))
    print("Test segments:", len(test_df))
    print("Windows:", f"{args.window_count} x 15 sec")
    print("Frames/window:", args.frames_per_window)
    print("Total frames/minute:", args.num_frames)
    print("Window pooling:", args.pooling)
    print(
        "Temporal attention hidden:",
        args.temporal_attention_hidden,
    )
    print(
        "Gradient accumulation:",
        args.grad_accum_steps,
    )
    print("Epochs:", args.epochs)
    print(
        "Max optimizer steps:",
        planned_steps,
    )
    print("LoRA LR:", args.learning_rate)
    print(
        "Aggregator/head LR:",
        args.head_learning_rate,
    )
    print("Output:", output_dir)

    run_config = vars(args).copy()
    run_config.update(
        {
            "architecture": "4_window_attention_MIL",
            "window_prompt": WINDOW_PROMPT,
            "label_level": "one_minute",
            "window_labels_used": False,
        }
    )

    base.write_json(
        output_dir / "run_config.json",
        run_config,
    )

    best_val = float("inf")
    best_epoch = None
    best_optimizer_step = None

    best_lora_state = None
    best_head_state = None
    best_aggregator_state = None

    optimizer_step = 0
    accum_counter = 0

    optimizer.zero_grad(set_to_none=True)
    stop_training = False

    for epoch in range(1, args.epochs + 1):
        print(f"\nEPOCH {epoch}/{args.epochs}")
        print("=" * 72)

        model.train()
        aggregator.train()
        head.train()

        epoch_df = train_df.sample(
            frac=1.0,
            random_state=args.seed + epoch,
        ).reset_index(drop=True)

        running_loss = 0.0
        successful = 0

        for idx, row in enumerate(
            epoch_df.itertuples(index=False),
            start=1,
        ):
            if (
                args.max_train_steps > 0
                and optimizer_step >= args.max_train_steps
            ):
                stop_training = True
                break

            t0 = time.time()

            try:
                frames = frame_cache.build(row)

                target = torch.tensor(
                    [[
                        float(row.WD_P_mean),
                        float(row.CF_P_mean),
                    ]],
                    device=device,
                    dtype=torch.float32,
                )

                pred, attention_weights = predict_temporal_scores(
                    model=model,
                    aggregator=aggregator,
                    head=head,
                    processor=processor,
                    frames=frames,
                    device=device,
                    num_windows=args.window_count,
                    frames_per_window=args.frames_per_window,
                    pooling=args.pooling,
                )

                per_target = F.smooth_l1_loss(
                    pred.float(),
                    target,
                    beta=args.huber_beta,
                    reduction="none",
                )

                loss = per_target.mean()
                scaled_loss = loss / args.grad_accum_steps
                scaled_loss.backward()

                accum_counter += 1
                successful += 1
                running_loss += float(
                    loss.detach().cpu()
                )

                should_step = (
                    accum_counter >= args.grad_accum_steps
                    or idx == len(epoch_df)
                )

                if should_step:
                    trainable_model_params = [
                        p for p in model.parameters()
                        if p.requires_grad
                    ]

                    torch.nn.utils.clip_grad_norm_(
                        trainable_model_params
                        + list(aggregator.parameters())
                        + list(head.parameters()),
                        max_norm=args.max_grad_norm,
                    )

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    optimizer_step += 1
                    accum_counter = 0

                pred_np = (
                    pred.detach()
                    .float()
                    .cpu()
                    .numpy()[0]
                )

                weights_np = (
                    attention_weights.detach()
                    .float()
                    .cpu()
                    .numpy()[0]
                )

                weight_text = ",".join(
                    f"{x:.2f}"
                    for x in weights_np
                )

                print(
                    f"  epoch={epoch} "
                    f"sample={idx:03d}/{len(epoch_df):03d} "
                    f"opt_step={optimizer_step:03d} "
                    f"loss={float(loss.detach()):.4f} "
                    f"WD={row.WD_P_mean:.1f}->{pred_np[0]:.2f} "
                    f"CF={row.CF_P_mean:.1f}->{pred_np[1]:.2f} "
                    f"W=[{weight_text}] "
                    f"time={time.time()-t0:.1f}s",
                    flush=True,
                )

            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                torch.cuda.empty_cache()

                print(
                    "\nCUDA OOM on sample "
                    f"{getattr(row, 'sample_id', '?')}. "
                    "Stop this run rather than increasing load.",
                    flush=True,
                )
                raise

            except Exception as exc:
                print(
                    f"  ERROR "
                    f"{getattr(row, 'sample_id', '?')}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0

            finally:
                gc.collect()
                torch.cuda.empty_cache()

        mean_loss = running_loss / max(successful, 1)

        print(
            f"\nEpoch {epoch} mean training loss: "
            f"{mean_loss:.4f}"
        )

        val_metrics = evaluate_split(
            split_df=val_df,
            split_name="val",
            model=model,
            aggregator=aggregator,
            head=head,
            processor=processor,
            frame_cache=frame_cache,
            device=device,
            args=args,
            output_dir=output_dir,
            max_examples=args.max_val_examples,
        )

        val_mae = (
            val_metrics.get(
                "WD_P_MAE",
                float("inf"),
            )
            + val_metrics.get(
                "CF_P_MAE",
                float("inf"),
            )
        ) / 2.0

        checkpoint_dir = (
            output_dir
            / f"checkpoint_epoch_{epoch}"
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.save_pretrained(
            checkpoint_dir / "adapter"
        )
        processor.save_pretrained(
            checkpoint_dir / "processor"
        )
        torch.save(
            head.state_dict(),
            checkpoint_dir / "rupture_head.pt",
        )
        torch.save(
            aggregator.state_dict(),
            checkpoint_dir / "temporal_aggregator.pt",
        )

        base.write_json(
            checkpoint_dir / "metrics.json",
            {
                "epoch": epoch,
                "optimizer_step": optimizer_step,
                "mean_train_loss": mean_loss,
                "val_metrics": val_metrics,
            },
        )

        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            best_optimizer_step = optimizer_step

            best_dir = output_dir / "best"
            best_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            model.save_pretrained(
                best_dir / "adapter"
            )
            processor.save_pretrained(
                best_dir / "processor"
            )
            torch.save(
                head.state_dict(),
                best_dir / "rupture_head.pt",
            )
            torch.save(
                aggregator.state_dict(),
                best_dir / "temporal_aggregator.pt",
            )

            base.write_json(
                best_dir / "metrics.json",
                {
                    "epoch": epoch,
                    "optimizer_step": optimizer_step,
                    "mean_train_loss": mean_loss,
                    "val_metrics": val_metrics,
                },
            )

            best_lora_state = {
                key: value.detach().cpu().clone()
                for key, value
                in get_peft_model_state_dict(model).items()
            }

            best_head_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }

            best_aggregator_state = {
                key: value.detach().cpu().clone()
                for key, value
                in aggregator.state_dict().items()
            }

            print(
                f"New best checkpoint: "
                f"epoch {epoch}, "
                f"mean val MAE={val_mae:.4f}",
                flush=True,
            )

        if stop_training:
            break

    print("\nRESTORING BEST VALIDATION CHECKPOINT")
    print("=" * 72)

    if (
        best_lora_state is None
        or best_head_state is None
        or best_aggregator_state is None
    ):
        raise RuntimeError(
            "No best validation checkpoint captured."
        )

    set_peft_model_state_dict(
        model,
        best_lora_state,
        adapter_name="default",
    )

    head.load_state_dict(best_head_state)
    aggregator.load_state_dict(
        best_aggregator_state
    )

    head.to(
        device=device,
        dtype=torch.float32,
    )
    aggregator.to(
        device=device,
        dtype=torch.float32,
    )

    model.eval()
    aggregator.eval()
    head.eval()

    print(
        f"Restored best checkpoint from "
        f"epoch {best_epoch} "
        f"(optimizer step {best_optimizer_step}, "
        f"mean val MAE={best_val:.4f}).",
        flush=True,
    )

    print("\nFINAL TEST EVALUATION")
    print("=" * 72)

    test_metrics = evaluate_split(
        split_df=test_df,
        split_name="test",
        model=model,
        aggregator=aggregator,
        head=head,
        processor=processor,
        frame_cache=frame_cache,
        device=device,
        args=args,
        output_dir=output_dir,
        max_examples=args.max_test_examples,
    )

    base.write_json(
        output_dir / "final_summary.json",
        {
            "architecture": "4_window_attention_MIL",
            "best_val_mean_mae": best_val,
            "best_epoch": best_epoch,
            "best_optimizer_step": best_optimizer_step,
            "test_checkpoint": "best_validation",
            "test_metrics": test_metrics,
        },
    )

    print("\nFINISHED")
    print(
        "Best adapter:",
        output_dir / "best" / "adapter",
    )
    print(
        "Best temporal aggregator:",
        output_dir / "best" / "temporal_aggregator.pt",
    )
    print(
        "Best head:",
        output_dir / "best" / "rupture_head.pt",
    )
    print(
        "Test predictions:",
        output_dir / "test_predictions.csv",
    )


def make_parser():
    p = base.make_parser()

    p.description = (
        "Qwen3-VL four-window temporal attention MIL rupture fine-tuning."
    )

    p.add_argument(
        "--window-count",
        type=int,
        default=4,
        help="Number of chronological windows per labelled minute.",
    )

    p.add_argument(
        "--frames-per-window",
        type=int,
        default=4,
        help="Frames in each temporal window.",
    )

    p.add_argument(
        "--temporal-attention-hidden",
        type=int,
        default=256,
        help="Hidden size of learned MIL attention scorer.",
    )

    p.add_argument(
        "--temporal-dropout",
        type=float,
        default=0.10,
        help="Dropout inside temporal attention aggregator.",
    )

    return p


def main():
    parser = make_parser()
    args = parser.parse_args()

    if args.max_val_examples <= 0:
        args.max_val_examples = None

    if args.max_test_examples <= 0:
        args.max_test_examples = None

    expected = (
        args.window_count
        * args.frames_per_window
    )

    if args.num_frames != expected:
        parser.error(
            f"--num-frames must equal "
            f"--window-count * --frames-per-window. "
            f"For the default experiment use "
            f"--num-frames 16 --window-count 4 "
            f"--frames-per-window 4."
        )

    print(
        "QWEN3-VL 4-WINDOW TEMPORAL MIL "
        "RUPTURE-SALIENCE PILOT"
    )
    print("=" * 72)
    print("Labels:", args.labels_csv)
    print("Video root:", args.video_root)
    print("Model:", args.model_id)
    print(
        "Temporal layout:",
        f"{args.window_count} windows x "
        f"{args.frames_per_window} frames",
    )
    print("Window pooling:", args.pooling)

    manifest = base.build_pilot_manifest(args)

    if args.prepare_only:
        print(
            "\nPREPARE-ONLY complete. "
            "No model was loaded."
        )
        return

    train(args, manifest)


if __name__ == "__main__":
    main()
