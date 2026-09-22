#!/usr/bin/env python
"""Forward/backward and modality-use checks for joint video-transcript Qwen3-VL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

import finetune_qwen3vl_wd_consensus_binary as joint


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = joint.merge_targets_with_manifest(
        args.manifest, args.labels_csv, 2.0, args.output
    )
    candidates = manifest[manifest["split"] == "val"].head(2)
    if len(candidates) < 2:
        raise RuntimeError("Need two validation examples for modality perturbation checks.")

    model_args = argparse.Namespace(
        model_id=args.model_id,
        no_4bit=args.no_4bit,
        attn_implementation=args.attn_implementation,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        head_dropout=args.head_dropout,
    )
    model, processor, unused_head = joint.base.build_model_and_processor(model_args)
    del unused_head
    hidden_size = int(model.get_base_model().config.text_config.hidden_size)
    head = joint.BinaryHead(hidden_size, args.head_dropout).to("cuda", dtype=torch.float32)
    cropper = joint.base.PatientCropper(Path(args.yunet_model), args.frame_width)
    cache = joint.base.FrameCache(args.frame_cache, cropper, args.num_frames)
    device = torch.device("cuda:0")
    rows = list(candidates.itertuples(index=False))
    frames_a, frames_b = cache.build(rows[0]), cache.build(rows[1])

    def probability(frames, transcript, grad=False):
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            logits = joint.predict_logits(
                model, head, processor, frames, device, "last_token", 2.0,
                input_mode="joint", transcript_text=transcript,
                max_input_tokens=args.max_input_tokens,
            )
            return logits, float(torch.sigmoid(logits[0, 0]).detach().cpu())

    inputs = joint.prepare_inputs(
        processor, frames_a, device, 2.0, input_mode="joint",
        transcript_text=rows[0].transcript_text,
        max_input_tokens=args.max_input_tokens,
    )
    token_count = int(inputs["attention_mask"].sum().item())
    visual_keys = sorted(k for k in inputs if k.startswith("pixel_values"))
    if not visual_keys:
        raise RuntimeError("Processor output has no visual tensor.")

    _, p_original = probability(frames_a, rows[0].transcript_text)
    _, p_transcript_swap = probability(frames_a, rows[1].transcript_text)
    _, p_frame_swap = probability(frames_b, rows[0].transcript_text)

    model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    logits, _ = probability(frames_a, rows[0].transcript_text, grad=True)
    target = torch.tensor([[float(rows[0].WD_soft)]], device=device)
    F.binary_cross_entropy_with_logits(logits.float(), target).backward()
    adapter_grad = sum(
        int(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
        for p in model.parameters() if p.requires_grad
    )
    head_grad = sum(
        int(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
        for p in head.parameters() if p.requires_grad
    )

    report = {
        "experiment": "joint video-transcript fusion",
        "sample_id": str(rows[0].sample_id),
        "num_frames": len(frames_a),
        "processed_tokens": token_count,
        "visual_tensor_keys": visual_keys,
        "pooling": "last_token",
        "probability_original": p_original,
        "probability_transcript_swapped": p_transcript_swap,
        "probability_frames_swapped": p_frame_swap,
        "absolute_change_transcript": abs(p_original - p_transcript_swap),
        "absolute_change_frames": abs(p_original - p_frame_swap),
        "trainable_adapter_tensors_with_nonzero_gradient": adapter_grad,
        "head_tensors_with_nonzero_gradient": head_grad,
        "both_modalities_present": bool(visual_keys and token_count > 0),
        "checks_passed": bool(
            visual_keys and token_count > 0 and adapter_grad > 0 and head_grad > 0
            and p_original != p_transcript_swap and p_original != p_frame_swap
        ),
    }
    (args.output / "implementation_checks.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["checks_passed"]:
        raise SystemExit("One or more joint multimodal implementation checks failed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--frame-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-id", default=joint.DEFAULT_MODEL_ID)
    p.add_argument("--yunet-model", default="models/face_detection_yunet/face_detection_yunet_2026may.onnx")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--frame-width", type=int, default=224)
    p.add_argument("--max-input-tokens", type=int, default=32768)
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--head-dropout", type=float, default=0.10)
    p.add_argument("--no-4bit", action="store_true")
    main(p.parse_args())
