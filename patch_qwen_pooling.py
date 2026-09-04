#!/usr/bin/env python
from pathlib import Path
import py_compile
import shutil

TARGET = Path(r".\scripts\finetune_qwen3vl_rupture_pilot.py")

if not TARGET.exists():
    raise FileNotFoundError(f"Target script not found: {TARGET.resolve()}")

text = TARGET.read_text(encoding="utf-8")

if 'def predict_scores(' not in text or 'def make_parser()' not in text:
    raise RuntimeError("Target file does not look like the expected Qwen rupture pilot script.")

backup = TARGET.with_suffix(".py.bak_before_pooling")
shutil.copy2(TARGET, backup)

start = text.index("def predict_scores(")
end = text.index(
    "\n\n# ---------------------------------------------------------------------------\n# Metrics",
    start,
)

replacement = '''def pool_hidden_state(
    model,
    inputs,
    hidden: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """
    Pool Qwen3-VL hidden states into one representation per labelled minute.

    Supported:
      final_token : original final non-padding token
      mean_all    : mean over all non-padding multimodal token positions
      mean_image  : mean over Qwen image-token positions only
    """
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
                "mean_image pooling requires input_ids from the processor."
            )

        base_model = (
            model.get_base_model()
            if hasattr(model, "get_base_model")
            else model
        )

        image_token_id = None

        for obj in (
            getattr(base_model, "config", None),
            getattr(getattr(base_model, "model", None), "config", None),
        ):
            if obj is not None and hasattr(obj, "image_token_id"):
                image_token_id = getattr(obj, "image_token_id")
                if image_token_id is not None:
                    break

        if image_token_id is None:
            raise RuntimeError(
                "Could not determine Qwen3-VL image_token_id "
                "for mean_image pooling."
            )

        image_mask = (input_ids == int(image_token_id)) & valid_mask
        image_counts = image_mask.sum(dim=1)

        if torch.any(image_counts == 0):
            raise RuntimeError(
                "mean_image pooling found zero image-token positions "
                "for at least one sample."
            )

        weights = image_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return summed / denom

    raise ValueError(f"Unsupported pooling mode: {pooling!r}")


def predict_scores(
    model,
    head,
    processor,
    frames,
    device,
    pooling: str = "final_token",
):
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

    hidden = outputs.last_hidden_state

    pooled = pool_hidden_state(
        model=model,
        inputs=inputs,
        hidden=hidden,
        pooling=pooling,
    )

    return head(pooled)
'''

text = text[:start] + replacement + text[end:]

old = '''    output_dir: Path,
    max_examples: Optional[int] = None,
):'''
new = '''    output_dir: Path,
    max_examples: Optional[int] = None,
    pooling: str = "final_token",
):'''
if old not in text:
    raise RuntimeError("Could not find evaluate_split signature.")
text = text.replace(old, new, 1)

eval_start = text.index("def evaluate_split(")
old = '''                frames,
                device,
            )[0].float().cpu().numpy()'''
new = '''                frames,
                device,
                pooling=pooling,
            )[0].float().cpu().numpy()'''
idx = text.find(old, eval_start)
if idx == -1:
    raise RuntimeError("Could not find evaluation predict_scores call.")
text = text[:idx] + text[idx:].replace(old, new, 1)

train_start = text.index("def train(")
old = '''                    frames,
                    device,
                )'''
new = '''                    frames,
                    device,
                    pooling=args.pooling,
                )'''
idx = text.find(old, train_start)
if idx == -1:
    raise RuntimeError("Could not find training predict_scores call.")
text = text[:idx] + text[idx:].replace(old, new, 1)

old = '''            output_dir=output_dir,
            max_examples=args.max_val_examples,
        )'''
new = '''            output_dir=output_dir,
            max_examples=args.max_val_examples,
            pooling=args.pooling,
        )'''
if old not in text:
    raise RuntimeError("Could not find validation evaluate_split call.")
text = text.replace(old, new, 1)

old = '''        output_dir=output_dir,
        max_examples=args.max_test_examples,
    )'''
new = '''        output_dir=output_dir,
        max_examples=args.max_test_examples,
        pooling=args.pooling,
    )'''
if old not in text:
    raise RuntimeError("Could not find test evaluate_split call.")
text = text.replace(old, new, 1)

old = '    print("Frames / labelled minute:", args.num_frames)\n'
new = (
    '    print("Frames / labelled minute:", args.num_frames)\n'
    '    print("Pooling:", args.pooling)\n'
)
if old not in text:
    raise RuntimeError("Could not find training-plan frame print.")
text = text.replace(old, new, 1)

old = '    p.add_argument("--frame-width", type=int, default=320)\n'
new = '''    p.add_argument("--frame-width", type=int, default=320)
    p.add_argument(
        "--pooling",
        default="final_token",
        choices=["final_token", "mean_all", "mean_image"],
        help=(
            "Pooling strategy for Qwen3-VL hidden states before the "
            "rupture regression head."
        ),
    )
'''
if old not in text:
    raise RuntimeError("Could not find --frame-width parser line.")
text = text.replace(old, new, 1)

TARGET.write_text(text, encoding="utf-8")
py_compile.compile(str(TARGET), doraise=True)

required = [
    '"--pooling"',
    '"mean_image"',
    "pooling=args.pooling",
    'print("Pooling:", args.pooling)',
    "def pool_hidden_state(",
]
missing = [x for x in required if x not in text]
if missing:
    raise RuntimeError(f"Patch validation failed; missing: {missing}")

print("PATCH SUCCESS")
print(f"Updated: {TARGET.resolve()}")
print(f"Backup : {backup.resolve()}")
print("Syntax check: PASSED")
print("Pooling CLI: final_token / mean_all / mean_image")