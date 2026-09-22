#!/usr/bin/env python
import importlib.util
import torch
import transformers
from transformers import AutoConfig, AutoProcessor, Qwen3VLMoeForConditionalGeneration

MODEL = "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"

print("Transformers:", transformers.__version__, flush=True)
print("Torch:", torch.__version__, flush=True)
print("GPTQModel installed:", bool(importlib.util.find_spec("gptqmodel")), flush=True)

if importlib.util.find_spec("gptqmodel") is None:
    raise RuntimeError("GPTQModel is required for this AWQ load test")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("GPU:", torch.cuda.get_device_name(0), flush=True)
print(
    "GPU total GiB:",
    torch.cuda.get_device_properties(0).total_memory / 1024**3,
    flush=True,
)

# Patch only the runtime config.  Do not modify the cached Hugging Face files.
config = AutoConfig.from_pretrained(MODEL)
qcfg = getattr(config, "quantization_config", None)
if qcfg is None:
    raise RuntimeError("Checkpoint has no AWQ quantization_config")

if isinstance(qcfg, dict):
    original = list(qcfg.get("modules_to_not_convert") or [])
    patched = list(original)
    for name in ("model.visual", "visual", "mlp.gate"):
        if name not in patched:
            patched.append(name)
    qcfg["modules_to_not_convert"] = patched
    config.quantization_config = qcfg
else:
    original = list(getattr(qcfg, "modules_to_not_convert", None) or [])
    patched = list(original)
    for name in ("model.visual", "visual", "mlp.gate"):
        if name not in patched:
            patched.append(name)
    qcfg.modules_to_not_convert = patched

print("Original AWQ skip list:", original, flush=True)
print("Patched runtime skip list:", patched, flush=True)
print("Loading cached pre-quantized AWQ checkpoint...", flush=True)

processor = AutoProcessor.from_pretrained(MODEL)
torch.cuda.empty_cache()

model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    MODEL,
    config=config,
    dtype=torch.float16,
    device_map={"": 0},
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
)
model.eval()

allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
free_b, total_b = torch.cuda.mem_get_info()

print("LOAD SUCCESS", flush=True)
print(f"Allocated: {allocated:.2f} GiB", flush=True)
print(f"Reserved:  {reserved:.2f} GiB", flush=True)
print(f"CUDA free: {free_b / 1024**3:.2f} GiB", flush=True)
print(f"CUDA total:{total_b / 1024**3:.2f} GiB", flush=True)
print("hf_device_map:", getattr(model, "hf_device_map", None), flush=True)

# Sanity check: visual tower must remain ordinary/full-precision modules.
visual = model.model.visual
awq_like = []
for name, module in visual.named_modules():
    cls = module.__class__.__name__.lower()
    if "awq" in cls or "quantlinear" in cls:
        awq_like.append((name, module.__class__.__name__))

print("AWQ-like modules inside visual tower:", len(awq_like), flush=True)
if awq_like:
    print("First unexpected visual AWQ modules:", awq_like[:5], flush=True)
    raise RuntimeError("Visual tower was unexpectedly converted to AWQ modules")

print("VISUAL SKIP CHECK: PASS", flush=True)
