#!/usr/bin/env python
import importlib.util
import torch
import transformers
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

MODEL = "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"

print("Transformers:", transformers.__version__, flush=True)
print("Torch:", torch.__version__, flush=True)
print("GPTQModel installed:", bool(importlib.util.find_spec("gptqmodel")), flush=True)

if importlib.util.find_spec("gptqmodel") is None:
    raise RuntimeError(
        "GPTQModel is required by the current Transformers AWQ loader. "
        r"Install with: .\.venv\Scripts\python.exe -m pip install -U gptqmodel"
    )

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("GPU:", torch.cuda.get_device_name(0), flush=True)
print(
    "GPU total GiB:",
    torch.cuda.get_device_properties(0).total_memory / 1024**3,
    flush=True,
)
print("Loading pre-quantized AWQ checkpoint directly...", flush=True)

processor = AutoProcessor.from_pretrained(MODEL)

torch.cuda.empty_cache()

model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    MODEL,
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
