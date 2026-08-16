#!/usr/bin/env python
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLMoeForConditionalGeneration

MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("Total GiB:", torch.cuda.get_device_properties(0).total_memory / 1024**3, flush=True)
print("Loading Qwen3-VL-30B-A3B as NF4 4-bit, ALL on CUDA:0", flush=True)

qconf = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(MODEL)
torch.cuda.empty_cache()
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    MODEL,
    quantization_config=qconf,
    dtype=dtype,
    device_map={"": 0},
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
)
model.eval()

allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
free_b, total_b = torch.cuda.mem_get_info()
print(f"LOAD SUCCESS", flush=True)
print(f"Allocated: {allocated:.2f} GiB", flush=True)
print(f"Reserved:  {reserved:.2f} GiB", flush=True)
print(f"CUDA free: {free_b / 1024**3:.2f} GiB", flush=True)
print(f"CUDA total:{total_b / 1024**3:.2f} GiB", flush=True)
print("hf_device_map:", getattr(model, "hf_device_map", None), flush=True)
