# Qwen3-8B zero-shot transcript classification

The official post-trained checkpoint is `Qwen/Qwen3-8B`. This experiment uses
its non-thinking instruction-following mode, with the official `Q4_K_M` GGUF
quantization to fit the local 6 GB GPU with partial CPU offload. Quantization is
part of the experiment specification; this is not an unquantized BF16 run.

Official sources: [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B),
[Qwen GGUF weights](https://huggingface.co/Qwen/Qwen3-8B-GGUF).

```powershell
# One-time download; verifies upstream SHA-256 hashes.
python scripts/setup_qwen3_8b_local.py

# Run or resume the primary cohort.
python scripts/run_qwen3_8b_text_zeroshot.py

# Alternative: all ready sessions, kept in a separate experiment directory.
python scripts/run_qwen3_8b_text_zeroshot.py --scope all-ready --output output/qwen3_8b_text_zeroshot/all_ready
```

Default input: `data/amberscript_llm/llm_segments_all.jsonl`. The primary cohort
is `inventory-ready`: **1,538** ready rows from the original audio inventory.
`all-ready` selects **5,535** rows including additional sessions.
`inventory-nonempty` explicitly includes review rows with unknown speaker roles
and selects **2,971** rows; use a separate output for this sensitivity analysis.
Empty transcripts are never classified as negative examples.

The task is four-class patient alliance-rupture prediction using the existing
VLM vocabulary: `NO_RUPTURE`, `WD_P`, `CF_P`, `MIXED_P`. It uses a fixed definition
prompt and each timestamped segment, without demonstrations, retrieval, training,
human labels, or prior model predictions. Patient IDs and transcript provider are
retained with results but excluded from model messages. The exact prompt is in
the runner and saved to `system_prompt.txt`.

Generation uses non-thinking mode, temperature 0.7, top-p 0.8, top-k 20, and a
stable seed derived independently from each segment ID. A JSON schema constrains
the allowed label. Maximum output length is 32 tokens. The output contains the
label only, with no generated explanation or unsupported confidence score.
Backend/hardware differences can still affect reproducibility despite fixed seeds.

The local backend is pinned to llama.cpp `b10909`, using Vulkan with 25 GPU
layers, four CPU threads, and a 4,096-token context. It binds only to localhost
and requires an ephemeral local API key. No transcript is sent to an external
inference service. The runner owns and stops its server process on exit.

Results are saved under `output/qwen3_8b_text_zeroshot/inventory_ready/`:

- `predictions.csv`: current predictions, raw output, provider, timing, review
  status, label, and derived withdrawal/confrontation/any-rupture binary flags.
- `predictions.jsonl`: append-only checkpoint log; a rerun retries failed rows
  and skips successful ones. The CSV keeps the latest record per segment.
- `summary.json`: progress, label counts, and counts by transcript provider.
- `run_config.json`: exact cohort, prompt, sampling settings, dataset hash, model
  revision/checksum, and configuration hash. Changed settings require a new output.
- `selected_segments.csv`, `excluded_segments.csv`: explicit cohort selection.
- `server.log`: local model loading and inference diagnostics.

Invalid JSON, unknown labels, truncated generation, and runtime errors remain
errors; they are never converted to `NO_RUPTURE`. Three consecutive failures stop
the run for diagnosis. Context overflow is an error, not silent text truncation.
Each response is checkpointed before processing the next segment.

`--prepare-only` writes the selection and configuration without inference.
`--limit N` supports a smoke run in a separate output directory. Tests:

```powershell
python -m unittest discover -s scripts -p test_qwen3_8b_text_zeroshot.py
```

No ground-truth rupture labels were supplied, so no accuracy, F1, or agreement
metric is reported. Use `segment_uid` to join the predictions to the intended
human target labels and matched VLM evaluation cohort.
