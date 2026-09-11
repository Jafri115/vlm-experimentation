# Qwen3-8B zero-shot classification on a 32 GB GPU

The intended runner is `scripts/run_qwen3_8b_zeroshot.py`. It loads the official
instruction-following `Qwen/Qwen3-8B` model in **BF16 without quantization** on one
CUDA GPU, with thinking disabled. Default batch size: 4. Default decoding:
greedy, fixed prompt, no examples, no training. No inference is run by importing
the module or reading this guide.

Install a CUDA-enabled PyTorch build for your machine, then:

```powershell
python -m pip install -r requirements_qwen3_text.txt
```

Run from the project root when you are ready:

```powershell
# Original inventory: 1,538 ready segments.
python scripts/run_qwen3_8b_zeroshot.py

# All ready sessions: 5,535 segments.
python scripts/run_qwen3_8b_zeroshot.py --scope all-ready --output output/qwen3_8b_bf16_zeroshot/all_ready

# Optional cohort inspection only; no model loading or inference.
python scripts/run_qwen3_8b_zeroshot.py --prepare-only
```

Copy the project scripts and `data/amberscript_llm/llm_segments_all.jsonl` to the
32 GB machine. The runner reads embedded transcript text, so old absolute audio
paths do not need to exist. Dependencies must be installed in the Python
environment you use for inference.

Labels: `NO_RUPTURE`, `WD_P`, `CF_P`, `MIXED_P`. The input is the timestamped
transcript; patient IDs and provider labels never enter the model prompt. Outputs
retain `segment_uid`, provider (`amberscript` / `voxtral`), review flags, raw model
response, and the predicted class and derived binary flags.

The default output directory is
`output/qwen3_8b_bf16_zeroshot/inventory_ready`. Results include `predictions.csv`,
an append-only `predictions.jsonl` checkpoint, `summary.json`, `run_config.json`,
`runtime.json`, and selection/exclusion CSVs. Run the same command to resume;
successful segment IDs are skipped. Changed input or settings require a new
output directory. Use `--revision COMMIT_SHA` to pin the exact model checkpoint;
the resolved revision is recorded and checked on resume.

`--scope inventory-nonempty` includes 2,971 nonempty inventory segments, including
review rows; use a separate output directory if deliberately evaluating that
cohort. No empty transcript receives a negative label.

The default batch size is conservative for a 32 GB GPU. If needed, set
`--batch-size 2`; if BF16 is unsupported, use `--dtype float16`. Inputs exceeding
`--max-input-tokens 4096` stop with an explicit error rather than being truncated.
Invalid or incomplete output is recorded as an error, never converted to
`NO_RUPTURE`. No accuracy metric is calculated without human labels.

`--temperature 0.7` enables sampling with top-p 0.8 / top-k 20; greedy decoding is
the default for the reproducible classification baseline. Sampling results can
change with batch composition on resume. Model or GPU/backend changes may also
affect results.

The earlier GGUF/Vulkan runner and its partial local predictions are separate
from this BF16 experiment. That local run has been stopped. This 32 GB runner has
not been used for inference here.

## Compare with human ground truth

Copy the completed BF16 `predictions.csv` back to the same relative output path
in this project, then run:

```powershell
python scripts/compare_qwen3_8b_to_ground_truth.py `
  --predictions output/qwen3_8b_bf16_zeroshot/inventory_ready/predictions.csv
```

The primary four-class target follows the existing VLM convention:
`mean(WD_P) > 1` and `mean(CF_P) > 1`. The comparison also reports a stricter
two-or-more-rater consensus analysis, binary rupture metrics, WD and CF
one-vs-rest metrics, confusion matrices, provider subgroups, and a row-level
file of errors. Results are written to
`output/qwen3_8b_ground_truth_comparison`.
