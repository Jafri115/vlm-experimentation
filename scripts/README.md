# Scripts

All active experiment entry points are organized into the folders below.
`scripts/` itself contains only this index; category bootstraps preserve direct
script execution and package imports.

## Categories

| Folder | Type | Purpose |
|---|---|---|
| `llm/` | LLM experiments | Transcript-based Qwen and future text-model runs. |
| `vlm/` | VLM experiments | Video/frame-based Qwen3-VL and multimodal model runs. |
| `data/` | Data preparation | Dataset, manifest, transcript, cache, and cohort construction. |
| `analysis/` | Analysis and reports | Evaluation, comparison, reliability, plotting, and report generation, including the WD review workbook generator. |
| `orchestration/` | Queues and operations | PowerShell queues, background launchers, and experiment operations. |
| `tests/` | Tests and diagnostics | Unit tests, load checks, and focused diagnostics. |
| `archive/` | Archived material | Legacy scripts and backups that are not part of the main workflow. |

## Main workflow

- Transcript LLM training: `llm/finetune_qwen3_8b_wd_text.py`
- Transcript LLM context helper: `llm/wd_context_inputs.py`
- Transcript LLM queues: `orchestration/run_wd_better_llm_regression_queue.ps1`, `orchestration/run_wd_expanded_full_queue.ps1`
- VLM training: `vlm/finetune_qwen3vl_wd_*.py`
- VLM inference: `vlm/run_qwen3vl_*.py`
- Reporting: `analysis/report_*.py`, `analysis/compile_*.py`, and `analysis/plot_*.py`
- Tests: `tests/test_*.py`

## Relocated active scripts

The migration moved the active files into these groups:

- `llm/finetune_qwen3_8b_wd_text.py`: transcript LLM training and evaluation
- `llm/wd_context_inputs.py`: previous-context input construction
- `tests/test_*.py`: focused data, LLM, Qwen, and model-load tests
- `analysis/generate_wd_experiment_review.py`: creates `output/wd_experiment_review.xlsx`
- `analysis/report_*.py`, `analysis/plot_*.py`, and `analysis/compare_*.py`: evaluation and reporting utilities
- `data/build_*.py`: dataset, manifest, cohort, and cache construction
- `vlm/finetune_qwen3vl_*.py` and `vlm/run_qwen3vl_*.py`: multimodal training and inference
- `orchestration/run_*.ps1`, `orchestration/start_*.ps1`, and collection queues: repeatable execution

## Archive

Archived files are under:

- `archive/legacy_qwen_visual_rag/`: superseded Qwen visual-RAG prototypes
- `archive/backups/`: generated `.bak` copies
- `archive/superseded_visual/`: superseded Qwen visual versions and visual pilot branches
- `archive/other_model_experiments/`: InternVL, LLaVA-Video, MiniCPM, Molmo, and related branches
- `archive/diagnostics_and_patches/`: diagnostics and one-off patch scripts
- `archive/metadata/`: generated prompt inventories

Do not place active experiment dependencies in `archive/`.