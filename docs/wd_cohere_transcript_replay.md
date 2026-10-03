# Cohere transcripts on the frozen expanded cohort

## Question

Does replacing the transcript source improve WD_P severity prediction while
keeping all 4,325 segments, 20 patients, human ratings and patient folds fixed?

The screenshot compares **ordinal** Qwen3-8B and Qwen3-14B, not the scalar
SmoothL1 regression experiment. Both use a five-class soft target constructed
from the two human ratings. Their output is the expected score on 1–5.

## Frozen setup

The ten original run configurations are stored in
`scripts/configs/wd_ordinal_expanded_original/`. The replay reads each fold's
configuration and uses its saved system prompt verbatim.

| Setting | Original and rerun |
|---|---|
| Models | Qwen/Qwen3-8B and Qwen/Qwen3-14B |
| Input | Prompt plus speaker-labelled, minute-level transcript |
| Objective | Soft ordinal cross-entropy |
| Model adaptation | NF4 QLoRA; rank 8, alpha 16, dropout 0.05 |
| Pooling | Last non-padding token |
| Patient balancing | Enabled |
| Epochs / seed | 3 / 42 |
| Learning rates | Backbone 3e-5; head 1e-4 |
| Maximum tokens | 2,048 |
| Batch / accumulation | 1 / 8 |
| Evaluation batch | 8B: 2; 14B: 1 |
| Rubric | Original saved manual_compact_v2 text |
| Selection | Validation MAE; outer test patients held out |

Only transcripts and their provenance fields change. Old datasets and outputs
remain available as the comparison baseline. The source configurations recorded
model revision `main`, not an immutable commit. Reuse the original machine's model
cache and Python environment; exact historical model/code equivalence cannot be
guaranteed from those configurations alone. The replay records the trainer hash.

## Release status

Requested object:
`obs://protect-ai/transcription_artifacts/datasets/v1/memopsy_196_dataset_versions_v1.tar.gz`

The object has not yet been downloaded in this workspace. No claims about this
release's coverage, timestamps, speaker attribution or ASR quality have been
verified. The earlier archive covered only 3,981 of the 4,325 original segments;
the new experiment must not silently drop the remaining segments.

## Download and inspect

On a machine with an authenticated OBS client:

```powershell
New-Item -ItemType Directory -Force artifacts\cohere_dataset_release | Out-Null
obsutil cp obs://protect-ai/transcription_artifacts/datasets/v1/memopsy_196_dataset_versions_v1.tar.gz artifacts/cohere_dataset_release/memopsy_196_dataset_versions_v1.tar.gz
python scripts/data/prepare_wd_cohere_replacement.py inspect `
  --archive artifacts/cohere_dataset_release/memopsy_196_dataset_versions_v1.tar.gz `
  --output artifacts/cohere_dataset_release/extracted
```

Alternatively use a temporary HTTPS download link. Do not commit transcripts,
audio, credentials or signed download links. Inspection reports file types and
table schemas without printing therapy transcript content.

## Construct the replacement

After inspecting the actual archive schema, export its aligned Cohere fine-tuned
transcripts into a segment-level JSONL or CSV. Required fields:

- `segment_uid`: exactly the original identifier.
- `transcript_text`: nonempty text with reliable patient/therapist roles.
- `start_sec`, `end_sec`: the original minute's rating boundaries.
- `transcript_provider`: `cohere_finetuned`.
- `timing_validated`, `speaker_roles_validated`: true only after actual validation.
- `asr_evaluation_role`: ASR train/validation/held-out provenance, when available.

These flags are declarations of completed checks, not instructions to mark
unverified data as ready. A full-session TXT cannot be split proportionally into
minutes. Arbitrary diarization speaker IDs cannot be assumed to be P/T.

```powershell
python scripts/data/prepare_wd_cohere_replacement.py build `
  --master-root output/wd_multimodal_master_expanded `
  --replacements artifacts/cohere_dataset_release/cohere_ft_segments.jsonl `
  --output output/wd_multimodal_master_expanded_cohere_ft
```

This requires every original segment and copies the old folds in their original
row order. Missing data or invalid readiness causes an error instead of a smaller
cohort. It validates patient disjointness and one outer-test appearance per segment.

## Run on the GPU machine

For the already extracted release, preparation and training can run sequentially
in the background with one command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts/orchestration/start_wd_cohere_ordinal_background.ps1 `
  -ReleaseRoot "C:\Data\Sequence_model\german-asr-pipeline\artifacts\memopsy_196_dataset_versions_v1"
```

The release scanner accepts explicitly identified `cohere_finetuned` segment
tables with original identifiers/boundaries and the validation fields described
above. It does not infer patient/therapist roles or timing from file names. If the
release uses another schema or contains only full-session text, preparation writes
`output/wd_multimodal_master_expanded_cohere_ft/release_inventory.json` and stops
before training. That inventory records table schemas without transcript content;
use it to adapt the loader to the actual release format. To resume an already
prepared experiment, omit `-ReleaseRoot`.

First copy the validated replacement data and pull the code. Plan without training:

```powershell
python scripts/orchestration/run_wd_cohere_ordinal_replay.py `
  --master-root output/wd_multimodal_master_expanded_cohere_ft
```

Then start sequential 14B and 8B runs in one hidden background process:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts/orchestration/start_wd_cohere_ordinal_background.ps1
```

Results appear under `output/wd_cohere_ordinal_replay/`, with separate
`llm_wd_ordinal_qwen3_14b_expanded_cohere_ft_cv` and
`llm_wd_ordinal_qwen3_8b_expanded_cohere_ft_cv` folders. The launcher prints logs.
Completed folds are skipped only when their dataset/config/trainer fingerprints
match. A forced shutdown can leave `queue.lock`; verify its PID before removing it.

## Analysis after completion

Compare old/new predictions on identical outer-test IDs: MAE, RMSE, Spearman,
prediction range/SD, error by human-rating pair, and patient-bootstrap paired error
differences. Keep continuous expected scores as the primary output. Report ASR
training/selection exposure separately; a fixed downstream patient split does not
make ASR-training patients independently held out. Lower WER is the hypothesis,
not a proven result of improved rupture detection.
