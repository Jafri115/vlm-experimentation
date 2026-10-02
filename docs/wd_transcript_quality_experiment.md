# Does lower ASR word error improve WD_P detection?

## Dataset versions

Build with `python scripts/data/build_wd_transcript_variants.py`.

Outputs: `output/wd_transcript_variants_v1/`.

| Version | Transcript source | Purpose |
| --- | --- | --- |
| amberscript_cohere_ft | Amberscript where its archived scoring source exists; fine-tuned Cohere otherwise | Maximum reference coverage with explicit fallback |
| cohere_ft | Fine-tuned Cohere exclusively | Consistent ASR pipeline |
| cohere_base | Base Cohere exclusively, where archived | Control for ASR fine-tuning |

“Cohere” in versions 1 and 2 explicitly means **cohere_finetuned**. Cohere base is never substituted for it silently. Sources are selected by archive scoring-source provenance, not by whichever file has the lowest reported WER. Original wording and speaker labels are preserved in session exports. Archive file hashes are checked before extraction.

Each version includes:

- `session_transcripts.jsonl`: full-session text and its source/hash, timing status, ASR evaluation role, and session WER when available.
- `available_coverage/segment_labels.jsonl`: existing expanded-cohort labels for windows whose session source exists.
- `paired_control/segment_labels.jsonl`: the exact same segment set for all three versions, restricted to sessions having all three sources.
- `*/fold_N/alignment_pending.jsonl`: original patient-disjoint train/validation/test assignments, preserved without rerandomizing patients.

The root contains `alignment_requests.jsonl` and `build_summary.json`.

## Current build, 2 October 2026

The original expanded master has 4,325 segments and 20 patients. Only 82 of its 89 sessions appear in this 196-session transcription archive, giving 3,981 existing labeled segments with sources. Do not assume 196 sessions means 196 labeled training sessions.

| Version | Archived transcript sessions | Existing labeled segments with a source |
| --- | ---: | ---: |
| Mixed Amberscript/Cohere fine-tuned | 196 | 3,981 |
| All Cohere fine-tuned | 196 | 3,981 |
| All Cohere base | 82 | 3,981 |

The three-source controlled comparison has **2,624 segments from 55 sessions**. This is the appropriate matched cohort for comparing Amberscript, base Cohere, and fine-tuned Cohere without provider fallback in the reference arm.

## Alignment is required before training

Fine-tuned Cohere transcripts are full-session plain text. Receipts contain inference metadata rather than word timestamps. Base Cohere SRT/JSON generally supplies coarse five-minute chunks. Amberscript exports include cue/CSV timestamps, but the archive explicitly marks alignment and speaker identities as unvalidated.

The generated segment manifests contain **empty segment transcript fields and `llm_ready=false`**. They are staging artifacts, not trainable datasets. This prevents the entire session text from being attached to each one-minute label and prevents old Voxtral text from surviving the source replacement.

Next processing steps on the audio machine:

1. Force-align each source transcript to the original session audio, preserving source wording; flag omissions/hallucinations that cannot align. Do not distribute words uniformly over session duration or reuse another provider's word boundaries.
2. Assign the same validated session speaker timeline to every transcript version. Retain uncertainty rather than guessing patient versus therapist.
3. Slice aligned words into the original half-open minute windows `[start_sec, end_sec)`, using the same policy for every source.
4. Audit alignment failures and P/T identity, then form one common eligible segment set and retain the original fold assignments.
5. Only then write training-ready `master_manifest.jsonl` files. Save exclusions by source and reason.

No audio or validated word alignments are included in the provided archive, so this build cannot complete those steps locally.

### Align on the audio machine

Install `stable-ts` into the intended Python environment. FFmpeg must be on PATH, CUDA must work, and the GPU must be available. The default alignment model is multilingual Whisper `large-v3`; its weights may download on first use. Alignment uses supplied transcript words rather than generating a replacement transcript.

```powershell
python -m pip install stable-ts
python scripts/data/align_wd_transcript_variants.py --plan-only
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/orchestration/start_wd_transcript_alignment_background.ps1
```

If archived media paths are no longer valid, pass `--media-root C:\path\to\sessions` to Python or `-MediaRoot C:\path\to\sessions` to the background launcher. This searches for exact session filenames such as `401001_S1.mp4`; ambiguous matches require `--audio-map` / `-AudioMap` with a CSV containing `session_uid,audio_path`.

Output is `output/wd_transcript_variants_aligned_v1/`. It contains per-source word alignments, a plan, failure diagnostics, dataset summaries, and common eligible segment manifests with the original five fold assignments. Repeated runs reuse successful alignments only if transcript hash, audio hash, and model match. A lock prevents concurrent alignment workers; after a forced termination, check that its PID is no longer running before removing a stale lock.

For a pilot, use `--max-sessions 1` or launcher `-MaxSessions 1`. A pilot produces partial dataset outputs; rerun without the limit before training. Alignment failures are excluded identically across the source comparison and reported. Zero-duration, nonmonotonic, very long words or changed/missing words trigger review. Automatic checks cannot establish real timing accuracy or detect every hallucination; inspect examples at the start, middle and end of sessions before interpreting minute-level results.

The initial runnable LLM datasets deliberately use **speaker-free text across all sources**. `llm_ready=true` means the text can be consumed for this wording control after automated checks; `alignment_reviewed=false` remains explicit. P/T-aware datasets require a separate shared, validated diarization timeline. The alignment run does not produce that timeline or infer identities from transcript semantics.

## Experimental hypothesis and controls

Primary hypothesis: a reduction in transcript WER improves binary WD_P detection on held-out patients. Secondary hypothesis: it improves severity prediction.

Primary comparison: base Cohere versus fine-tuned Cohere on identical audio/labels/splits, architecture, prompt, objective, epochs, and model-selection rules. Amberscript is a reference condition on the common 55-session subset. Mixed fallback is a practical coverage condition; it is not by itself a clean test of the WER hypothesis.

Use the same text-only input format first, stripping speaker tags consistently from all three sources, to isolate wording quality. After reliable P/T mapping exists, a separate comparison can test the value of speaker roles. Hold punctuation normalization and context length constant and report truncation rates.

Amberscript-based WER measures agreement with an automatic reference; it does not establish true transcription accuracy. Archive reference hash changes are unverified, so audit changed references before interpreting scores. Session WER must not be described as measured minute-level WER.

## Avoid ASR training leakage

The metadata identifies 47 sessions from patients used for ASR training and 18 sessions from patients used for ASR model selection. The other 131 are marked `not_training_or_validation_patient`. This flag is retained as `strict_asr_held_out`; it does not mean the sample is held out from LLM training in every fold.

For an independent end-to-end test, outer LLM test patients must also be outside ASR training and selection. The frozen original folds remain available for comparability, but results on contaminated test patients must be separated and identified. Define the common eligible patient set before any new downstream tuning.

## Evaluation

Binary: balanced accuracy, sensitivity, specificity, AUROC, Brier score, confusion matrices, and paired patient-bootstrap differences. Use binary-consensus labels for headline hard-decision evaluation and all eligible soft targets for probabilistic evaluation when appropriate.

Severity: MAE, RMSE, Spearman, prediction variance/range, and errors for each human-rating pair. Keep the prediction rounding rule explicit.

To relate WER to rupture results, compute paired differences for the same segment or session; account for patient clustering. A correlation across unrelated sessions can reflect audio quality, patient differences, and label distribution. Preserve failed alignments and source exclusions in coverage reporting rather than improving apparent performance by dropping difficult examples silently.

Begin with one fixed development fold and unchanged hyperparameters. If an ASR source improves validation performance, run the same predefined five-fold comparison. Do not choose a transcription source or threshold using test labels.
