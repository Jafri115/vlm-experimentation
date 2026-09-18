# Expanded shared WD_P cohort

This builder starts from `data/completed_segments_merged.csv` and creates one
physical-segment cohort for both Qwen3-8B and Qwen3-VL. It requires exactly two
distinct WD_P raters, a nonempty transcript marked `llm_ready`, a resolved raw
video, a resolved patient side, and a successfully built 16-frame patient cache.
Any failure is removed from both modalities before folds are assigned.

The CPU audit currently finds 4,325 label/transcript-ready rows from 20 patients
and 89 videos, including 3,026 binary-consensus rows. This is an upper bound until
visual processing finishes. Audit artifacts are in
`output/wd_multimodal_master_expanded`.

On the GPU/video machine, first verify the three machine-specific inputs:

```powershell
Test-Path C:\Data\Sequence_model\Memopsy_videos\CONVERTED
Test-Path output\qwen3vl_visual_experiment_v5\patient_role_cache.json
Test-Path models\face_detection_yunet\face_detection_yunet_2026may.onnx
```

Then launch video resolution and frame construction once in the background:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts\start_wd_expanded_cohort_background.ps1
```

If the video root differs, pass `-VideoRoot`. Monitor the timestamped output and
error logs printed by the launcher. Rerunning reuses already completed frame-cache
entries. Do not train until `master_cohort_summary.json` exists.

The finalized files are:

- `paired_master_soft.csv` and `.jsonl`: common regression/soft-label cohort.
- `paired_master_consensus.csv` and `.jsonl`: unanimous binary subset.
- `frozen_rater_labels_long.csv`: labels expected by the legacy VLM trainer.
- `paired_cv_patient_assignments.csv`: one outer test fold per patient.
- `fold_1` through `fold_5`: identical LLM/VLM train, validation, and test splits.
- `frame_cache_16`: visual inputs for every retained segment.
- exclusion CSVs and `master_cohort_summary.json`: complete audit trail.

The final row count may be lower than 4,325 when a video is missing, patient side
is unresolved, or crop extraction fails. Those exclusions are deliberate: the LLM
must not retain a segment that the VLM cannot evaluate.

## Resolve a small number of missing patient sides

When `excluded_patient_side.csv` contains patients, generate a compact local HTML
review. It shows one representative video and three frames per patient:

```powershell
python scripts\wd_patient_side_review.py build
Start-Process output\wd_patient_side_review\index.html
```

Choose LEFT or RIGHT on every card and download `wd_patient_side_decisions.json`.
Apply it safely; the script first backs up the role cache:

```powershell
python scripts\wd_patient_side_review.py apply `
  --decisions "$env:USERPROFILE\Downloads\wd_patient_side_decisions.json"
```

Then rerun the cohort builder. Existing frames are reused and only newly eligible
segments are processed. The HTML is entirely local and uploads nothing.
