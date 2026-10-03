# Cohere transcript rerun on available original minutes

The user authorized excluding the seven sessions absent from the new release.
The prepared cohort contains **3,939 original minutes from all 20 patients**.

| Preparation outcome | Minutes |
|---|---:|
| Original expanded cohort | 4,325 |
| Session absent from Cohere release | 344 |
| Session present, but no valid timed words in the rating minute | 42 |
| Nonempty new inputs | 3,939 |

These counts come from the downloaded archive with SHA256
`584be82e0e343f819cdd2c44efea053665b80b10c57e43e5f9da0ad9fd4cb791`.

## Construction

`build_wd_cohere_available_subset.py` reads `master_v1/<session>/<session>.words.json`
and applies `timed_dialogue_v1/<session>/word_updates.json`. It checks that overlay
indices exist and wording is unchanged. A word belongs to a rating minute when
its timestamp midpoint is within that original half-open interval. Invalid or
absent timestamps are not synthesized. Text order follows the original word indices.

All remaining labels, rating intervals, segment IDs and train/validation/test
assignments are retained. The release's new patient split and its 625 unlabelled
rupture windows are not used. Each retained segment appears in exactly one outer
test fold. Missing/empty exclusions are based on input availability, not label value.

The output contains two datasets on **identical retained rows**:

- `output/wd_cohere_available_subset/cohere_new`: new transcripts.
- `output/wd_cohere_available_subset/original_subset`: original transcripts for a controlled rerun.

## Important interpretation

Timing and role mappings are **provisional**, not verified. The source roles are
retained as P/T where available; unknown roles are explicitly written as UNKNOWN.
Unknown words are not assigned to the patient by guesswork.

**3,501 minutes contain some unknown-role words; 793 have no word assigned P.**
Nonempty inputs are enabled for this exploratory run, while the verification
flags remain false. The cohort audit records these counts. Results test this
transcript/timing/role pipeline, not WER in isolation. ASR exposure may also limit
independent generalization claims.

## Start on the GPU machine

After committing and pulling the code, from `C:\Data\Sequence_model\VLM_experiments`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts/orchestration/start_wd_cohere_subset_background.ps1
```

The default release root is
`C:\Data\Sequence_model\german-asr-pipeline\artifacts\memopsy_196_dataset_versions_v1`.
The original cohort root defaults to `output/wd_multimodal_master_expanded`.
Preparation happens automatically on that computer if not already present.

The queue runs Qwen3-14B, then Qwen3-8B, with five folds each, original saved
prompts and ordinal settings (three epochs). It aggregates OOF predictions and
reports MAE/RMSE. New data and result folders are separate from earlier experiments.
The launcher prints its PID and stdout/stderr log locations.

For the strongest transcript-only comparison, also retrain the old transcript
control on the **same reduced training, validation and test rows**:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts/orchestration/start_wd_cohere_subset_background.ps1 `
  -WithOriginalControl
```

This adds another ten fold jobs after the new-transcript experiments. Merely
rescoring old saved predictions on the 3,939 test rows is a useful descriptive
comparison, but their original models trained on more rows. Do not attribute
that difference solely to transcription quality.

Results:

- `output/wd_cohere_available_queue/cohere_new/llm_wd_ordinal_qwen3_14b_cohere_available_new_cv`
- `output/wd_cohere_available_queue/cohere_new/llm_wd_ordinal_qwen3_8b_cohere_available_new_cv`
- Optional original-control outputs under `output/wd_cohere_available_queue/original_subset`.

## Local preparation and evidence

## Timestamped cues with bounded role filling

A separate prepared variant is stored at `output/wd_cohere_available_cues`.
It retains the same 3,939 IDs, labels and frozen folds. Input lines now look like:

```text
[10:25.1] P: konstant auch irgendwie immer noch so,
[10:28.5] P: dass es halt immer wieder hochkommt und mich
[10:31.0] P: belastet und so.
[10:32.2] T: Mhm
```

Those lines illustrate the format, using the example supplied by the user.
Times are absolute session times. The formatter uses release turn boundaries and
subdivides long turns into cues of at most 20 words / 6 seconds; these shorter
cues are display/input chunks, not newly asserted changes of speaker.

Unknown-role runs can inherit P/T only if **both original neighbouring roles
agree**, the run is at most three words and spans at most two seconds, each gap
is at most one second, word indices/timing are continuous, and raw diarization
does not contradict the assignment. No inference at session edges or across P/T
changes, untimed words, long gaps or conflicting raw speaker IDs. Inferred roles
never seed subsequent inference. Original source files are not edited.

Actual result: 3,341 words inferred across 1,536 retained minutes. Minutes with
some remaining UNKNOWN words decrease from 3,501 to 3,290. The 793 minutes with
no assigned P words remain unchanged. Source roles and all inferred labels remain
provisional, not manually verified. All repairs, including word indices and
bracketing indices, are logged in `speaker_role_repairs.json`.

Run this distinct variant on the GPU machine after committing/pulling the code:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts/orchestration/start_wd_cohere_subset_background.ps1 `
  -DatasetRoot output\wd_cohere_available_cues `
  -QueueRoot output\wd_cohere_cues_queue `
  -FillUnknownRoles `
  -TurnStyle timestamped_cues
```

Use separate output roots as shown. Adding cues and inferred roles changes input
format as well as the ASR transcript; keep the previous plain/no-fill variant as
an ablation if testing which change helped. Main model prompts and hyperparameters
remain the archived originals. The launcher stops if an existing prepared dataset
uses different role-fill/format settings.

Preparation summary: `output/wd_cohere_available_subset/preparation_summary.json`.
Exclusions: `output/wd_cohere_available_subset/excluded_segments.json`.
Plan: `output/wd_cohere_available_queue/cohere_new/replay_plan.json`.

These manifests were generated locally. The GPU training queue has not been
launched from this workstation; it must be started with the command above on
the GPU computer. No remote shell connection is configured.

## Extended contextual role repair

`output/wd_cohere_contextual_cues` contains the separate `contextual_v2` variant,
prepared before one-minute slicing. It uses only original P/T source labels as
anchors. Same-role islands may be up to 12 words/6 seconds. Sentence-tail and
sentence-prefix rules use punctuation/connectors plus close timing. These are
heuristics, not German grammatical parsing or an acoustic speaker recognizer.

Acknowledgments, untimed fragments, long gaps, raw-speaker conflicts and
overlapping contrary-role speech are not forced. All assigned labels retain
`roles_verified=false`, `role_source=context_inferred`, and a qualitative
`strong_context_rule` or `moderate_heuristic` support label. These labels are not
calibrated probabilities. Word-level repair records are in
`speaker_role_repairs.json`; unresolved fragments/reasons are in
`speaker_role_review_candidates.json`.

Prepared results: 6,895 inferred words in 2,186 retained minutes. Minutes with
unknown roles decrease to 3,093; 792 still have no assigned patient words. The
same 3,939 segment IDs, words, original human labels and patient folds are retained.

Of the twelve examples proposed by the user, seven were assigned automatically.
`Okay.` remains for review. Some other fragments sit next to untimed words or
within larger uncertain passages; their suggested attribution is plausible but
has not been established by the automatic rules. Examples are in `cue_preview.txt`.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts/orchestration/start_wd_cohere_subset_background.ps1 `
  -DatasetRoot output\wd_cohere_contextual_cues `
  -QueueRoot output\wd_cohere_contextual_queue `
  -FillUnknownRoles `
  -RoleFillPolicy contextual `
  -TurnStyle timestamped_cues
```

Prior exploratory inspection of examples does not establish test-set accuracy.
Compare this variant against no-fill/bounded controls without claiming automatic
inference is manually corrected ground truth.
