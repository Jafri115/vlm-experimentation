# Timestamped Amberscript + Voxtral LLM dataset

The current dataset is in `data/amberscript_llm`. It combines the original
Amberscript exports with Voxtral transcripts, using the new complete-session
pyannote diarization for the Voxtral speaker assignments. No predictions have
been run.

## Start here

| File | Contents |
| --- | --- |
| `inventory_all_segments.csv` / `.jsonl` | All **2,989** original inventory rows, including review and empty rows. |
| `inventory_ready.csv` / `.jsonl` | **1,538** inventory rows passing text, timing, and speaker-role checks. |
| `llm_segments_all.csv` / `.jsonl` | **7,893** rows: inventory plus previously generated windows from additional Amberscript sessions. |
| `llm_segments_ready.csv` / `.jsonl` | **5,535** ready rows across both collections. |
| `amberscript_segments.csv`, `voxtral_segments.csv` | Provider-specific views of the complete dataset, including review rows. |
| `llm_requests.jsonl` | Model messages for ready rows only; retains provider and segment identifiers outside the messages. |
| `llm_requests_all_nonempty.jsonl` | All nonempty text inputs, including explicitly flagged review rows. Use only if the experiment intentionally includes these rows. |
| `segments/<segment_uid>.txt` | Timestamped transcript for each nonempty row. Select files through the current manifest. |
| `sessions/<patient>_S<session>.txt` | Timestamped dialogue from the selected dataset windows, ordered by time. These files are not guaranteed to cover every minute of a recording. |
| `session_utterances.jsonl` | Selected session dialogue with provider and segment ID retained per cue. |
| `dataset_summary.json` | Current coverage, source counts, and preparation rules. |

The inventory has **1,365 Amberscript** rows and **1,624 Voxtral** rows.
**2,971** inventory rows contain text; **18** have no usable text. Missing content
is never filled with invented dialogue. Existing ready Amberscript text is
preferred; pending inventory windows use Voxtral when available. If Voxtral is
empty but Amberscript contains text, the human export is retained for review.

## Provider labels and timestamps

Every row has `transcript_provider`, with value `amberscript` or `voxtral`.
`diarization_provider=pyannote` is separate: diarization does not change which
system produced the words. Source paths and source hashes retain provenance.

Every retained utterance uses the requested format:

```text
[01:02.3] T: Beispiel einer Frage.
[01:05.7] P: Beispiel einer Antwort.
```

These are **full-session timestamps**, rounded to tenths of a second; minutes
can exceed 59. Every cue keeps its own timestamp, including consecutive cues
from the same speaker. JSONL retains unrounded `start_sec` and `end_sec` and
`segment_local_start_sec` / `segment_local_end_sec`. `transcript_text` contains
the formatted dialogue; `transcript_text_plain` contains the same dialogue
without displayed timestamps.

Amberscript SRT supplies cue intervals. DOCX exports usually supply utterance end
timestamps; the previous exported endpoint supplies the start. One Word-named
file contains HTML; its timestamps are starts, and the following start supplies
the endpoint. The final HTML cue has unknown duration and is retained as a point.
These are cue-level approximations, not word-level alignment.

Amberscript cues are assigned to inventory windows by their midpoints, retaining
their original timestamps. A cue may cross a segment boundary or have a negative
segment-relative start; `boundary_crossing_cues` records that condition. Displayed
session times remain nonnegative. Voxtral starts and ends are relative to the
one-minute file and receive the inventory's `start_sec` offset. Invalid source
timing is flagged rather than silently corrected.

HTML tags, `<br>`, entities, embedded timestamp markers, editorial `CAVE` text,
and balanced parenthesized content are cleaned. Parentheses can span subtitle
cues. Unmatched opening parentheses trigger cleaning review rather than deleting
the rest of the session. Original source files are unchanged.

## Full-session diarization and T/P roles

The supplied run is:

```text
D:/ukhd-Research/01_Projects/german-asr-pipeline/artifacts/all_session_diarization/pyannote_0bd29f528561_no_audio
```

It contains usable diarization files for **all 60 inventory sessions**. Turn
files are checked against their metadata SHA-256 and configuration signatures.
The model is `pyannote/speaker-diarization-community-1`. Its exported speaker
inventory marks every speaker `UNASSIGNED`: diarization itself does not establish
therapist/patient identity.

Session role mapping therefore uses the following available evidence in order:

1. Explicit T/P entries in the new run's speaker inventory, if supplied later.
2. Previously QC-passed manual therapist seed intervals, rematched by time to the
   new clusters. Require at least 3 seconds of overlap and 80% purity; assign the
   other cluster as patient only for a two-speaker session.
3. Timestamp overlap with ready Amberscript T/P dialogue from the original build.
4. Timestamp overlap with older session role timelines, excluding intervals whose
   old diarization was disputed in manual review.
5. Session-wide consensus of provisional local role mappings with similarity
   margin at least 0.08, rematched by time to the new clusters.

Methods 3-5 require at least 30 seconds of evidence in total, at least 5 seconds
per new cluster, at least 75% overall agreement, and 65% agreement per cluster.
These are explicit preparation thresholds, not validated accuracy guarantees.
No old `SPEAKER_00` identity is copied directly into the new run. No rupture
annotations, model predictions, or generated completion labels are mapping evidence.
All inferred roles remain provisional and `role_mapping_is_ground_truth=false`.

Each Voxtral cue is matched to the new full-session turns by absolute time. A
speaker assignment requires at least 50% cue coverage and 80% overlap purity.
No nearest-speaker extrapolation or fallback to the ASR file's old T/P label is
used. Original ASR speaker IDs and labels are retained in JSONL for audit.

Six inventory sessions lack adequate role-mapping evidence. Other cues may have
ambiguous overlap even when their session mapping is available. These use
`[MM:SS.s] UNKNOWN: text` in the complete dataset and are excluded from ready
inputs. This avoids presenting a guessed T/P identity as resolved. Noncanonical
human labels such as `P2` or `I` are retained as original metadata and similarly
require evidence before being displayed as T/P.

| Audit file | Purpose |
| --- | --- |
| `pyannote_session_role_mapping.csv` | New cluster-to-role maps, evidence scores, provenance, and conflicts. |
| `pyannote_unresolved_inventory_sessions.csv` | The six inventory sessions needing additional role evidence. |
| `pyannote_completion_audit.csv` | Provider and status selected for each pending original inventory row. |
| `segments_needing_review.csv` | Non-ready rows and ready rows with preparation caveats. |
| `timestamped_validation.json` | Validation counts and checks from the current build. |

`READY` means the implemented checks passed, not that ASR words, role labels,
or audiovisual offsets are ground truth. Keep provider and quality indicators
when comparing human-edited and automatic transcripts.

## Rebuilding and using the dataset

From the repository root:

```powershell
python scripts/build_timestamped_llm_dataset.py
python -m unittest discover -s scripts -p 'test_*llm_dataset*.py'
```

The builder uses the original Amberscript snapshot at
`data/amberscript_llm/before_memopsy_completion/llm_segments_all.jsonl`, the ASR
artifacts, and the supplied pyannote run. This prevents previous automatic
completion results from becoming evidence for their own role decisions.
Arguments: `--dataset`, `--artifacts`, and `--diarization`. If the diarization
root has multiple runs, select a specific run directory explicitly.

For a completely fresh output directory, first run
`build_amberscript_llm_dataset.py --output PATH`, preserve its
`llm_segments_all.jsonl` under `PATH/before_memopsy_completion/`, then run the
new timestamped builder with `--dataset PATH`. The original and legacy completion
scripts remain available for reproducibility; running them directly against the
current output overwrites its newer manifests. Use a separate output for those.

The previous completion's all-row JSONL and summary are preserved in
`before_pyannote_completion/`. Its records contain the previous text. Older
`memopsy_*` audit files describe the previous legacy-cache run; the current
results are the `pyannote_*` audit files and `dataset_summary.json`.

To compare with VLM results, join by `segment_uid`, or normalized patient/session
and exact interval. The generated `segment_idx` is not the VLM's existing
`eval_id`. No human rupture targets or train/test splits were added. Reuse the
same labels, target rule, and patient-grouped splits for a matched comparison.

Generated data is local and covered by the existing `data` gitignore rule.
The supplied audio paths are preserved verbatim and may refer to another machine;
they were not used as evidence that an audio file exists locally.
