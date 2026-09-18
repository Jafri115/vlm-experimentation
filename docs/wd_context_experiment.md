# Transcript context experiment

Run on the GPU machine after pulling this commit. No old predictions are overwritten.
The question is whether preceding dialogue improves withdrawal prediction beyond a
TF-IDF baseline that receives exactly the same dialogue. Improvement is not guaranteed.

| Run | Input | Pooling | Training |
| --- | --- | --- | --- |
| TF-IDF control | Target segment | Word features | Consensus labels |
| TF-IDF context | Previous + target | Word features | Consensus labels |
| Qwen3-8B control | Target segment | Mean all tokens | Consensus labels, five folds |
| Qwen3-8B context | Previous + target | Mean all tokens | Consensus labels, five folds |
| Optional Qwen3-8B | Previous + target | Target patient tokens | Consensus labels, five folds |

The previous segment must be contiguous (up to one second timing tolerance), from
the same patient, session, and split, and present in the frozen paired cohort.
Missing context never removes a target. Context can have disagreeing human labels;
no human label is included in model input. Only the target is rated. Dialogue remains
in its original language. No future segment is included.

All runs reuse the repaired master folds. Both classifiers train on binary consensus
only and evaluate the same consensus targets. This is a binary experiment, not a
rerun of regression or soft-label training. TF-IDF vocabulary is fit on training
only; C is selected by validation Brier. Qwen keeps the saved consensus settings and
validation AUPRC checkpoint selection. All decisions use threshold 0.5.

Both fresh Qwen controls use seed 42, maximum 4096 tokens, evaluation batch size 1,
and the same target-only instruction. These deliberate changes mean the historical
control should not replace the new control. Inputs exceeding the budget fail during
CPU token preflight before GPU training; nothing is silently truncated. Increase
`--max-length` equally for all variants and use a new output folder if necessary.
Longer inputs may need more GPU memory. The optional patient mask excludes previous
patient text and therapist text from pooling, while retaining them as attention
context. Rows without patient text fall back to the whole target and are counted.

## Commands on the GPU machine

From `C:\Data\Sequence_model\VLM_experiments`, after `git pull`:

```powershell
python scripts/run_wd_context_experiment.py --prepare-only
```

This fits CPU baselines and exports up to 30 validation errors to
`output/wd_context_experiment/validation_review/fold_1_review.csv`.
Review speaker errors, missing context, ordinary short answers, and annotation
ambiguity. This is fold 1 validation only; those patients can be test patients in
other folds, so any later design change is exploratory, not an untouched test.

Launch the two main Qwen variants sequentially in the background:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_wd_context_background.ps1
```

To also include the optional patient-token pooling experiment, use this command
instead (or run it after the main queue finishes; completed main jobs are skipped):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_wd_context_background.ps1 -PatientPooling
```

No waiting scheduler is used. A file lock rejects a second queue using the same
output directory. Ensure other GPU jobs have finished. Failures stop the queue;
rerunning resumes completed folds only when input/config/code fingerprints match.
Changes require a new `--output` folder. The existing virtual environment needs the
same PyTorch, Transformers, PEFT, bitsandbytes, pandas, scikit-learn and matplotlib
dependencies as the previous experiments; this code does not upgrade them.

```powershell
Get-Content output/wd_context_experiment/queue.log -Tail 30 -Wait
```

Outputs: `report.md`, `metrics.csv`, `comparisons.csv`, `comparison.png`,
`context_coverage.csv`, `token_preflight.csv`, validation review, and per-fold model
artifacts including `training_history.csv`. The report tests context vs control
and context LLM vs context TF-IDF using paired patient-bootstrap Brier intervals.
Balanced accuracy and AUROC are also shown. A positive Brier gain whose interval
excludes zero is evidence for improvement under this exploratory protocol; a higher
point estimate alone does not establish superiority. Do not choose variants or
thresholds by inspecting test results.
