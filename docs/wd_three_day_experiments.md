# Three-day presentation experiments

Run from `C:\Data\Sequence_model\VLM_experiments` after committing and pulling the new scripts, tests, and this guide. Existing experiment outputs and checkpoints remain in their original folders. New work goes to `output/wd_presentation_3day`.

## Quick commands

Use the existing environment that successfully ran the experiments. The CPU steps require numpy, pandas, scipy, scikit-learn and matplotlib. The optional seed repeats require the existing CUDA, transformers, peft, and bitsandbytes environment. Do not upgrade the working GPU environment for this task.

Preflight only (does not fit a model or run inference):

```powershell
python scripts/run_wd_three_day_queue.py --include-vlm-swap --dry-run
```

Run the essential CPU experiments in the foreground:

```powershell
python scripts/run_wd_three_day_queue.py --include-vlm-swap
```

Or start them once in the background:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_wd_three_day_background.ps1 -IncludeVlmSwap
```

To run everything including controlled GPU training repeats, use this instead:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_wd_three_day_background.ps1 -IncludeVlmSwap -RunSeeds
```

Choose one launch command, not both. The queue has an OS lock that rejects a second queue using the same output root. This lock does not schedule around unrelated GPU jobs. The background launcher prints separate, timestamped stdout and stderr paths. It uses the repository's `.venv\Scripts\python.exe`; override with `-Python 'C:\path\to\python.exe'` if needed.

Monitor after startup:

```powershell
Get-Content output/wd_presentation_3day/queue.log -Tail 30 -Wait
```

If `queue.log` is not present yet, inspect the background output/error paths printed by the launcher. A PID alone is not confirmation of successful startup. Individual stage logs are in `output/wd_presentation_3day/logs`. The queue stops on failure. It never marks a failed stage completed.

## What runs, in order

1. TF-IDF logistic regression, five folds, on binary-consensus rows.
2. TF-IDF ridge regression, five folds, on all paired rows and mean human severity.
3. Three within-patient content-swap diagnostics for the soft-label LLM, plus VLM when requested.
4. Paired model comparisons, patient-bootstrap intervals and presentation figures.
5. Optional controlled LLM soft-label repeats: seeds 42, 43, 44, five folds each.
6. Refresh the report with all completed seed results.

The first report is available before the optional GPU work begins. The CPU baseline steps are recomputed on a rerun. Seed folds resume only when a completion marker matches the saved configuration, dataset hash and training-code hash, and the expected artifacts exist. An incomplete seed fold restarts; there is no mid-epoch resume. Changed completed settings require a new output directory.

## Baseline methods

All folds come from `output/wd_multimodal_master_repaired`. Validation checks patient-disjoint splits, frozen human ratings and exactly one held-out assignment per segment. The vectorizer is fitted only on training transcripts, using word unigrams/bigrams, min_df=2, at most 30,000 features and sublinear TF. It reads the same `transcript_text` supplied to the LLM, including therapist context.

Logistic regularization C is selected from 0.1, 1, 10 using validation Brier error. Ridge alpha is selected from 1, 10, 100 using validation MAE. Settings are fixed in the code before examining test results. No class weighting is used for the logistic comparator. No refit on validation data is performed after selection. Ridge outputs are clipped to 1-5 at validation and test, consistent with the neural regression output range. Binary threshold is fixed at 0.5.

The baseline report compares TF-IDF with the saved consensus LLM, consensus VLM posweight1, and regression runs specified in `scripts/wd_pairwise_reliability_specs.json`. The same successful IDs are used for each comparison. Missing files or mismatched outer folds cause failure. TF-IDF consensus is not represented as a soft-label training method.

## Content swaps without expensive new inference

The soft-label models predict each segment independently. For a fixed held-out checkpoint, feeding segment B instead of segment A produces f(B). That value is already saved in the OOF table. Reassigning B's saved output to A is therefore the cached equivalent of exchanging their inputs, conditional on the saved numerical outputs. This is explicitly reported as a saved-prediction diagnostic, not fresh inference or shuffled training.

Swaps use random cycles within each held-out patient and fold. A segment never receives itself, and each donor is used once per replicate. LLM and VLM use identical donor mappings when both are included. Singleton groups are excluded from both correct and swapped evaluation and listed separately. The mapping CSV records donor IDs and session IDs. Swaps may cross sessions within the same patient.

The script preserves target labels while exchanging only the corresponding model output. Correct-versus-swapped Brier difference is averaged over three fixed mappings; positive values favor correct inputs. Its interval resamples whole patients and conditions on these mappings and fitted models. Replicate variation is reported separately. The script does not claim an exact permutation-test p-value. Similar content/labels within a patient can reduce the performance drop, and cross-session swaps can reflect session-specific information, not only minute-level cues.

This diagnostic applies to the independent soft-label models; do not reuse it without review for models whose inputs include previous predictions, temporal state, or sample-dependent demonstrations.

## Controlled seed repeats

The previous training code seeded Python, NumPy, and DataLoader sampling but omitted PyTorch initialization/dropout seeding. This has been corrected. Consequently, use three fresh runs (42, 43, 44), not the old seed-42 run plus two new runs, for the controlled stability table. Historical predictions remain historical comparators.

Each repeat uses its original fold's `run_config.json`, changing only dataset/output paths and seed. The system prompt must match. CSV/JSONL splits and labels are cross-checked. Original epochs, learning rates, weighting, quantization, architecture and validation checkpoint selection are retained. PyTorch seeding improves control but does not promise bitwise-identical results across CUDA/library versions.

No existing adapter/head is needed to perform the CPU swap diagnostic. Seed repeats train new adapters and heads. They do not warm-start from the prior fine-tuned model. The combined report displays every seed, its mean and range; it does not select the best seed. Multiple seeds on the same cohort are not independent patient replications.

Preflight controlled repeats without training:

```powershell
python scripts/repeat_wd_llm_seeds.py --dry-run
```

## Main outputs for the presentation

`output/wd_presentation_3day/report/` contains:

- `presentation_report.md`: tables, methods, limitations and slide order.
- `model_comparison.csv` and `model_comparison.png`: TF-IDF, LLM, VLM and training-constant baselines.
- `paired_gains.csv`: paired patient-bootstrap error improvements. Positive means the named model beats the comparator.
- `content_swap.png`: correct versus swapped inputs.
- `seed_metrics.csv` and optional `seed_stability.png`: all completed controlled seeds.

The baseline OOF tables are in `baselines/consensus/` and `baselines/regression/`. Per-fold summaries retain validation scores and vocabulary sizes. Swap maps and metrics are in `content_swap/`. These exports contain segment identifiers and labels but not transcript text.

Binary report metrics use consensus rows. Swap Brier error uses all eligible soft-label rows, while its balanced accuracy uses their consensus subset. Therefore these N values differ deliberately. Confidence intervals are exploratory, conditional on the fitted models, clustered by patient, and not corrected for multiple comparisons. No test-set threshold tuning is performed.

## Individual stages

```powershell
python scripts/run_wd_tfidf_baselines.py
python scripts/run_wd_content_swap.py --vlm-predictions output/vlm_wd_soft_repaired_paired_cv/oof_predictions.csv
python scripts/report_wd_presentation.py
```

After optional seed training, regenerate the report with the same final command.

## Verification

```powershell
python -m unittest discover -s tests -p test_wd_three_day.py
```

The synthetic checks fit only tiny CPU baselines and verify no-self within-patient donor mappings, fold rejection, paired gains, report/figure generation, and duplicate-process locking. They do not download models or run GPU experiments.
