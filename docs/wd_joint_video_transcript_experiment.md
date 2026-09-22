# Joint video–transcript fusion

This experiment tests whether Qwen3-VL can use the video frames and the exact same-segment German transcript together for WD_P binary detection. It is a new supervised condition called **joint video–transcript fusion**. It is separate from the existing video-only outputs and from late probability fusion.

## Controlled setup

- Expanded shared cohort: 4,325 segments, 20 patients, 89 videos.
- Five existing patient-disjoint folds; each fold has the same segment IDs and explicit train/validation/test patients.
- Video input: the existing 16 chronological patient crops at 224 pixels.
- Transcript input: `transcript_text` from the same segment, retaining `T:` and `P:` speaker labels.
- Target: existing soft binary target `(I(rater1 >= 2) + I(rater2 >= 2)) / 2`; no labels, predictions, patient IDs, filenames, or fold assignments are inserted into model input.
- Backbone: `Qwen/Qwen3-VL-8B-Instruct`, 4-bit NF4 QLoRA, rank 4, alpha 8, dropout 0.05, one binary logit and sigmoid probability.
- Pooling: final non-padding assistant-start token. It follows all visual and transcript tokens, so its causal hidden state can attend to both modalities. Padding and target labels are excluded.
- Checkpoint selection: validation AUPRC. The probability threshold is selected on validation data only using the existing threshold grid and validation balanced accuracy tie rule, then frozen for the outer test set.

## Files

- Training implementation: `scripts/vlm/finetune_qwen3vl_wd_consensus_binary.py` (`--input-mode joint`). Existing video-only behavior remains the default.
- Dataset audit: `scripts/analysis/audit_wd_joint_video_transcript.py`.
- Modality and gradient smoke test: `scripts/vlm/check_wd_joint_multimodal.py`.
- Paired report: `scripts/analysis/report_wd_joint_video_transcript.py`.
- Background queue: `scripts/orchestration/run_wd_joint_video_transcript_queue.ps1` and `scripts/orchestration/start_wd_joint_video_transcript_background.ps1`.
- Output root: `output/wd_joint_video_transcript_expanded/`.

## Run on the GPU machine

From the repository root, after committing and pulling these files:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts/orchestration/start_wd_joint_video_transcript_background.ps1
```

The queue first runs the dataset audit and a representative forward/backward modality check. If those pass, it trains five fresh fold adapters and heads. It then writes matched metrics against the transcript LLM, video VLM, and equal 50/50 late fusion, including 10,000 paired patient bootstrap resamples.

For a smoke check only:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts/orchestration/start_wd_joint_video_transcript_background.ps1 -SmokeOnly
```

Monitor with:

```powershell
Get-Content output/wd_joint_video_transcript_expanded/queue.log -Tail 30 -Wait
```

The queue stops on the first failure and leaves the log and partial fold output. Rerunning it resumes completed folds only when their `final_summary.json` exists.

## Verified locally

The transferred expanded cohort passed the data audit: all folds contain 4,325 rows; 20 patients remain patient-disjoint; all transcripts have speaker labels; every sample has 16 cached frames; and the binary-consensus evaluation population is 3,026 rows. Python syntax and PowerShell parsing also pass.

The current workstation cannot run the Qwen3-VL smoke test: its installed Transformers build does not expose `Qwen3VLForConditionalGeneration`, and its visible GPU is a 6 GB RTX 2060. The queue therefore must be run in the compatible GPU environment with the model and `peft` installed. No joint training result is claimed until the queue completes.

## Interpretation after completion

Compare joint fusion with the exact matched equal late-fusion baseline. A numerical gain is exploratory because the cohort and labels have already been used for model development. The paired patient bootstrap describes uncertainty in saved predictions, not retraining variability. Report sensitivity and specificity together: a higher balanced accuracy can still hide a clinically undesirable false-positive or false-negative trade-off.
