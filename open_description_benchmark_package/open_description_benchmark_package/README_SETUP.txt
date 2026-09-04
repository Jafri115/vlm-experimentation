Open Description Benchmark package

Extract this ZIP directly into:
C:\Data\Sequence_model\VLM_experiments\

The ZIP adds:
  reference_human_verified_chatgpt.csv
  run_open_description_4models.ps1
  evaluate_open_descriptions.ps1
  scripts\run_open_description_benchmark.py
  scripts\compare_open_descriptions.py

It expects these existing helper scripts to already be in your project scripts folder:
  scripts\run_qwen3vl_visual_experiment_v5.py
  scripts\run_minicpm_v45_behavior_benchmark.py
  scripts\run_molmo2_8b_behavior_benchmark.py
  scripts\run_internvl35_14b_behavior_benchmark.py

These helper scripts are not duplicated in this ZIP because they are your existing working model/preprocessing runners.

After extraction, run:
  .\run_open_description_4models.ps1

Then evaluate:
  .\evaluate_open_descriptions.ps1
