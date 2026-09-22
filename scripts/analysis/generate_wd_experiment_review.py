from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "wd_experiment_review.xlsx"
OUT.parent.mkdir(parents=True, exist_ok=True)

NAVY = "17365D"
BLUE = "D9EAF7"
GREEN = "E2F0D9"
YELLOW = "FFF2CC"
GREY = "F2F2F2"
WHITE = "FFFFFF"
thin = Side(style="thin", color="D9E2F3")

def style_sheet(ws, widths):
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for col, width in widths.items():
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=thin)

def header(ws, row, labels):
    for col, value in enumerate(labels, 1):
        c = ws.cell(row, col, value)
        c.fill = PatternFill("solid", fgColor=NAVY)
        c.font = Font(color=WHITE, bold=True)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[row].height = 34

major = [
    ["16-patient zero-shot", "Qwen3-8B", "LLM / transcript", "16 patients; 1,734 binary-consensus test segments", "Prompt + timestamped German transcript", "No fine-tuning; pretrained Qwen representation used for classification", "One WD_P probability; positive if >=0.5", "BA 0.621; F1 0.732; AUROC 0.621", "Transcript contained useful binary signal without task-specific training."],
    ["16-patient zero-shot", "Qwen3-VL", "VLM / patient-only video", "Same 16 patients and segments", "Prompt + 16 sampled patient-only frames", "No fine-tuning; visual-language backbone used directly", "One WD_P probability; positive if >=0.5", "BA 0.498", "Video zero-shot was approximately chance on this cohort."],
    ["16-patient consensus fine-tuning", "Qwen3-8B", "LLM / transcript", "16 patients; 1,734 consensus test segments; five patient-disjoint folds", "Prompt + timestamped transcript", "4-bit QLoRA; LoRA rank 8; mean-token pooling; binary head; BCE", "Probability and hard binary WD_P decision", "BA 0.621; F1 0.728; AUROC 0.630", "Strongest paired binary result; specificity remained limited."],
    ["16-patient consensus fine-tuning", "Qwen3-VL", "VLM / video", "Same paired cohort and folds", "Prompt + 16 patient-only frames", "4-bit QLoRA; LoRA rank 4; multimodal pooling; binary head; BCE; pos_weight=1", "Probability and hard binary WD_P decision", "BA 0.574; F1 0.725; AUROC 0.569", "High sensitivity but many false positives."],
    ["16-patient standard regression", "Qwen3-8B", "LLM / transcript", "16 patients; 2,457 regression rows; five folds", "Prompt + transcript", "4-bit QLoRA; scalar regression head", "Continuous WD_P score", "RMSE 0.710; MAE 0.568; Spearman 0.225; range 1.00-2.24", "Predictions collapsed toward the lower range."],
    ["16-patient standard regression", "Qwen3-VL", "VLM / video", "Same paired regression rows", "Prompt + 16 patient-only frames", "4-bit QLoRA; scalar regression head", "Continuous WD_P score", "RMSE 0.695; MAE 0.583; Spearman 0.185; range 1.24-2.26", "Slightly lower RMSE than transcript, but still severe range collapse."],
    ["20-patient expanded standard regression", "Qwen3-8B", "LLM / transcript", "20 patients; 4,325 rows; five folds", "Prompt + transcript", "4-bit QLoRA; standard scalar regression head", "Continuous WD_P score", "RMSE 0.690; predicted range 1.00-2.74", "Lower error than the mean baseline, but only a small improvement."],
    ["20-patient expanded standard regression", "Qwen3-VL", "VLM / video", "Same 20-patient expanded cohort", "Prompt + 16 patient-only frames", "4-bit QLoRA; standard scalar regression head", "Continuous WD_P score", "RMSE 0.699; predicted range 1.18-2.90", "Similar to baseline; high scores remain underrepresented."],
    ["20-patient expanded binary fine-tuning", "Qwen3-8B", "LLM / transcript", "20 patients; 4,325 training rows; 3,026 consensus evaluation rows", "Prompt + transcript", "4-bit QLoRA; mean-token pooling; binary head; BCE", "WD_P probability and binary decision", "BA 0.612; TP 1,125; TN 742; FP 544; FN 615", "Above chance; better specificity than the video model."],
    ["20-patient expanded binary fine-tuning", "Qwen3-VL", "VLM / video", "Same expanded cohort and consensus evaluation", "Prompt + patient-only frames", "4-bit QLoRA; multimodal pooling; binary head; BCE", "WD_P probability and binary decision", "BA 0.596; TP 1,427; TN 479; FP 807; FN 313", "Above chance; higher sensitivity but many false positives."],
    ["20-patient ordinal regression", "Qwen3-8B", "LLM / transcript", "20 patients; 4,325 rows; five patient-disjoint folds", "Prompt + transcript", "4-bit QLoRA; soft ordinal cross-entropy; ordinal score head", "Expected 1-5 score", "RMSE 0.690; range about 1.02-2.39", "Ordinal objective helped only modestly; collapse persisted."],
    ["20-patient ordinal regression", "Qwen3-14B", "LLM / transcript", "Same expanded cohort and folds", "Prompt + transcript", "4-bit QLoRA; soft ordinal cross-entropy; ordinal score head", "Expected 1-5 score", "RMSE 0.681; range about 1.01-2.35", "Best regression RMSE, but model size did not recover high-severity ratings."],
]

wb = Workbook()
ws = wb.active
ws.title = "Readme"
ws.append(["WD_P experiment review workbook"])
ws["A1"].font = Font(size=16, bold=True, color=NAVY)
ws.merge_cells("A1:F1")
readme = [
    ["Purpose", "A compact record of the major Qwen3-VL and Qwen3-8B/14B experiments, using the strongest or most comparable version of each experiment rather than minor reruns."],
    ["How to use", "Start with Major_Experiments for methods and interpretation. Use Results_Summary for slide-ready metrics. Add future runs as new rows while preserving the cohort, fold, prompt, and objective fields."],
    ["Cohorts", "Repaired 16-patient cohort: 2,457 regression rows and 1,734 binary-consensus evaluation rows. Expanded cohort: 20 patients, 4,325 shared rows, 3,026 binary-consensus rows."],
    ["Binary target", "WD_P >= 2 is positive. Balanced accuracy is the average of sensitivity and specificity. Chance is 0.50."],
    ["Regression target", "Mean human WD_P rating on a 1-5 scale. RMSE and MAE are lower-is-better; Spearman measures rank ordering."],
    ["Reliability warning", "Human exact agreement on the expanded 1-5 task is about 52.8%, human ICC is 0.435, and AI-human ICC is much lower. High AC2 is partly driven by low-score skew."],
    ["Main conclusion", "Transcript models carried stronger binary signal than video in the paired 16-patient comparison. On the expanded cohort both modalities remained above chance, while severity regression stayed close to a mean baseline and collapsed toward low scores."],
    ["Source files", "output/wd_expanded_all_reliability/report.md; docs/wd_presentation_slide_explainer.md; output/wd_presentation_slides_2026-09-19/WD_experiment_presentation_slides.md"],
]
for row in readme: ws.append(row)
ws.column_dimensions["A"].width = 24; ws.column_dimensions["B"].width = 115
for c in ws[1]: c.fill = PatternFill("solid", fgColor=NAVY); c.font = Font(color=WHITE, bold=True)
for row in ws.iter_rows(min_row=2, max_col=2):
    row[0].font = Font(bold=True, color=NAVY); row[0].fill = PatternFill("solid", fgColor=BLUE)
    row[1].alignment = Alignment(vertical="top", wrap_text=True)
    ws.row_dimensions[row[0].row].height = 42
ws.sheet_view.showGridLines = False

ws = wb.create_sheet("Major_Experiments")
headers = ["Experiment", "Model", "Modality", "Data / evaluation", "Input", "Architecture / training", "Output", "Results", "Interpretation"]
header(ws, 1, headers)
for row in major: ws.append(row)
style_sheet(ws, {"A": 30, "B": 16, "C": 23, "D": 40, "E": 33, "F": 52, "G": 28, "H": 48, "I": 56})
for r in range(2, ws.max_row+1):
    ws.row_dimensions[r].height = 72
    if "ordinal" in str(ws.cell(r,1).value).lower():
        for c in ws[r]: c.fill = PatternFill("solid", fgColor=YELLOW)
    elif "binary" in str(ws.cell(r,1).value).lower() or "consensus" in str(ws.cell(r,1).value).lower():
        for c in ws[r]: c.fill = PatternFill("solid", fgColor=GREEN)

ws = wb.create_sheet("Results_Summary")
headers = ["Cohort", "Task", "Model", "N evaluated", "Balanced accuracy", "RMSE", "MAE", "Spearman", "F1", "AUROC", "TP", "TN", "FP", "FN", "Main reading"]
header(ws, 1, headers)
rows = [
    ["16-patient repaired", "Binary consensus", "Qwen3-VL", 1734, 0.574, None, None, None, 0.725, 0.569, 865, 212, 508, 149, "Sensitive but many false positives"],
    ["16-patient repaired", "Binary consensus", "Qwen3-8B", 1734, 0.621, None, None, None, 0.728, 0.630, 811, 318, 402, 203, "Best paired binary point estimate"],
    ["16-patient repaired", "Regression", "Qwen3-VL", 2457, None, 0.695, 0.583, 0.185, None, None, None, None, None, None, "Slightly lower RMSE; compressed range"],
    ["16-patient repaired", "Regression", "Qwen3-8B", 2457, None, 0.710, 0.568, 0.225, None, None, None, None, None, None, "Similar performance; compressed range"],
    ["20-patient expanded", "Binary consensus", "Qwen3-VL", 3026, 0.596, None, None, None, 0.718, 0.569, 1427, 479, 807, 313, "Higher sensitivity, lower specificity"],
    ["20-patient expanded", "Binary consensus", "Qwen3-8B", 3026, 0.612, None, None, None, 0.660, 0.630, 1125, 742, 544, 615, "More balanced errors"],
    ["20-patient expanded", "Regression", "Qwen3-VL", 4325, None, 0.699, None, None, None, None, None, None, None, None, "Near mean baseline"],
    ["20-patient expanded", "Regression", "Qwen3-8B", 4325, None, 0.690, None, None, None, None, None, None, None, None, "Best standard-regression RMSE"],
    ["20-patient expanded", "Ordinal regression", "Qwen3-8B", 4325, None, 0.690, None, None, None, None, None, None, None, None, "Soft ordinal loss did not remove collapse"],
    ["20-patient expanded", "Ordinal regression", "Qwen3-14B", 4325, None, 0.681, None, None, None, None, None, None, None, None, "Small improvement; high scores still missing"],
]
for row in rows: ws.append(row)
style_sheet(ws, {"A": 23, "B": 20, "C": 18, "D": 12, "E": 17, "F": 12, "G": 12, "H": 12, "I": 10, "J": 10, "K": 8, "L": 8, "M": 8, "N": 8, "O": 46})
for r in range(2, ws.max_row+1):
    ws.row_dimensions[r].height = 38
    for col in [5,6,7,8,9,10]: ws.cell(r,col).number_format = "0.000"
    for col in [4,11,12,13,14]: ws.cell(r,col).number_format = "#,##0"

ws = wb.create_sheet("Next_Experiments")
headers = ["Priority", "Experiment", "Question", "Keep fixed", "Change", "Primary metric", "Success criterion", "Status"]
header(ws, 1, headers)
pending = [
    [1, "Fast prompt/objective screening", "Does the manual-informed prompt improve severity ranking?", "Same frozen patient split and seed", "Prompt or objective only", "Validation MAE / Spearman", "Improvement over standard regression without range collapse", "Next"],
    [2, "High-severity review", "Are 3-4 ratings supported by transcript patterns absent from the current prompt?", "Same labels and cohort", "Manual review and targeted features", "High-score MAE and recall", "Predictions increase on held-out high-score segments", "Next"],
    [3, "Imbalance-aware severity training", "Can rare high scores be learned without false inflation?", "Patient-disjoint folds", "Fold-local weighting or oversampling", "RMSE, MAE, score-band recall", "Better 3+ recall with stable low-score error", "Next"],
    [4, "Late fusion", "Does transcript plus video add complementary signal?", "Same segments and folds", "Combine OOF probabilities/scores", "Paired balanced accuracy and RMSE", "Improves both modalities on the same held-out rows", "Next"],
    [5, "Temporal context", "Does context from prior minutes help severity?", "Patient-disjoint evaluation", "Add previous 2-3 segments", "RMSE, Spearman, onset detection", "Improved ranking without patient leakage", "Next"],
]
for row in pending: ws.append(row)
style_sheet(ws, {"A": 10, "B": 30, "C": 47, "D": 31, "E": 32, "F": 26, "G": 45, "H": 14})
for r in range(2, ws.max_row+1): ws.row_dimensions[r].height = 58

for ws in wb.worksheets:
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0

wb.save(OUT)
print(f"Created: {OUT}")
