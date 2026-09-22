from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out_path = root / 'output' / 'experiment_tracker_4weeks.xlsx'
out_path.parent.mkdir(parents=True, exist_ok=True)

completed = [
    {
        'Time_order': 'W1',
        'Date_window': 'Week 1',
        'Experiment': 'Qwen3-8B zero-shot WD_P',
        'Modality': 'LLM transcript',
        'Target': 'WD_P >= 2, repaired common cohort',
        'Data': '1,734 / 1,734 rows',
        'Result_summary': 'Balanced accuracy 0.621; F1 0.732; precision 0.668; recall 0.811; specificity 0.432',
        'Status': 'Completed'
    },
    {
        'Time_order': 'W1',
        'Date_window': 'Week 1',
        'Experiment': 'Qwen3-VL zero-shot WD_P',
        'Modality': 'VLM patient-only video',
        'Target': 'WD_P >= 2, repaired common cohort',
        'Data': '1,734 / 1,734 rows',
        'Result_summary': 'Balanced accuracy 0.498; F1 0.002; precision 0.250; recall 0.001; specificity 0.996',
        'Status': 'Completed'
    },
    {
        'Time_order': 'W1',
        'Date_window': 'Week 1',
        'Experiment': 'Qwen3-8B 3+3 few-shot WD_P',
        'Modality': 'LLM transcript',
        'Target': 'WD_P >= 2, repaired common cohort',
        'Data': '1,720 / 1,734 successful',
        'Result_summary': 'Balanced accuracy 0.596; F1 0.648; precision 0.673; recall 0.625; specificity 0.567',
        'Status': 'Completed'
    },
    {
        'Time_order': 'W1',
        'Date_window': 'Week 1',
        'Experiment': 'Qwen3-VL 3+3 few-shot WD_P',
        'Modality': 'VLM patient-only video',
        'Target': 'WD_P >= 2, repaired common cohort',
        'Data': '1,734 / 1,734 rows',
        'Result_summary': 'Balanced accuracy 0.417; F1 0.503; precision 0.513; recall 0.493; specificity 0.340',
        'Status': 'Completed'
    },
    {
        'Time_order': 'W2',
        'Date_window': 'Week 2',
        'Experiment': 'Continuous WD_P regression',
        'Modality': 'LLM transcript + VLM video',
        'Target': 'Continuous WD_P mean regression',
        'Data': 'Five outer folds; 2,457 evaluation rows',
        'Result_summary': 'LLM MAE 0.568, RMSE 0.710, Spearman 0.225; VLM MAE 0.583, RMSE 0.695, Spearman 0.185',
        'Status': 'Completed'
    },
    {
        'Time_order': 'W3',
        'Date_window': 'Week 3',
        'Experiment': 'Consensus binary fine-tuning',
        'Modality': 'LLM transcript + VLM video',
        'Target': 'Consensus WD_P binary target',
        'Data': 'Five outer folds; 1,734 evaluation rows',
        'Result_summary': 'LLM balanced accuracy 0.621, F1 0.728, AUPRC 0.640, AUROC 0.630; VLM balanced accuracy 0.574, F1 0.725, AUPRC 0.628, AUROC 0.569',
        'Status': 'Completed'
    },
    {
        'Time_order': 'W4',
        'Date_window': 'Week 4',
        'Experiment': 'Soft-label fine-tuning',
        'Modality': 'LLM transcript + VLM video',
        'Target': 'Soft WD_P label (0 / 0.5 / 1)',
        'Data': 'Five outer folds; 1,734 evaluation rows',
        'Result_summary': 'LLM balanced accuracy 0.615, F1 0.702, AUPRC 0.654, AUROC 0.640; VLM balanced accuracy 0.585, F1 0.666, AUPRC 0.646, AUROC 0.596',
        'Status': 'Completed'
    },
    {
        'Time_order': 'W4',
        'Date_window': 'Week 4',
        'Experiment': 'Paired VLM-vs-LLM comparison',
        'Modality': 'Cross-model benchmark',
        'Target': 'Consensus and soft-label paired benchmark',
        'Data': 'Same outer-fold rows across models',
        'Result_summary': 'LLM exceeded VLM in balanced accuracy (+0.047), specificity (+0.147), AUPRC (+0.012), AUROC (+0.061); McNemar p = 0.0418; bootstrap CI for balanced accuracy difference [-0.031, 0.111]',
        'Status': 'Completed'
    },
]

pending = [
    {
        'Time_order': 'Next',
        'Date_window': 'Next 1-2 days',
        'Experiment': 'Qwen3-8B transcript WD_P regression on paired benchmark',
        'Modality': 'LLM transcript',
        'Target': 'Five-fold patient-disjoint regression on the paired benchmark',
        'Data': 'Paired master manifest / fold manifests',
        'Planned_check': 'Run across all five folds and aggregate row-level OOF predictions',
        'Status': 'Pending'
    },
    {
        'Time_order': 'Next',
        'Date_window': 'Next 1-2 days',
        'Experiment': 'Qwen3-8B consensus binary WD_P fine-tuning on paired benchmark',
        'Modality': 'LLM transcript',
        'Target': 'Consensus WD_P binary target',
        'Data': 'Paired master manifest / fold manifests',
        'Planned_check': 'Complete all five folds and pair with VLM consensus output',
        'Status': 'Pending'
    },
    {
        'Time_order': 'Next',
        'Date_window': 'Next 1-2 days',
        'Experiment': 'Qwen3-8B soft-label WD_P fine-tuning on paired benchmark',
        'Modality': 'LLM transcript',
        'Target': 'Soft WD_P label (0/0.5/1)',
        'Data': 'Paired master manifest / fold manifests',
        'Planned_check': 'Run all five folds and evaluate consensus rows',
        'Status': 'Pending'
    },
    {
        'Time_order': 'Next',
        'Date_window': 'Next 1-2 days',
        'Experiment': 'VLM re-run on same paired benchmark',
        'Modality': 'VLM video',
        'Target': 'Zero-shot, few-shot, regression, consensus, soft-label benchmark parity',
        'Data': 'Same fold manifests and frame cache',
        'Planned_check': 'Align outputs with the LLM cohort for paired statistical comparison',
        'Status': 'Pending'
    },
]

for ws_name, rows in [('Completed_4_weeks', completed), ('Pending', pending)]:
    wb = Workbook()
    ws = wb.active
    ws.title = ws_name

    if ws_name == 'Completed_4_weeks':
        headers = ['Time_order', 'Date_window', 'Experiment', 'Modality', 'Target', 'Data', 'Result_summary', 'Status']
    else:
        headers = ['Time_order', 'Date_window', 'Experiment', 'Modality', 'Target', 'Data', 'Planned_check', 'Status']

    ws.append(headers)
    for row in rows:
        if ws_name == 'Completed_4_weeks':
            ws.append([
                row['Time_order'],
                row['Date_window'],
                row['Experiment'],
                row['Modality'],
                row['Target'],
                row['Data'],
                row['Result_summary'],
                row['Status'],
            ])
        else:
            ws.append([
                row['Time_order'],
                row['Date_window'],
                row['Experiment'],
                row['Modality'],
                row['Target'],
                row['Data'],
                row['Planned_check'],
                row['Status'],
            ])

    header_fill = PatternFill('solid', fgColor='1F4E78')
    header_font = Font(color='FFFFFF', bold=True)
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center', vertical='center')

    for column_cells in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in column_cells)
        ws.column_dimensions[column_cells[0].column_letter].width = min(max_len + 3, 44)

    ws.freeze_panes = 'A2'
    ws.auto_filter.ref = ws.dimensions

    # Keep a second sheet in same workbook for completeness
    if ws_name == 'Completed_4_weeks':
        wb.create_sheet('Pending')
        p_ws = wb['Pending']
        p_headers = ['Time_order', 'Date_window', 'Experiment', 'Modality', 'Target', 'Data', 'Planned_check', 'Status']
        p_ws.append(p_headers)
        for row in pending:
            p_ws.append([
                row['Time_order'],
                row['Date_window'],
                row['Experiment'],
                row['Modality'],
                row['Target'],
                row['Data'],
                row['Planned_check'],
                row['Status'],
            ])
        for cell in p_ws[1]:
            cell.fill = PatternFill('solid', fgColor='7F8C8D')
            cell.font = Font(color='FFFFFF', bold=True)
            cell.alignment = Alignment(horizontal='center', vertical='center')
        for column_cells in p_ws.columns:
            max_len = max(len(str(cell.value or '')) for cell in column_cells)
            p_ws.column_dimensions[column_cells[0].column_letter].width = min(max_len + 3, 44)
        p_ws.freeze_panes = 'A2'
        p_ws.auto_filter.ref = p_ws.dimensions

    wb.save(out_path)

print(f'Created: {out_path}')
