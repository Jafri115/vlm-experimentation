"""Evaluate WD-only Qwen3-8B predictions against human WD_P ratings."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from compare_qwen3_8b_to_ground_truth import (
    ROOT,
    binary_metrics,
    build_ground_truth,
    read_csv,
    write_csv,
)

LABELS = {'NO_WD_P', 'WD_P'}


def _is_true(value):
    return value in (True, 1, '1', 'true', 'True')


def evaluate_wd(prediction_path, ground_truth, output):
    predictions = read_csv(prediction_path)
    required = {'segment_uid', 'status', 'primary_label', 'transcript_provider'}
    if not predictions or not required.issubset(predictions[0]):
        raise ValueError(f'Prediction CSV requires {sorted(required)}')

    latest = {}
    for row in predictions:
        latest[row['segment_uid']] = row
    successful = {uid: row for uid, row in latest.items() if row['status'].upper() == 'OK'}
    bad_labels = sorted({row['primary_label'] for row in successful.values()} - LABELS)
    if bad_labels:
        raise ValueError(f'Unexpected WD-only prediction labels: {bad_labels}')

    truth = {row['segment_uid']: row for row in ground_truth}
    unknown = sorted(set(successful) - set(truth))
    if unknown:
        raise ValueError(f'Predictions do not match the ground-truth inventory: {unknown[:10]}')

    joined = []
    for uid, prediction in successful.items():
        pred_wd = int(prediction['primary_label'] == 'WD_P')
        if str(prediction.get('wd_p_score', '')).strip():
            score = int(prediction['wd_p_score'])
            if score not in range(1, 6) or pred_wd != int(score > 1):
                raise ValueError(f'Inconsistent score/label for {uid}')
        row = {
            **truth[uid],
            **{f'prediction_{key}': value for key, value in prediction.items()},
            'primary_label': prediction['primary_label'],
            'transcript_provider': prediction['transcript_provider'],
            'pred_wd': pred_wd,
            'correct_wd': int(pred_wd == int(truth[uid]['WD_P_binary_mean_gt1'])),
        }
        joined.append(row)
    joined.sort(key=lambda row: (row['patient_id'], int(row['session_id']), row['segment_number']))

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'matched_predictions_ground_truth.csv', joined)
    write_csv(output/'false_negatives.csv', [row for row in joined if int(row['WD_P_binary_mean_gt1']) and not row['pred_wd']])
    write_csv(output/'false_positives.csv', [row for row in joined if not int(row['WD_P_binary_mean_gt1']) and row['pred_wd']])

    primary = binary_metrics(joined, 'WD_P_binary_mean_gt1', 'pred_wd')
    strict_rows = [row for row in joined if _is_true(row['strict_wd_two_plus_rater_consensus'])]
    strict = binary_metrics(strict_rows, 'strict_wd_binary', 'pred_wd')
    providers = []
    for provider in sorted({row['transcript_provider'] for row in joined}):
        subset = [row for row in joined if row['transcript_provider'] == provider]
        providers.append({'transcript_provider': provider, **binary_metrics(subset, 'WD_P_binary_mean_gt1', 'pred_wd')})
    write_csv(output/'metrics_by_transcript_provider.csv', providers)

    score_counts = Counter()
    for row in joined:
        score = str(row.get('prediction_wd_p_score', '')).strip()
        if score:
            score_counts[score] += 1
    failed = [row for row in latest.values() if row['status'].upper() != 'OK']
    write_csv(output/'failed_prediction_rows.csv', failed, list(predictions[0]))
    summary = {
        'task': '3RS v2022 patient withdrawal only',
        'prediction_file': str(prediction_path.resolve()),
        'ground_truth_rows': len(ground_truth),
        'prediction_rows_latest': len(latest),
        'successful_predictions_matched': len(joined),
        'failed_prediction_rows': len(failed),
        'ground_truth_rule': 'mean(WD_P) > 1',
        'prediction_rule': 'wd_p_score > 1',
        'prediction_label_counts': dict(Counter(row['primary_label'] for row in joined)),
        'prediction_score_counts': dict(sorted(score_counts.items())),
        'primary_mean_rating_metrics': primary,
        'strict_wd_consensus_rows': len(strict_rows),
        'strict_wd_consensus_metrics': strict,
        'metrics_by_transcript_provider': providers,
    }
    (output/'evaluation_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    return summary


def main(args):
    ground_truth, missing = build_ground_truth(args.annotations, args.inventory)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'ground_truth_patient_withdrawal.csv', ground_truth)
    write_csv(args.output/'ground_truth_unmatched_inventory.csv', missing, ['segment_uid', 'reason'])
    print(f'Ground truth: {len(ground_truth)} rows; unmatched inventory: {len(missing)}')
    print(json.dumps(evaluate_wd(args.predictions, ground_truth, args.output), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, default=ROOT.parent/'VLM_experiments/data/completed_segments_merged (1).csv')
    parser.add_argument('--inventory', type=Path, default=ROOT/'data/amberscript_llm/inventory_all_segments.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'output/qwen3_8b_wd_3rs_ground_truth_comparison')
    main(parser.parse_args())
