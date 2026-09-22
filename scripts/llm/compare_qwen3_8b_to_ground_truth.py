"""Build patient-rupture ground truth and evaluate Qwen3-8B predictions.

The primary target matches the existing VLM comparison rule: mean WD_P > 1
and mean CF_P > 1. Joins use patient/session/segment identity, never segment_idx.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('NO_RUPTURE', 'WD_P', 'CF_P', 'MIXED_P')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_id(value):
    text = str(value).strip()
    if text.endswith('.0'):
        text = text[:-2]
    return text


def normalize_session(value):
    text = normalize_id(value)
    match = re.fullmatch(r'[sS]?(\d+)', text)
    if not match:
        raise ValueError(f'Invalid session ID: {value!r}')
    return str(int(match.group(1)))


def parse_hms(value):
    parts = str(value).strip().split(':')
    if len(parts) != 3:
        raise ValueError(f'Invalid timestamp: {value!r}')
    return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])


def numeric(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'Missing/non-numeric {name}: {value!r}') from None
    if not math.isfinite(result):
        raise ValueError(f'Non-finite {name}: {value!r}')
    return result


def type_label(wd, cf):
    if wd and cf:
        return 'MIXED_P'
    if wd:
        return 'WD_P'
    if cf:
        return 'CF_P'
    return 'NO_RUPTURE'


def build_ground_truth(annotation_path, inventory_path, threshold=1.0):
    annotations = read_csv(annotation_path)
    required = {'patient_id', 'session_id', 'coder', 'segment_id', 'segment_start', 'segment_end', 'WD_P', 'CF_P'}
    if not annotations or not required.issubset(annotations[0]):
        raise ValueError(f'Annotation CSV requires {sorted(required)}')
    grouped = defaultdict(list)
    for row in annotations:
        if not all(str(row.get(k, '')).strip() for k in required):
            continue
        key = (normalize_id(row['patient_id']), normalize_session(row['session_id']), int(float(row['segment_id'])))
        grouped[key].append(row)
    inventory = read_csv(inventory_path)
    result, missing = [], []
    seen = set()
    for item in inventory:
        uid = item['segment_uid']
        if uid in seen:
            raise ValueError(f'Duplicate inventory segment: {uid}')
        seen.add(uid)
        key = (normalize_id(item['patient_id']), normalize_session(item['session_id']), int(float(item['segment_number'])))
        raters = grouped.get(key, [])
        if not raters:
            missing.append({'segment_uid': uid, 'reason': 'NO_ANNOTATION_MATCH'})
            continue
        coders = [str(r['coder']).strip() for r in raters]
        if len(coders) != len(set(coders)):
            raise ValueError(f'Duplicate coder for physical segment {key}')
        wd = [numeric(r['WD_P'], 'WD_P') for r in raters]
        cf = [numeric(r['CF_P'], 'CF_P') for r in raters]
        starts = {parse_hms(r['segment_start']) for r in raters}
        ends = {parse_hms(r['segment_end']) + 1.0 for r in raters}
        if len(starts) != 1:
            raise ValueError(f'Raters disagree on interval start for {key}')
        annotation_start = next(iter(starts))
        inventory_start, inventory_end = float(item['start_sec']), float(item['end_sec'])
        if abs(annotation_start-inventory_start) > 0.01:
            raise ValueError(f'Inventory/annotation start mismatch for {uid}: {inventory_start} vs {annotation_start}')
        if any(end < annotation_start-0.01 or end > inventory_end+0.01 for end in ends):
            raise ValueError(f'Annotation end outside inventory interval for {uid}: {sorted(ends)} vs {inventory_end}')
        if all(abs(end-inventory_end) <= 0.01 for end in ends):
            interval_status = 'EXACT'
        elif all(abs(end-annotation_start) <= 0.01 for end in ends):
            interval_status = 'EMPTY_FINAL_ANNOTATION_INTERVAL'
        elif len(ends) > 1:
            interval_status = 'RATER_END_DISAGREEMENT_TRUNCATED_FINAL_INTERVAL'
        else:
            interval_status = 'TRUNCATED_FINAL_ANNOTATION_INTERVAL'
        wd_mean, cf_mean = sum(wd)/len(wd), sum(cf)/len(cf)
        wd_binary, cf_binary = int(wd_mean > threshold), int(cf_mean > threshold)
        wd_votes, cf_votes = [int(x > threshold) for x in wd], [int(x > threshold) for x in cf]
        strict_wd = len(set(wd_votes)) == 1 and len(raters) >= 2
        strict = len(set(wd_votes)) == 1 and len(set(cf_votes)) == 1 and len(raters) >= 2
        row = {
            'segment_uid': uid, 'patient_id': key[0], 'session_id': key[1], 'segment_number': key[2],
            'start_sec': inventory_start, 'end_sec': inventory_end,
            'annotation_start_sec': annotation_start, 'annotation_end_sec_min': min(ends),
            'annotation_end_sec_max': max(ends), 'interval_match_status': interval_status,
            'n_raters': len(raters),
            'coders': '|'.join(coders), 'WD_P_ratings': '|'.join(str(x) for x in wd),
            'CF_P_ratings': '|'.join(str(x) for x in cf), 'WD_P_mean': wd_mean, 'CF_P_mean': cf_mean,
            'WD_P_binary_mean_gt1': wd_binary, 'CF_P_binary_mean_gt1': cf_binary,
            'ground_truth_label': type_label(wd_binary, cf_binary),
            'ground_truth_rupture': int(wd_binary or cf_binary),
            'strict_wd_two_plus_rater_consensus': strict_wd,
            'strict_wd_binary': wd_votes[0] if strict_wd else '',
            'strict_two_plus_rater_consensus': strict,
            'strict_consensus_label': type_label(wd_votes[0], cf_votes[0]) if strict else '',
            'strict_consensus_rupture': int(wd_votes[0] or cf_votes[0]) if strict else '',
            'ground_truth_rule': 'mean(WD_P)>1; mean(CF_P)>1',
        }
        result.append(row)
    return result, missing


def safe_div(a, b):
    return a/b if b else 0.0


def binary_metrics(rows, truth, prediction):
    tp = sum(int(r[truth]) == 1 and int(r[prediction]) == 1 for r in rows)
    tn = sum(int(r[truth]) == 0 and int(r[prediction]) == 0 for r in rows)
    fp = sum(int(r[truth]) == 0 and int(r[prediction]) == 1 for r in rows)
    fn = sum(int(r[truth]) == 1 and int(r[prediction]) == 0 for r in rows)
    sensitivity, specificity = safe_div(tp, tp+fn), safe_div(tn, tn+fp)
    precision = safe_div(tp, tp+fp)
    return {'N': len(rows), 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'accuracy': safe_div(tp+tn, len(rows)), 'balanced_accuracy': (sensitivity+specificity)/2,
            'precision': precision, 'recall_sensitivity': sensitivity, 'specificity': specificity,
            'f1': safe_div(2*precision*sensitivity, precision+sensitivity),
            'ground_truth_positive_rate': safe_div(tp+fn, len(rows)),
            'predicted_positive_rate': safe_div(tp+fp, len(rows))}


def multiclass_metrics(rows, truth='ground_truth_label', prediction='primary_label'):
    confusion = []
    recalls, precisions, f1s = [], [], []
    per_class = []
    for actual in CLASSES:
        confusion.append({'actual': actual, **{f'pred_{pred}': sum(r[truth] == actual and r[prediction] == pred for r in rows) for pred in CLASSES}})
        tp = sum(r[truth] == actual and r[prediction] == actual for r in rows)
        fp = sum(r[truth] != actual and r[prediction] == actual for r in rows)
        fn = sum(r[truth] == actual and r[prediction] != actual for r in rows)
        precision, recall = safe_div(tp, tp+fp), safe_div(tp, tp+fn)
        f1 = safe_div(2*precision*recall, precision+recall)
        precisions.append(precision); recalls.append(recall); f1s.append(f1)
        per_class.append({'class': actual, 'support': sum(r[truth] == actual for r in rows),
                          'predicted': sum(r[prediction] == actual for r in rows),
                          'precision': precision, 'recall': recall, 'f1': f1})
    accuracy = safe_div(sum(r[truth] == r[prediction] for r in rows), len(rows))
    observed = accuracy
    expected = sum(safe_div(sum(r[truth] == c for r in rows), len(rows)) * safe_div(sum(r[prediction] == c for r in rows), len(rows)) for c in CLASSES)
    return {'N': len(rows), 'accuracy': accuracy, 'macro_precision': sum(precisions)/len(CLASSES),
            'macro_recall_balanced_accuracy': sum(recalls)/len(CLASSES), 'macro_f1': sum(f1s)/len(CLASSES),
            'weighted_f1': safe_div(sum(x['f1']*x['support'] for x in per_class), len(rows)),
            'cohen_kappa': safe_div(observed-expected, 1-expected)}, per_class, confusion


def evaluate(prediction_path, ground_truth, output):
    predictions = read_csv(prediction_path)
    required = {'segment_uid', 'status', 'primary_label', 'transcript_provider'}
    if not predictions or not required.issubset(predictions[0]):
        raise ValueError(f'Prediction CSV requires {sorted(required)}')
    latest = {}
    for row in predictions:
        latest[row['segment_uid']] = row
    ok = {uid: r for uid, r in latest.items() if r['status'].upper() == 'OK'}
    bad_labels = sorted({r['primary_label'] for r in ok.values()} - set(CLASSES))
    if bad_labels:
        raise ValueError(f'Unexpected prediction labels: {bad_labels}')
    truth = {r['segment_uid']: r for r in ground_truth}
    unknown_predictions = sorted(set(ok)-set(truth))
    if unknown_predictions:
        raise ValueError(f'Predictions do not match ground truth inventory: {unknown_predictions[:10]}')
    joined = [{**truth[uid], **{f'prediction_{k}': v for k, v in p.items()},
               'primary_label': p['primary_label'], 'transcript_provider': p['transcript_provider'],
               'pred_rupture': int(p['primary_label'] != 'NO_RUPTURE'),
               'pred_wd': int(p['primary_label'] in {'WD_P', 'MIXED_P'}),
               'pred_cf': int(p['primary_label'] in {'CF_P', 'MIXED_P'}),
               'correct_4class': int(p['primary_label'] == truth[uid]['ground_truth_label']),
               'correct_rupture': int(int(p['primary_label'] != 'NO_RUPTURE') == truth[uid]['ground_truth_rupture'])}
              for uid, p in ok.items()]
    joined.sort(key=lambda r: (r['patient_id'], int(r['session_id']), r['segment_number']))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'matched_predictions_ground_truth.csv', joined)
    write_csv(output/'misclassified_segments.csv', [r for r in joined if not r['correct_4class']])
    four, per_class, confusion = multiclass_metrics(joined)
    binary = binary_metrics(joined, 'ground_truth_rupture', 'pred_rupture')
    wd = binary_metrics(joined, 'WD_P_binary_mean_gt1', 'pred_wd')
    cf = binary_metrics(joined, 'CF_P_binary_mean_gt1', 'pred_cf')
    strict_rows = [r for r in joined if r['strict_two_plus_rater_consensus'] in (True, 'True', 'true', '1', 1)]
    strict_four, strict_per_class, strict_confusion = multiclass_metrics(strict_rows, 'strict_consensus_label')
    strict_binary = binary_metrics(strict_rows, 'strict_consensus_rupture', 'pred_rupture')
    subgroup = []
    for provider in sorted({r['transcript_provider'] for r in joined}):
        sub = [r for r in joined if r['transcript_provider'] == provider]
        m4, _, _ = multiclass_metrics(sub)
        subgroup.append({'transcript_provider': provider, **m4, **{f'binary_{k}': v for k, v in binary_metrics(sub, 'ground_truth_rupture', 'pred_rupture').items() if k != 'N'}})
    write_csv(output/'four_class_per_class.csv', per_class)
    write_csv(output/'four_class_confusion_matrix.csv', confusion)
    write_csv(output/'strict_consensus_four_class_per_class.csv', strict_per_class)
    write_csv(output/'strict_consensus_confusion_matrix.csv', strict_confusion)
    write_csv(output/'metrics_by_transcript_provider.csv', subgroup)
    failed = [r for r in latest.values() if r['status'].upper() != 'OK']
    write_csv(output/'failed_prediction_rows.csv', failed, list(predictions[0]))
    summary = {'prediction_file': str(prediction_path.resolve()), 'ground_truth_rows': len(ground_truth),
               'prediction_rows_latest': len(latest), 'successful_predictions_matched': len(joined),
               'failed_prediction_rows': len(failed),
               'prediction_label_counts': dict(Counter(r['primary_label'] for r in joined)),
               'ground_truth_label_counts': dict(Counter(r['ground_truth_label'] for r in joined)),
               'primary_ground_truth_rule': 'mean(WD_P)>1 and mean(CF_P)>1, matching existing VLM comparison',
               'four_class': four, 'any_patient_rupture_binary': binary, 'withdrawal_one_vs_rest': wd,
               'confrontation_one_vs_rest': cf, 'strict_consensus_rows': len(strict_rows),
               'strict_consensus_four_class': strict_four, 'strict_consensus_any_rupture': strict_binary,
               'note': 'Metrics cover successful rows present in the prediction file; the full ground-truth table also contains inventory rows outside that inference cohort.'}
    (output/'evaluation_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    return summary


def main(args):
    ground_truth, missing = build_ground_truth(args.annotations, args.inventory)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'ground_truth_patient_rupture.csv', ground_truth)
    write_csv(args.output/'ground_truth_unmatched_inventory.csv', missing, ['segment_uid', 'reason'])
    print(f'Ground truth: {len(ground_truth)} rows; unmatched inventory: {len(missing)}')
    if args.predictions:
        summary = evaluate(args.predictions, ground_truth, args.output)
        print(json.dumps(summary, indent=2))
    else:
        print('Ground truth prepared. Pass --predictions PATH to evaluate.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--annotations', type=Path, default=ROOT.parent/'VLM_experiments/data/completed_segments_merged (1).csv')
    parser.add_argument('--inventory', type=Path, default=ROOT/'data/amberscript_llm/inventory_all_segments.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'output/qwen3_8b_ground_truth_comparison')
    main(parser.parse_args())
