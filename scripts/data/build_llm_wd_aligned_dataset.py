"""Build a transcript WD_P dataset aligned row-for-row to a VLM manifest.

No model is loaded. The VLM manifest supplies the cohort and split; the transcript
JSONL supplies text; the raw annotation CSV supplies two-rater WD_P targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def norm_id(value):
    text = str(value).strip()
    return text[:-2] if text.endswith('.0') else text


def norm_session(value):
    match = re.fullmatch(r'[sS]?(\d+)', norm_id(value))
    if not match:
        raise ValueError(f'Invalid session ID: {value!r}')
    return str(int(match.group(1)))


def segment_uid(patient_id, session_id, segment_id):
    return f'{norm_id(patient_id)}_S{norm_session(session_id)}_seg{int(float(segment_id)):03d}'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_transcripts(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    result = {}
    for row in rows:
        uid = row['segment_uid']
        if uid in result:
            raise ValueError(f'Duplicate transcript segment_uid: {uid}')
        result[uid] = row
    return result


def load_two_rater_targets(path, threshold):
    raw = pd.read_csv(path, encoding='utf-8-sig')
    required = {'patient_id', 'session_id', 'segment_id', 'coder', 'WD_P'}
    if not required.issubset(raw.columns):
        raise ValueError(f'Annotation CSV requires {sorted(required)}')
    raw = raw.dropna(subset=list(required)).copy()
    raw['segment_uid'] = [segment_uid(p, s, n) for p, s, n in zip(raw.patient_id, raw.session_id, raw.segment_id)]
    raw['coder'] = raw['coder'].astype(str).str.strip().replace({'segments Alex': 'Alex'})
    raw['WD_P'] = pd.to_numeric(raw['WD_P'], errors='raise')
    duplicate = raw.groupby(['segment_uid', 'coder']).size()
    if (duplicate > 1).any():
        raise ValueError('Duplicate physical-segment/coder annotations found')
    targets = {}
    for uid, group in raw.groupby('segment_uid', sort=False):
        if group['coder'].nunique() != 2 or len(group) != 2:
            continue
        group = group.sort_values('coder')
        ratings = [float(x) for x in group['WD_P']]
        votes = [int(x >= threshold) for x in ratings]
        targets[uid] = {
            'coder_1': str(group.iloc[0]['coder']), 'coder_2': str(group.iloc[1]['coder']),
            'WD_P_rater1': ratings[0], 'WD_P_rater2': ratings[1],
            'WD_P_mean': sum(ratings)/2.0,
            'WD_hard_mean': int(sum(ratings)/2.0 >= threshold),
            'WD_soft': sum(votes)/2.0,
            'WD_consensus': votes[0] if votes[0] == votes[1] else None,
            'WD_binary_disagreement': int(votes[0] != votes[1]),
            'WD_absolute_rater_difference': abs(ratings[0]-ratings[1]),
        }
    return targets


def build(args):
    vlm = pd.read_csv(args.vlm_manifest, encoding='utf-8-sig')
    required = {'patient_id', 'session_id', 'segment_id'}
    if not required.issubset(vlm.columns):
        raise ValueError(f'VLM manifest requires {sorted(required)}')
    vlm = vlm.copy()
    vlm['segment_uid'] = [segment_uid(p, s, n) for p, s, n in zip(vlm.patient_id, vlm.session_id, vlm.segment_id)]
    if vlm['segment_uid'].duplicated().any():
        raise ValueError('VLM manifest has duplicate physical segments')

    transcripts = load_transcripts(args.transcripts)
    targets = load_two_rater_targets(args.annotations, args.positive_threshold)
    included, excluded = [], []
    for item in vlm.to_dict('records'):
        uid = item['segment_uid']
        transcript = transcripts.get(uid)
        reason = None
        if transcript is None:
            reason = 'NO_TRANSCRIPT_ROW'
        elif not str(transcript.get('transcript_text', '')).strip():
            reason = 'EMPTY_TRANSCRIPT'
        elif not args.allow_review_transcripts and not bool(transcript.get('llm_ready')):
            reason = f"TRANSCRIPT_NOT_READY:{transcript.get('transcript_status', '')}"
        elif uid not in targets:
            reason = 'NOT_EXACTLY_TWO_RATERS'
        if reason:
            excluded.append({'segment_uid': uid, 'reason': reason,
                             'split': item.get('split', ''), 'patient_id': norm_id(item['patient_id'])})
            continue
        target = targets[uid]
        row = {
            **item,
            'segment_uid': uid,
            'patient_id': norm_id(item['patient_id']),
            'session_id': norm_session(item['session_id']),
            'segment_id': int(float(item['segment_id'])),
            'transcript_text': transcript['transcript_text'],
            'transcript_text_plain': transcript.get('transcript_text_plain', ''),
            'transcript_provider': transcript.get('transcript_provider', ''),
            'transcript_status': transcript.get('transcript_status', ''),
            'review_flags': transcript.get('review_flags', ''),
            'llm_ready': bool(transcript.get('llm_ready')),
            **target,
        }
        included.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(included).to_csv(args.output/'aligned_manifest.csv', index=False, encoding='utf-8-sig')
    with (args.output/'aligned_manifest.jsonl').open('w', encoding='utf-8') as stream:
        for row in included:
            stream.write(json.dumps(json_clean(row), ensure_ascii=False)+'\n')
    pd.DataFrame(excluded).to_csv(args.output/'excluded_vlm_segments.csv', index=False, encoding='utf-8-sig')

    split_counts = Counter(str(row.get('split', 'unspecified')) for row in included)
    excluded_reasons = Counter(row['reason'] for row in excluded)
    summary = {
        'vlm_manifest': str(args.vlm_manifest.resolve()),
        'vlm_manifest_sha256': sha256(args.vlm_manifest),
        'transcripts': str(args.transcripts.resolve()),
        'transcripts_sha256': sha256(args.transcripts),
        'annotations': str(args.annotations.resolve()),
        'annotations_sha256': sha256(args.annotations),
        'positive_threshold': args.positive_threshold,
        'require_llm_ready': not args.allow_review_transcripts,
        'vlm_rows': len(vlm), 'aligned_rows': len(included), 'excluded_rows': len(excluded),
        'coverage_fraction': len(included)/len(vlm) if len(vlm) else 0,
        'aligned_split_counts': dict(split_counts),
        'excluded_reason_counts': dict(excluded_reasons),
        'aligned_provider_counts': dict(Counter(row['transcript_provider'] for row in included)),
        'soft_target_counts': dict(Counter(str(row['WD_soft']) for row in included)),
        'consensus_rows': sum(row['WD_consensus'] is not None for row in included),
        'patients': len({row['patient_id'] for row in included}),
    }
    (args.output/'dataset_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vlm-manifest', type=Path, required=True,
                        help='The exact VLM manifest whose cohort/split must be preserved.')
    parser.add_argument('--transcripts', type=Path, default=ROOT/'data/amberscript_llm/llm_segments_all.jsonl')
    parser.add_argument('--annotations', type=Path,
                        default=ROOT.parent/'facs-openface-tools/data/completed_segments_merged (1).csv')
    parser.add_argument('--positive-threshold', type=float, default=2.0)
    parser.add_argument('--allow-review-transcripts', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'output/llm_wd_aligned')
    build(parser.parse_args())
