"""Supplement unavailable Amberscript inputs using full-session role timelines.

Never trust segment-local ASR speaker labels. Unresolved candidates are staged,
not promoted to ready inputs. A baseline snapshot makes reruns reproducible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

from build_amberscript_llm_dataset import (
    ROOT, SYSTEM_PROMPT, STAMP_NOISE, dialogue_text, normalize_space,
    remove_parentheses, strip_markup, write_csv, write_jsonl,
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def union_duration(intervals):
    total, right = 0.0, -math.inf
    for start, end in sorted(intervals):
        total += max(0.0, end - max(start, right))
        right = max(right, end)
    return total


def validate_turns(turns):
    for turn in turns:
        start, end = float(turn['start']), float(turn['end'])
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
            raise ValueError('Invalid full-session diarization interval')
        if turn['role'] not in {'T', 'P'}:
            raise ValueError('Unresolved full-session role')
    return sorted(turns, key=lambda t: (t['start'], t['end']))


def load_timelines(artifacts):
    """Use roles only with their original full-session cluster namespace."""
    timelines, report = {}, []
    root = artifacts / 'full_session_role_mapping'
    for path in sorted((root / 'session_role_turns').glob('*.json')):
        data = read_json(path)
        uid = f"{data['patient_id']}_S{int(data['session_id'])}"
        t, p = data['therapist_global_speaker'], data['patient_global_speaker']
        if not t or not p or t == p:
            raise ValueError(f'Invalid global role mapping: {path}')
        for turn in data['turns']:
            expected = {t: 'T', p: 'P'}.get(turn['global_speaker'])
            if turn['role'] != expected:
                raise ValueError(f'Global speaker/role contradiction: {path}')
        timelines[uid] = {'turns': validate_turns(data['turns']), 'source': str(path.resolve()),
                          'sha256': sha256(path), 'role_source': str(path.resolve()),
                          'role_source_sha256': sha256(path), 'role_status': data['mapping_status'],
                          'therapist_id': data['therapist_id'],
                          'provenance': 'legacy_full_session_global_mapping_unversioned'}
    profiles_root = artifacts / 'therapist_profile_builder_v2'
    profiles_path = profiles_root / 'session_profiles.csv'
    if profiles_path.exists():
        for profile in read_csv(profiles_path):
            uid = f"{profile['patient_id']}_S{int(profile['session_id'])}"
            if uid in timelines:
                continue
            path = profiles_root / 'full_session_diarization' / f'{uid}.json'
            if not path.exists():
                continue
            if profile.get('seed_qc_status') != 'PASS' or profile.get('timestamp_saved_speaker_agree', '').lower() != 'true':
                report.append({'session_uid': uid, 'status': 'SEED_ROLE_QC_NOT_PASSED', 'source': str(path)})
                continue
            raw = read_json(path)
            speakers = {t['speaker'] for t in raw}
            therapist = profile['session_therapist_speaker']
            if len(speakers) != 2 or therapist not in speakers:
                report.append({'session_uid': uid, 'status': 'AMBIGUOUS_SESSION_SPEAKERS', 'source': str(path)})
                continue
            turns = [{**turn, 'global_speaker': turn['speaker'], 'role': 'T' if turn['speaker'] == therapist else 'P'} for turn in raw]
            timelines[uid] = {'turns': validate_turns(turns), 'source': str(path.resolve()),
                              'sha256': sha256(path), 'role_source': str(profiles_path.resolve()),
                              'role_source_sha256': sha256(profiles_path),
                              'role_status': 'MANUAL_THERAPIST_SEED_QC_PASS_PATIENT_BY_EXCLUSION',
                              'therapist_id': profile['therapist_id'],
                              'provenance': 'legacy_full_session_seed_mapping_unversioned'}
    for uid, source in sorted(timelines.items()):
        report.append({'session_uid': uid, 'status': 'AVAILABLE_FULL_SESSION_TIMELINE',
                       **{k: v for k, v in source.items() if k != 'turns'}, 'turn_count': len(source['turns'])})
    reviews_path = root / 'global_only_review_results.csv'
    reviews = {r['segment_uid']: r for r in read_csv(reviews_path)} if reviews_path.exists() else {}
    return timelines, reviews, report


def resolve_role(start, end, turns, min_coverage=0.5, min_purity=0.8):
    """Match by absolute time, never by segment-local SPEAKER_XX identity."""
    overlap = {}
    for role in ('T', 'P'):
        overlap[role] = union_duration([(max(start, t['start']), min(end, t['end']))
                                       for t in turns if t['role'] == role and t['start'] < end and t['end'] > start])
    winner = max(overlap, key=overlap.get)
    duration = end-start
    coverage = overlap[winner]/duration if duration > 0 else 0.0
    purity = overlap[winner]/sum(overlap.values()) if sum(overlap.values()) else 0.0
    known = coverage >= min_coverage and purity >= min_purity
    reason = 'ASSIGNED' if known else 'NO_SESSION_SPEECH_OVERLAP' if not sum(overlap.values()) else 'INSUFFICIENT_OR_AMBIGUOUS_OVERLAP'
    return {'speaker': winner if known else 'UNKNOWN', 'role_assignment_status': reason,
            'role_overlap_sec': overlap, 'role_coverage': round(coverage, 6), 'role_purity': round(purity, 6)}


def candidate(row, transcript_path, timeline, review, args):
    data = read_json(transcript_path)
    cues = data.get('segments')
    if not isinstance(cues, list):
        raise ValueError(f'Missing segments in {transcript_path}')
    flags = []
    metadata = data.get('transcription_metadata', {})
    if metadata.get('prediction_anomalous'):
        flags.append('ASR_PREDICTION_ANOMALOUS')
    if not timeline:
        flags.append('MISSING_FULL_SESSION_DIARIZATION')
    elif timeline['therapist_id'] != row.get('therapist_id'):
        flags.append('SESSION_THERAPIST_METADATA_MISMATCH')
    if review and review.get('global_result_supported', '').lower() != 'true':
        flags.append('MANUAL_REVIEW_DISPUTES_GLOBAL_DIARIZATION')
    cleaned, diagnostics = remove_parentheses('\x1e'.join(strip_markup(str(c.get('text', ''))) for c in cues))
    if diagnostics['unmatched_open_parentheses']:
        flags.append('UNMATCHED_OPEN_PARENTHESIS')
    utterances, previous_start = [], -math.inf
    offset, duration = float(row['start_sec']), float(row['end_sec'])-float(row['start_sec'])
    for i, (cue, text) in enumerate(zip(cues, cleaned.split('\x1e')), 1):
        text = normalize_space(STAMP_NOISE.sub('', text))
        if not text or not re.search(r'\w', text):
            continue
        start, end = float(cue['start']), float(cue['end'])
        valid = math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + 0.001
        if not valid or start < previous_start:
            flags.append('INVALID_ASR_TIMESTAMPS')
        previous_start = start
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f'Nonfinite ASR timestamps: {transcript_path}')
        global_start, global_end = offset+start, offset+end
        assignment = resolve_role(global_start, global_end, timeline['turns'], args.min_coverage, args.min_purity) if valid and timeline else {'speaker': 'UNKNOWN', 'role_assignment_status': 'NO_VALID_SESSION_ALIGNMENT'}
        utterances.append({'cue_id': i, 'start_sec': global_start, 'end_sec': global_end, 'segment_local_start_sec': start,
                           'segment_local_end_sec': end, 'text': text, 'original_asr_role': cue.get('speaker'),
                           'original_asr_raw_speaker': cue.get('raw_speaker'), **assignment})
    unknown = sum(u['speaker'] == 'UNKNOWN' for u in utterances)
    if unknown:
        flags.append('UNRESOLVED_UTTERANCE_ROLES')
    if not utterances:
        flags.append('NO_TRANSCRIPT_TEXT')
    flags = list(dict.fromkeys(flags))
    status = 'READY_FULL_SESSION_ROLES' if not flags else flags[0]
    return {'segment_uid': row['segment_uid'], 'transcript_source': str(transcript_path.resolve()),
            'transcript_source_sha256': sha256(transcript_path), 'asr_model': data.get('model'),
            'asr_transcription_metadata': metadata, 'full_session_diarization_source': timeline['source'] if timeline else '',
            'full_session_diarization_sha256': timeline['sha256'] if timeline else '',
            'full_session_role_source': timeline['role_source'] if timeline else '',
            'full_session_role_source_sha256': timeline['role_source_sha256'] if timeline else '',
            'full_session_role_status': timeline['role_status'] if timeline else '',
            'full_session_diarization_provenance': timeline['provenance'] if timeline else '',
            'manual_global_review': review or {}, 'cleaning_diagnostics': diagnostics,
            'memopsy_status': status, 'memopsy_review_flags': ';'.join(flags), 'unresolved_utterances': unknown,
            'changed_role_utterances': sum(u['speaker'] != 'UNKNOWN' and u['speaker'] != u['original_asr_role'] for u in utterances),
            'transcript_text': dialogue_text(utterances), 'utterances': utterances}


def build(args):
    output = args.dataset.resolve()
    baseline = output / 'before_memopsy_completion'
    if not baseline.exists():
        baseline.mkdir()
        for name in ('llm_segments_all.jsonl', 'dataset_summary.json'):
            shutil.copy2(output/name, baseline/name)
    rows = [json.loads(line) for line in (baseline/'llm_segments_all.jsonl').read_text(encoding='utf-8').splitlines()]
    # A stricter rerun can reject a formerly promoted row. Restore its original
    # text first so the baseline manifest never points at a previous replacement.
    for path in (baseline/'segments').glob('*.txt'):
        shutil.copy2(path, output/'segments'/path.name)
    summary = read_json(baseline/'dataset_summary.json')
    timelines, reviews, timeline_report = load_timelines(args.artifacts)
    candidates, audit, replacements = [], [], []
    for row in rows:
        if not row['in_audio_inventory'] or row['llm_ready']:
            continue
        uid = row['segment_uid']
        session_uid = uid.split('_seg')[0]
        paths = list((args.artifacts/'memopsy_segments'/f'{uid}_sync_ctc').glob(f'{uid}_*.json'))
        if len(paths) > 1:
            raise ValueError(f'Ambiguous transcript sources for {uid}')
        if not paths:
            result = {'segment_uid': uid, 'memopsy_status': 'MISSING_ASR_TRANSCRIPT', 'memopsy_review_flags': '', 'unresolved_utterances': 0, 'changed_role_utterances': 0}
        else:
            result = candidate(row, paths[0], timelines.get(session_uid), reviews.get(uid), args)
            candidates.append(result)
        original_status = row['transcript_status']
        row.update(memopsy_status=result['memopsy_status'], memopsy_review_flags=result['memopsy_review_flags'])
        if result['memopsy_status'] == 'READY_FULL_SESSION_ROLES':
            previous_path = Path(row['transcript_path']) if row['transcript_path'] else None
            if previous_path and previous_path.exists():
                (baseline/'segments').mkdir(exist_ok=True)
                if not (baseline/'segments'/previous_path.name).exists():
                    shutil.copy2(previous_path, baseline/'segments'/previous_path.name)
            row.update({k: result[k] for k in ('transcript_text', 'utterances', 'transcript_source', 'transcript_source_sha256',
                                             'full_session_diarization_source', 'full_session_diarization_sha256',
                                             'full_session_role_source', 'full_session_role_source_sha256',
                                             'full_session_role_status', 'full_session_diarization_provenance')})
            path = output/'segments'/f'{uid}.txt'
            path.write_text(row['transcript_text']+'\n', encoding='utf-8')
            row.update(transcript_path=str(path), transcript_status='READY', llm_ready=True,
                       previous_transcript_status=original_status, transcript_origin='memopsy_asr',
                       transcript_speaker_source='full_session_time_overlap_roles_provisional',
                       role_mapping_is_ground_truth=False, timing_method='asr_cue_local_time_plus_inventory_offset',
                       alignment_policy='same_segment_uid_then_absolute_time_overlap_with_full_session_roles',
                       word_count=sum(len(u['text'].split()) for u in result['utterances']), boundary_crossing_cues=0,
                       review_flags='ASR_TEXT_UNVERIFIED;FULL_SESSION_ROLES_PROVISIONAL;LEGACY_DIARIZATION_UNVERSIONED')
            replacements.append(uid)
        audit.append({'segment_uid': uid, 'patient_id': row['patient_id'], 'session_id': row['session_id'],
                      'previous_transcript_status': original_status, 'added_to_ready': uid in replacements,
                      **{k: v for k, v in result.items() if k not in {'segment_uid', 'transcript_text', 'utterances', 'asr_transcription_metadata', 'manual_global_review', 'cleaning_diagnostics'}}})
    # Preserve every original segment index and inventory field, including provisional local roles.
    flat = [{k: v for k, v in row.items() if k != 'utterances'} for row in rows]
    ready = [r for r in rows if r['llm_ready']]
    for name, selection in [('llm_segments_all', flat), ('llm_segments_ready', [r for r in flat if r['llm_ready']]),
                            ('inventory_all_segments', [r for r in flat if r['in_audio_inventory']]),
                            ('inventory_ready', [r for r in flat if r['llm_ready'] and r['in_audio_inventory']])]:
        write_csv(output/f'{name}.csv', selection)
    write_jsonl(output/'llm_segments_all.jsonl', rows)
    write_jsonl(output/'llm_segments_ready.jsonl', ready)
    write_jsonl(output/'inventory_ready.jsonl', [r for r in ready if r['in_audio_inventory']])
    requests = []
    for row in ready:
        prompt = SYSTEM_PROMPT
        if row.get('transcript_origin') == 'memopsy_asr':
            prompt = prompt.replace('Speaker labels are taken from the transcript export and have not been independently verified.',
                                    'Speaker labels were reassigned using full-session diarization and provisional therapist/patient mappings. Transcription and speaker labels may contain errors.')
        requests.append({'segment_uid': row['segment_uid'], 'segment_idx': row['segment_idx'], 'messages': [
            {'role': 'system', 'content': prompt}, {'role': 'user', 'content': row['transcript_text']}]})
    write_jsonl(output/'llm_requests.jsonl', requests)
    write_csv(output/'segments_needing_review.csv', [r for r in flat if not r['llm_ready'] or r['review_flags']])
    write_jsonl(output/'memopsy_candidates.jsonl', candidates)
    write_csv(output/'memopsy_completion_audit.csv', audit)
    write_csv(output/'full_session_diarization_sources.csv', timeline_report)
    pending = [r for r in audit if not r['added_to_ready']]
    write_csv(output/'memopsy_remaining.csv', pending)
    missing = Counter(f"{r['patient_id']}_S{r['session_id']}" for r in pending if r['memopsy_status'] == 'MISSING_FULL_SESSION_DIARIZATION')
    write_csv(output/'missing_full_session_diarization.csv', [{'session_uid': uid, 'pending_segments': count} for uid, count in sorted(missing.items())])
    completion = {'attempted_inventory_segments': len(audit), 'asr_transcripts_found': len(candidates),
                  'full_session_timelines_found': len(timelines), 'segments_added_to_ready': len(replacements),
                  'candidate_status_counts': dict(Counter(r['memopsy_status'] for r in audit)),
                  'min_role_coverage': args.min_coverage, 'min_role_purity': args.min_purity,
                  'remaining_inventory_not_ready': sum(not r['llm_ready'] and r['in_audio_inventory'] for r in rows),
                  'new_diarization_computed': False, 'baseline_snapshot': str(baseline),
                  'notes': ['Legacy full-session caches found; no completed all-session fixed diarization set found.',
                            'Only candidates with every nonempty utterance assigned unambiguously are promoted.',
                            'Manual reviews disputing global diarization block promotion.',
                            'Original segment-local ASR speaker labels never determine corrected roles.',
                            'Missing-session candidates are staged but not added to prediction-ready inputs.']}
    summary.update(ready_segments=len(ready), inventory_with_transcript_source=sum(bool(r['transcript_source']) for r in rows if r['in_audio_inventory']),
                   inventory_status_counts=dict(Counter(r['transcript_status'] for r in rows if r['in_audio_inventory'])),
                   all_status_counts=dict(Counter(r['transcript_status'] for r in rows)),
                   boundary_crossing_segments=sum(r['boundary_crossing_cues'] > 0 for r in rows), memopsy_completion=completion)
    (output/'dataset_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(completion, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/amberscript_llm')
    parser.add_argument('--artifacts', type=Path, default=ROOT.parents[1]/'german-asr-pipeline/artifacts')
    parser.add_argument('--min-coverage', type=float, default=0.5)
    parser.add_argument('--min-purity', type=float, default=0.8)
    args = parser.parse_args()
    if not 0 < args.min_coverage <= 1 or not 0.5 < args.min_purity <= 1:
        parser.error('Require 0 < coverage <= 1 and 0.5 < purity <= 1')
    build(args)
