"""Build provider-labelled timed LLM inputs using complete-session pyannote output.

The input snapshot is the original Amberscript build, so previous automatic
completion passes cannot become evidence for their own speaker-role decisions.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from build_amberscript_llm_dataset import ROOT, SYSTEM_PROMPT, STAMP_NOISE, normalize_space, remove_parentheses, strip_markup, write_csv, write_jsonl
from complete_llm_dataset_from_memopsy import load_timelines, read_csv, read_json, sha256, union_duration


def timestamp(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Displayed timestamps must be finite and nonnegative')
    tenths = int(math.floor(seconds * 10 + 0.5))
    minutes, remainder = divmod(tenths, 600)
    return f'{minutes:02d}:{remainder // 10:02d}.{remainder % 10}'


def canonical_role(value):
    return {'t': 'T', 'therapeut': 'T', 'therapeutin': 'T',
            'p': 'P', 'patient': 'P', 'patientin': 'P'}.get(str(value).strip().lower())


def timed_dialogue(utterances):
    # Keep every timestamped cue; merging turns could conceal a long pause.
    return '\n'.join(f"[{timestamp(u['start_sec'])}] {u['speaker']}: {u['text']}" for u in utterances)


def overlap_scores(turns, evidence):
    scores = {s: {'T': 0.0, 'P': 0.0} for s in sorted({t['speaker'] for t in turns})}
    left = 0
    for start, end, role in sorted(evidence):
        if role not in {'T', 'P'} or end <= start:
            continue
        while left < len(turns) and turns[left]['end'] <= start:
            left += 1
        per_speaker = defaultdict(list)
        for i in range(left, len(turns)):
            turn = turns[i]
            if turn['start'] >= end:
                break
            a, b = max(start, turn['start']), min(end, turn['end'])
            if b > a:
                per_speaker[turn['speaker']].append((a, b))
        for speaker, intervals in per_speaker.items():
            scores[speaker][role] += union_duration(intervals)
    return scores


def infer_mapping(turns, evidence, method):
    scores = overlap_scores(turns, evidence)
    if len(scores) != 2:
        return {'method': method, 'mapping': {}, 'status': 'REVIEW_SPEAKER_COUNT', 'scores': scores}
    a, b = sorted(scores)
    if method == 'manual_therapist_seed_time_overlap':
        ranked = sorted(scores, key=lambda s: scores[s]['T'], reverse=True)
        winner, other = ranked
        total = sum(v['T'] for v in scores.values())
        purity = scores[winner]['T']/total if total else 0
        accepted = scores[winner]['T'] >= 3 and purity >= 0.8
        return {'method': method, 'mapping': {winner: 'T', other: 'P'} if accepted else {},
                'status': 'PROVISIONAL' if accepted else 'INSUFFICIENT_SEED_OVERLAP',
                'agreement': purity, 'scores': scores}
    orientations = [(scores[a]['T']+scores[b]['P'], {a: 'T', b: 'P'}),
                    (scores[a]['P']+scores[b]['T'], {a: 'P', b: 'T'})]
    orientations.sort(key=lambda item: item[0], reverse=True)
    best, mapping = orientations[0]
    total = sum(sum(v.values()) for v in scores.values())
    agreement = best/total if total else 0
    per_speaker = {s: scores[s][role]/sum(scores[s].values()) if sum(scores[s].values()) else 0 for s, role in mapping.items()}
    accepted = total >= 30 and agreement >= 0.75 and all(v >= 0.65 for v in per_speaker.values()) and all(sum(v.values()) >= 5 for v in scores.values())
    return {'method': method, 'mapping': mapping if accepted else {}, 'status': 'PROVISIONAL' if accepted else 'INSUFFICIENT_OR_CONFLICTING_ROLE_EVIDENCE',
            'agreement': agreement, 'per_speaker_agreement': per_speaker, 'scores': scores}


def load_new_sessions(root):
    if (root/'full_session_diarization').is_dir():
        run = root
    else:
        runs = sorted(p for p in root.iterdir() if (p/'full_session_diarization').is_dir())
        if len(runs) != 1:
            raise ValueError('Pass a single diarization run directory with --diarization')
        run = runs[0]
    summaries = {r['session_uid']: r for r in read_csv(run/'session_diarization_summary.csv')}
    inventory = defaultdict(dict)
    for row in read_csv(run/'speaker_inventory.csv'):
        if role := canonical_role(row['role']):
            inventory[row['session_uid']][row['speaker']] = role
    sessions = {}
    for path in sorted((run/'full_session_diarization').glob('*.json')):
        if path.name.endswith('.meta.json'):
            continue
        meta = read_json(path.with_suffix('.meta.json'))
        if sha256(path) != meta['turns_sha256']:
            raise ValueError(f'Diarization hash mismatch: {path}')
        raw = read_json(path)
        for t in raw:
            if not (math.isfinite(t['start']) and math.isfinite(t['end']) and 0 <= t['start'] < t['end'] <= meta['duration_sec'] + 0.001):
                raise ValueError(f'Invalid full-session turn: {path}')
        if summaries[path.stem]['config_signature'] != meta['config_signature']:
            raise ValueError(f'Diarization signature mismatch: {path}')
        sessions[path.stem] = {'turns': sorted(raw, key=lambda t: (t['start'], t['end'])), 'source': str(path.resolve()),
                               'sha256': meta['turns_sha256'], 'config_signature': meta['config_signature'],
                               'duration_sec': meta['duration_sec'], 'explicit_roles': inventory[path.stem],
                               'summary_status': summaries[path.stem]['status']}
    return sessions, run


def build_role_maps(sessions, base, artifacts):
    evidence = defaultdict(lambda: defaultdict(list))
    source_refs = defaultdict(dict)
    for row in base:
        if not row['llm_ready']:
            continue
        uid = row['segment_uid'].split('_seg')[0]
        for u in row['utterances']:
            if (role := canonical_role(u['speaker'])) and u['end_sec'] > u['start_sec']:
                evidence[uid]['amberscript_session_time_overlap'].append((u['start_sec'], u['end_sec'], role))
        source_refs[uid]['amberscript_session_time_overlap'] = row['transcript_source']
    profiles_path = artifacts/'therapist_profile_builder_v2/session_profiles.csv'
    for r in read_csv(profiles_path):
        if r.get('seed_qc_status') == 'PASS' and r.get('timestamp_saved_speaker_agree', '').lower() == 'true':
            uid = f"{r['patient_id']}_S{int(r['session_id'])}"
            evidence[uid]['manual_therapist_seed_time_overlap'].append((float(r['seed_absolute_start_sec']), float(r['seed_absolute_end_sec']), 'T'))
            source_refs[uid]['manual_therapist_seed_time_overlap'] = str(profiles_path.resolve())
    legacy, reviews, _ = load_timelines(artifacts)
    for uid, data in legacy.items():
        for t in data['turns']:
            # Old manually disputed intervals cannot vote in the new mapping.
            disputed = any(r['segment_uid'].startswith(uid+'_seg') and r.get('global_result_supported', '').lower() != 'true'
                           and float(r['start_sec']) < t['end'] and float(r['end_sec']) > t['start'] for r in reviews.values())
            if not disputed:
                evidence[uid]['legacy_session_time_overlap'].append((t['start'], t['end'], t['role']))
        source_refs[uid]['legacy_session_time_overlap'] = data['source']
    for row in base:
        if not row['in_audio_inventory'] or row['mapping_status'] != 'PROVISIONAL_TWO_SPEAKER':
            continue
        if float(row.get('similarity_margin') or 0) < 0.08:
            continue
        roles = {row['therapist_local_speaker']: 'T', row['patient_local_speaker']: 'P'}
        if '' in roles or len(roles) != 2:
            continue
        uid = row['segment_uid'].split('_seg')[0]
        path = artifacts/'therapist_role_mapping/diarization'/f"{row['segment_uid']}.json"
        if not path.exists():
            continue
        for t in read_json(path):
            if t['speaker'] in roles:
                start = max(float(row['start_sec']), float(row['start_sec'])+t['start'])
                end = min(float(row['end_sec']), float(row['start_sec'])+t['end'])
                if end > start:
                    evidence[uid]['provisional_local_mapping_session_consensus'].append((start, end, roles[t['speaker']]))
        source_refs[uid]['provisional_local_mapping_session_consensus'] = str((artifacts/'therapist_role_mapping/therapist_role_mapping_all_segments.csv').resolve())
    priority = ['manual_therapist_seed_time_overlap', 'amberscript_session_time_overlap', 'legacy_session_time_overlap', 'provisional_local_mapping_session_consensus']
    reports = []
    for uid, session in sessions.items():
        attempts = [infer_mapping(session['turns'], evidence[uid][method], method) for method in priority if evidence[uid][method]]
        explicit = session['explicit_roles']
        actual_speakers = {t['speaker'] for t in session['turns']}
        if set(explicit) == actual_speakers and set(explicit.values()) == {'T', 'P'}:
            chosen = {'mapping': explicit, 'method': 'explicit_speaker_inventory', 'agreement': 1.0}
            role_source = str(Path(session['source']).parents[1]/'speaker_inventory.csv')
        else:
            chosen = next((a for a in attempts if a['mapping']), {'mapping': {}, 'method': 'UNRESOLVED', 'agreement': 0.0})
            role_source = source_refs[uid].get(chosen['method'], '')
        conflicts = [a['method'] for a in attempts if a['mapping'] and a['mapping'] != chosen['mapping']]
        session.update(role_mapping=chosen['mapping'], role_mapping_method=chosen['method'], role_mapping_source=role_source,
                       role_mapping_agreement=chosen['agreement'], role_mapping_conflicts=conflicts)
        reports.append({'session_uid': uid, 'mapping_status': 'PROVISIONAL' if chosen['mapping'] else 'UNRESOLVED',
                        'mapping_method': chosen['method'], 'role_mapping': json.dumps(chosen['mapping']),
                        'mapping_agreement': chosen['agreement'], 'conflicting_evidence': ';'.join(conflicts),
                        'role_mapping_source': role_source, 'diarization_source': session['source'], 'diarization_sha256': session['sha256'],
                        'config_signature': session['config_signature'], 'evidence': json.dumps(attempts),
                        'role_mapping_is_ground_truth': False})
    return reports


def assign_cue(start, end, session):
    overlap = {}
    for speaker in {t['speaker'] for t in session['turns']}:
        overlap[speaker] = union_duration([(max(start, t['start']), min(end, t['end'])) for t in session['turns']
                                          if t['speaker'] == speaker and t['start'] < end and t['end'] > start])
    ordered = sorted(overlap, key=overlap.get, reverse=True)
    best = ordered[0] if ordered else None
    total = sum(overlap.values())
    coverage = overlap.get(best, 0)/(end-start) if end > start else 0
    purity = overlap.get(best, 0)/total if total else 0
    assigned = best if coverage >= 0.5 and purity >= 0.8 else None
    role = session['role_mapping'].get(assigned)
    return {'speaker': role or 'UNKNOWN', 'global_speaker': assigned,
            'role_coverage': coverage, 'role_purity': purity, 'speaker_overlap_sec': overlap,
            'role_assignment_status': 'PROVISIONAL_FULL_SESSION_ROLE' if role else 'UNRESOLVED_SESSION_ROLE' if assigned else 'AMBIGUOUS_OR_NO_DIARIZATION_OVERLAP'}


def voxtral(row, path, session):
    data = read_json(path)
    if 'voxtral' not in str(data.get('model', '')).lower():
        raise ValueError(f'Unexpected model in Voxtral input: {path}')
    cues = data['segments']
    cleaned, diagnostics = remove_parentheses('\x1e'.join(strip_markup(str(c.get('text', ''))) for c in cues))
    offset, duration = float(row['start_sec']), float(row['end_sec'])-float(row['start_sec'])
    utterances, flags = [], []
    if diagnostics['unmatched_open_parentheses']:
        flags.append('REVIEW_CLEANING')
    if data.get('transcription_metadata', {}).get('prediction_anomalous'):
        flags.append('REVIEW_ASR_ANOMALY')
    previous = -math.inf
    for i, (cue, text) in enumerate(zip(cues, cleaned.split('\x1e')), 1):
        text = normalize_space(STAMP_NOISE.sub('', text))
        if not re.search(r'\w', text):
            continue
        start, end = float(cue['start']), float(cue['end'])
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f'Nonfinite ASR timestamp: {path}')
        valid = 0 <= start < end <= duration + 0.001 and start >= previous
        if not valid:
            flags.append('REVIEW_TIMING')
        previous = start
        assignment = assign_cue(offset+start, offset+end, session) if valid and session else {'speaker': 'UNKNOWN', 'role_assignment_status': 'INVALID_TIME_OR_MISSING_DIARIZATION'}
        utterances.append({'cue_id': i, 'start_sec': offset+start, 'end_sec': offset+end, 'segment_local_start_sec': start,
                           'segment_local_end_sec': end, 'text': text, 'original_asr_role': cue.get('speaker'),
                           'original_asr_raw_speaker': cue.get('raw_speaker'), **assignment})
    if not session:
        flags.append('MISSING_FULL_SESSION_DIARIZATION')
    elif not session['role_mapping']:
        flags.append('REVIEW_SESSION_ROLE_MAPPING')
    if any(u['speaker'] == 'UNKNOWN' for u in utterances):
        flags.append('REVIEW_UTTERANCE_ROLES')
    if not utterances:
        flags.append('NO_TRANSCRIPT_TEXT')
    flags = list(dict.fromkeys(flags))
    return {'utterances': utterances, 'transcript_source': str(path.resolve()), 'transcript_source_sha256': sha256(path),
            'asr_model': data.get('model'), 'transcript_provider': 'voxtral', 'transcript_status': flags[0] if flags else 'READY',
            'review_flags': ';'.join(flags+['ASR_TEXT_UNVERIFIED', 'FULL_SESSION_ROLES_PROVISIONAL']),
            'llm_ready': not flags, 'timing_method': 'asr_local_cue_plus_segment_offset',
            'alignment_policy': 'segment_uid_then_absolute_time_overlap_with_full_session_diarization',
            'transcript_speaker_source': 'pyannote_full_session_time_overlap',
            'unresolved_utterances': sum(u['speaker'] == 'UNKNOWN' for u in utterances)}


def build(args):
    output = args.dataset.resolve()
    input_path = output/'before_memopsy_completion/llm_segments_all.jsonl'
    base = [json.loads(line) for line in input_path.read_text(encoding='utf-8').splitlines()]
    sessions, run = load_new_sessions(args.diarization)
    reports = build_role_maps(sessions, base, args.artifacts)
    # Write an early role audit so an interrupted run remains inspectable.
    write_csv(output/'pyannote_session_role_mapping.csv', reports)
    rows, choices = [], []
    for old in base:
        row = copy.deepcopy(old)
        uid = row['segment_uid'].split('_seg')[0]
        session = sessions.get(uid)
        row['transcript_provider'] = 'amberscript'
        row['previous_amberscript_status'] = old['transcript_status']
        if row['in_audio_inventory'] and not row['llm_ready']:
            paths = list((args.artifacts/'memopsy_segments'/f"{row['segment_uid']}_sync_ctc").glob(f"{row['segment_uid']}_*.json"))
            if len(paths) > 1:
                raise ValueError(f"Ambiguous Voxtral source: {row['segment_uid']}")
            if paths:
                replacement = voxtral(row, paths[0], session)
                # Keep available nonempty human text when the ASR output is empty.
                if replacement['utterances'] or not row['utterances']:
                    row.update(replacement)
            choices.append({'segment_uid': row['segment_uid'], 'previous_status': old['transcript_status'],
                            'transcript_provider': row['transcript_provider'], 'transcript_status': row['transcript_status']})
        if row['transcript_provider'] == 'amberscript':
            for u in row['utterances']:
                u['original_transcript_speaker'] = u['speaker']
                # Noncanonical labels cannot be renamed to T/P without evidence.
                u['speaker'] = canonical_role(u['speaker']) or 'UNKNOWN'
                if u['speaker'] == 'UNKNOWN' and session and u['end_sec'] > u['start_sec']:
                    u.update(assign_cue(u['start_sec'], u['end_sec'], session))
                u['segment_local_start_sec'] = u['start_sec']-float(row['start_sec'])
                u['segment_local_end_sec'] = u['end_sec']-float(row['start_sec'])
            if any(u['speaker'] == 'UNKNOWN' for u in row['utterances']):
                row['llm_ready'] = False
                row['transcript_status'] = 'REVIEW_UTTERANCE_ROLES'
                row['review_flags'] += ';REVIEW_UTTERANCE_ROLES'
        row.update(timestamp_reference='full_session', timestamp_format='MM:SS.s', role_mapping_is_ground_truth=False,
                   diarization_provider='pyannote' if session else '', full_session_diarization_source=session['source'] if session else '',
                   full_session_diarization_sha256=session['sha256'] if session else '',
                   full_session_role_mapping_method=session['role_mapping_method'] if session else '',
                   full_session_role_mapping=json.dumps(session['role_mapping']) if session else '{}',
                   full_session_role_mapping_source=session['role_mapping_source'] if session else '')
        if session and session['role_mapping_conflicts']:
            row['review_flags'] += ';CONFLICTING_SESSION_ROLE_EVIDENCE'
        row['transcript_text'] = timed_dialogue(row['utterances'])
        row['transcript_text_plain'] = '\n'.join(f"{u['speaker']}: {u['text']}" for u in row['utterances'])
        row['word_count'] = sum(len(u['text'].split()) for u in row['utterances'])
        row['boundary_crossing_cues'] = sum(u['start_sec'] < float(row['start_sec']) or u['end_sec'] > float(row['end_sec']) for u in row['utterances'])
        row['text_available'] = bool(row['utterances'])
        row['all_roles_resolved'] = bool(row['utterances']) and all(u['speaker'] in {'T', 'P'} for u in row['utterances'])
        row['transcript_path'] = str(output/'segments'/f"{row['segment_uid']}.txt") if row['utterances'] else ''
        rows.append(row)
    snapshot = output/'before_pyannote_completion'
    if not snapshot.exists():
        snapshot.mkdir()
        for name in ('llm_segments_all.jsonl', 'dataset_summary.json'):
            shutil.copy2(output/name, snapshot/name)
    for row in rows:
        if row['transcript_path']:
            Path(row['transcript_path']).write_text(row['transcript_text']+'\n', encoding='utf-8')
    ready = [r for r in rows if r['llm_ready']]
    flat = [{k: v for k, v in row.items() if k != 'utterances'} for row in rows]
    for name, selected in [('llm_segments_all', rows), ('llm_segments_ready', ready),
                           ('inventory_all_segments', [r for r in rows if r['in_audio_inventory']]),
                           ('inventory_ready', [r for r in ready if r['in_audio_inventory']])]:
        write_jsonl(output/f'{name}.jsonl', selected)
        write_csv(output/f'{name}.csv', [{k: v for k, v in r.items() if k != 'utterances'} for r in selected])
    for provider in ('amberscript', 'voxtral'):
        write_csv(output/f'{provider}_segments.csv', [r for r in flat if r['transcript_provider'] == provider])
    prompt = SYSTEM_PROMPT.replace('Speaker labels are taken from the transcript export and have not been independently verified.',
                                  'Speaker roles may come from transcript labels or provisional full-session diarization mappings and are not ground truth.')
    prompt += '\nEach [MM:SS.s] timestamp is measured from the beginning of the full session, not the segment.'
    write_jsonl(output/'llm_requests.jsonl', [{'segment_uid': r['segment_uid'], 'segment_idx': r['segment_idx'],
        'transcript_provider': r['transcript_provider'], 'messages': [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': r['transcript_text']}]} for r in ready])
    # Complete nonempty inputs are also available for explicitly including review
    # rows in an experiment; their statuses accompany the text rather than vanish.
    write_jsonl(output/'llm_requests_all_nonempty.jsonl', [{'segment_uid': r['segment_uid'], 'segment_idx': r['segment_idx'],
        'transcript_provider': r['transcript_provider'], 'llm_ready': r['llm_ready'], 'transcript_status': r['transcript_status'],
        'review_flags': r['review_flags'], 'messages': [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': r['transcript_text']}]} for r in rows if r['text_available']])
    write_csv(output/'segments_needing_review.csv', [r for r in flat if not r['llm_ready'] or r['review_flags']])
    write_csv(output/'pyannote_completion_audit.csv', choices)
    # Session files reconstruct the chosen dataset windows and retain per-cue source provenance.
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['segment_uid'].split('_seg')[0]].extend([{**u, 'transcript_provider': row['transcript_provider'], 'segment_uid': row['segment_uid']} for u in row['utterances']])
    for uid, utterances in grouped.items():
        utterances.sort(key=lambda u: (u['start_sec'], u['end_sec']))
        (output/'sessions'/f'{uid}.txt').write_text(timed_dialogue(utterances)+'\n', encoding='utf-8')
    write_jsonl(output/'session_utterances.jsonl', [{'session_uid': uid, 'utterances': us} for uid, us in sorted(grouped.items())])
    inventory = [r for r in rows if r['in_audio_inventory']]
    inventory_session_ids = {r['segment_uid'].split('_seg')[0] for r in inventory}
    write_csv(output/'pyannote_unresolved_inventory_sessions.csv', [r for r in reports if r['session_uid'] in inventory_session_ids and r['mapping_status'] == 'UNRESOLVED'])
    summary = {'inventory_segments': len(inventory), 'all_segments': len(rows), 'ready_segments': len(ready),
               'inventory_ready_segments': sum(r['llm_ready'] for r in inventory),
               'inventory_text_available': sum(r['text_available'] for r in inventory),
               'inventory_provider_counts': dict(Counter(r['transcript_provider'] for r in inventory)),
               'all_provider_counts': dict(Counter(r['transcript_provider'] for r in rows)),
               'inventory_status_counts': dict(Counter(r['transcript_status'] for r in inventory)),
               'all_status_counts': dict(Counter(r['transcript_status'] for r in rows)),
               'diarization_run': str(run.resolve()), 'diarization_sessions': len(sessions),
               'inventory_sessions_with_diarization': len({r['segment_uid'].split('_seg')[0] for r in inventory if r['full_session_diarization_source']}),
               'inventory_session_role_methods': dict(Counter(sessions[uid]['role_mapping_method'] for uid in sorted(inventory_session_ids))),
               'timestamp_reference': 'full_session', 'timestamp_format': '[MM:SS.s] T/P: text',
               'notes': ['Amberscript is preferred for previously ready windows; Voxtral supplies pending inventory windows when text exists.',
                         'UNASSIGNED pyannote clusters are mapped using explicit roles, seed intervals, Amberscript, legacy timelines, then provisional local-role consensus, in that order.',
                         'Every inferred T/P assignment remains provisional; speaker IDs are never assumed to have the same identity across diarization runs.',
                         'Unknown roles are retained as UNKNOWN in the complete dataset and excluded from ready inputs.',
                         'ASR cue assignments require at least 50% winning-speaker coverage and 80% overlap purity.',
                         'No timestamps are invented for empty transcripts; actual cue start/end times and segment-relative times are retained.',
                         'Session text files contain the selected dataset windows; full original human session files remain reproducible from the Amberscript builder.',
                         'No predictions were run.'], 'baseline_input': str(input_path), 'baseline_sha256': sha256(input_path)}
    (output/'dataset_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/amberscript_llm')
    parser.add_argument('--artifacts', type=Path, default=ROOT.parents[1]/'german-asr-pipeline/artifacts')
    parser.add_argument('--diarization', type=Path, default=ROOT.parents[1]/'german-asr-pipeline/artifacts/all_session_diarization')
    build(parser.parse_args())
