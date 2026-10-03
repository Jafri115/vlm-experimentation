"""Inspect an ASR release or replace transcripts in the frozen 4,325-row cohort.

No full-session text is assigned to minutes without validated timing. Replacement
input is segment-level JSONL/CSV with the original segment_uid and transcript_text.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def inspect_archive(args):
    destination = args.output.resolve()
    if destination.exists():
        raise ValueError(f'Use a new extraction directory: {destination}')
    with tarfile.open(args.archive, 'r:gz') as archive:
        members = archive.getmembers()
        if sum(m.size for m in members) > 20 * 1024 ** 3:
            raise ValueError('Archive exceeds 20 GiB extraction limit')
        names = set()
        for member in members:
            target = (destination / member.name).resolve()
            if (not target.is_relative_to(destination) or ':' in member.name
                    or member.issym() or member.islnk() or not (member.isfile() or member.isdir())):
                raise ValueError(f'Unsafe archive member: {member.name}')
            if str(target).casefold() in names:
                raise ValueError(f'Duplicate archive destination: {member.name}')
            names.add(str(target).casefold())
        destination.mkdir(parents=True)
        # Only regular files/directories were approved; do not apply archived metadata.
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('wb') as out:
                    import shutil
                    shutil.copyfileobj(source, out)
    files = [p for p in destination.rglob('*') if p.is_file()]
    candidates = []
    for path in files:
        if path.suffix.lower() not in {'.csv', '.jsonl', '.json'}:
            continue
        record = {'path': str(path.relative_to(destination))}
        try:
            if path.suffix.lower() == '.json':
                obj = json.loads(path.read_text(encoding='utf-8-sig'))
                record['top_level'] = list(obj) if isinstance(obj, dict) else type(obj).__name__
            else:
                rows = read_rows(path)
                record.update(rows=len(rows), columns=list(rows[0]) if rows else [])
        except Exception as exc:
            record['inspection_error'] = type(exc).__name__
        candidates.append(record)
    report = {'archive_sha256': digest(args.archive), 'files': len(files),
              'extensions': dict(Counter(p.suffix.lower() for p in files)),
              'tables': candidates}
    write_json(destination / 'archive_inspection.json', report)
    print(json.dumps(report, indent=2))


def index(rows):
    result = {}
    for row in rows:
        uid = row.get('segment_uid')
        if not uid or uid in result:
            raise ValueError(f'Missing or duplicate segment_uid: {uid}')
        result[uid] = row
    return result


def true(value):
    return value is True or str(value).lower() in {'true', '1'}


def from_directory(args):
    """Discover explicitly identified, segment-level Cohere FT tables.

    Session-level text is inventoried but never treated as aligned minute text.
    Multiple copies of the same segment are accepted only if they agree.
    """
    if not args.release_root.is_dir():
        raise ValueError(f'Release directory missing: {args.release_root}')
    inventory = []
    selected = {}
    sources = []
    for path in sorted(args.release_root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.jsonl', '.csv'}:
            continue
        relative = str(path.relative_to(args.release_root))
        try:
            rows = read_rows(path)
        except Exception as exc:
            inventory.append({'path': relative, 'error': type(exc).__name__})
            continue
        columns = list(rows[0]) if rows else []
        inventory.append({'path': relative, 'rows': len(rows), 'columns': columns})
        if not {'segment_uid', 'transcript_text', 'start_sec', 'end_sec'}.issubset(columns):
            continue
        contributed = False
        for row in rows:
            # Provider must be explicit; directory names alone are not provenance.
            if row.get('transcript_provider') != 'cohere_finetuned':
                continue
            uid = row['segment_uid']
            fields = ('segment_uid', 'transcript_text', 'transcript_text_plain',
                      'start_sec', 'end_sec', 'transcript_provider', 'timing_validated',
                      'speaker_roles_validated', 'asr_evaluation_role')
            replacement = {k: row[k] for k in fields if k in row}
            if uid in selected and selected[uid] != replacement:
                raise ValueError(f'Conflicting Cohere transcript/metadata for {uid}: {relative}')
            selected[uid] = replacement
            contributed = True
        if contributed:
            sources.append({'path': relative, 'sha256': digest(path)})
    write_json(args.output / 'release_inventory.json', {
        'release_root': str(args.release_root.resolve()), 'tables': inventory,
        'selected_segment_count': len(selected), 'selected_sources': sources})
    if not selected:
        raise ValueError('No explicitly identified, aligned Cohere fine-tuned segment table found. '
                         f'See {args.output / "release_inventory.json"}; training was not started. '
                         'Full-session text needs timing and speaker mapping before replacement.')
    args.replacements = args.output / 'cohere_ft_replacement_segments.jsonl'
    write_rows(args.replacements, list(selected.values()))
    build(args)


def build(args):
    original = read_rows(args.master_root / 'paired_master_soft.jsonl')
    old = index(original)
    if len(old) != 4325 or len({r['patient_id'] for r in original}) != 20:
        raise ValueError('Expected the original expanded cohort: 4,325 segments and 20 patients')
    replacements = index(read_rows(args.replacements))
    missing = sorted(set(old) - set(replacements))
    if missing:
        write_json(args.output / 'coverage_failure.json', {'missing_count': len(missing), 'missing_segment_uids': missing})
        raise ValueError(f'{len(missing)} original segments lack new transcripts; cohort was NOT reduced')
    updates = {}
    errors = []
    replacement_hash = digest(args.replacements)
    for uid, row in old.items():
        new = replacements[uid]
        text = str(new.get('transcript_text', '')).strip()
        if not text or not true(new.get('timing_validated')) or not true(new.get('speaker_roles_validated')):
            errors.append({'segment_uid': uid, 'reason': 'Needs nonempty text, timing_validated and speaker_roles_validated'})
            continue
        for key in ('start_sec', 'end_sec'):
            if key not in new or abs(float(new[key]) - float(row[key])) > .001:
                raise ValueError(f'{uid}: rating window changed or missing {key}')
        provider = new.get('transcript_provider', '')
        if provider != 'cohere_finetuned':
            raise ValueError(f'{uid}: expected cohere_finetuned, got {provider!r}')
        updates[uid] = dict(transcript_text=text,
                            transcript_text_plain=new.get('transcript_text_plain', text),
                            transcript_provider=provider, transcript_status='ready',
                            llm_ready=True,
                            transcript_replacement_source_sha256=replacement_hash,
                            transcript_asr_evaluation_role=new.get('asr_evaluation_role', 'unknown'))
    if errors:
        write_json(args.output / 'readiness_failure.json', errors)
        raise ValueError(f'{len(errors)} transcripts are not ready; no training manifests written')
    fold_outputs = []
    test_counts = Counter()
    hashes = {}
    for fold in range(1, 6):
        source = args.master_root / f'fold_{fold}' / 'master_manifest.jsonl'
        rows = read_rows(source)
        if set(index(rows)) != set(old):
            raise ValueError(f'Fold {fold} does not contain the frozen cohort')
        patients = {s: {r['patient_id'] for r in rows if r['split'] == s} for s in ('train', 'val', 'test')}
        if set.union(*patients.values()) != {r['patient_id'] for r in original}:
            raise ValueError(f'Fold {fold}: unknown/missing split')
        if any(patients[a] & patients[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
            raise ValueError(f'Fold {fold}: patient leakage')
        for row in rows:
            uid = row['segment_uid']
            for key in ('patient_id', 'WD_P_rater1', 'WD_P_rater2', 'start_sec', 'end_sec'):
                if row[key] != old[uid][key]:
                    raise ValueError(f'{uid}: master/fold mismatch for {key}')
            if row['split'] == 'test':
                test_counts[uid] += 1
        fold_outputs.append((fold, [{**r, **updates[r['segment_uid']]} for r in rows]))
        hashes[str(fold)] = digest(source)
    if set(test_counts) != set(old) or set(test_counts.values()) != {1}:
        raise ValueError('Every original segment must occur in exactly one outer test fold')
    if any(args.output.glob('fold_*/master_manifest.jsonl')):
        raise ValueError('Output already contains training manifests; use a fresh output root')
    for fold, rows in fold_outputs:
        write_rows(args.output / f'fold_{fold}' / 'master_manifest.jsonl', rows)
    write_rows(args.output / 'paired_master_soft.jsonl', [{**r, **updates[r['segment_uid']]} for r in original])
    summary = {'rows': len(original), 'patients': 20, 'provider': 'cohere_finetuned',
               'unchanged': 'segment IDs, rating windows, human labels, row order and patient folds',
               'original_fold_sha256': hashes, 'replacement_sha256': digest(args.replacements),
               'changed_transcripts': sum(old[u]['transcript_text'] != updates[u]['transcript_text'] for u in old),
               'asr_exposure': dict(Counter(v['transcript_asr_evaluation_role'] for v in updates.values())),
               'note': 'This is the fixed-cohort ASR comparison; ASR exposure must be reported separately.'}
    write_json(args.output / 'replacement_audit.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('inspect')
    a.add_argument('--archive', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.set_defaults(function=inspect_archive)
    a = sub.add_parser('build')
    a.add_argument('--master-root', type=Path, required=True)
    a.add_argument('--replacements', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.set_defaults(function=build)
    a = sub.add_parser('from-directory')
    a.add_argument('--release-root', type=Path, required=True)
    a.add_argument('--master-root', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.set_defaults(function=from_directory)
    args = p.parse_args()
    args.function(args)
