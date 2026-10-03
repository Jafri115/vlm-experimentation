"""Replay saved 8B/14B ordinal configurations with only a new transcript dataset.

Run from the repository root. Default is planning only; --run trains sequentially.
The archived system prompt overrides current rubric text, preventing prompt drift.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / 'scripts/llm/finetune_qwen3_8b_wd_text.py'
CONFIGS = REPO / 'scripts/configs/wd_ordinal_expanded_original'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def worker(args):
    saved = read(args.config)
    namespace = runpy.run_path(str(TRAINER))
    parser = namespace['make_parser']()
    parsed = parser.parse_args(['--dataset', str(args.master_root), '--mode', 'ordinal', '--output', str(args.output)])
    for key, value in saved.items():
        if hasattr(parsed, key):
            setattr(parsed, key, value)
    parsed.dataset = args.master_root
    parsed.output = args.output
    parsed.prepare_only = False
    namespace['SYSTEM_PROMPTS'][saved['rubric']] = saved['system_prompt']
    namespace['main'](parsed)


def main(args):
    args.master_root = args.master_root.resolve()
    args.output = args.output.resolve()
    audit = read(args.master_root / 'replacement_audit.json')
    if audit['rows'] != 4325 and not (audit.get('subset_authorized') and audit.get('original_rows')==4325):
        raise ValueError('Replacement audit does not describe the full or explicitly authorized subset cohort')
    jobs = []
    for tag in args.models:
        for fold in range(1, 6):
            config = CONFIGS / tag / f'fold_{fold}.json'
            saved = read(config)
            if saved['mode'] != 'ordinal' or saved['context_input'] or saved['prepare_only']:
                raise ValueError(f'Unexpected baseline configuration: {config}')
            manifest = args.master_root / f'fold_{fold}/master_manifest.jsonl'
            data = rows(manifest)
            if len(data) != audit['rows'] or any(not r['transcript_text'].strip() for r in data):
                raise ValueError(f'Incomplete manifest: {manifest}')
            out = args.output / f'llm_wd_ordinal_{tag}_{args.cohort_tag}_cv' / f'fold_{fold}'
            identity = {'baseline_config_sha256': sha(config), 'new_manifest_sha256': sha(manifest),
                        'training_script_sha256': sha(TRAINER), 'model': saved['model'],
                        'saved_revision': saved['revision']}
            jobs.append((config, manifest, out, identity))
    write(args.output / 'replay_plan.json', {
        'jobs': [{'output': str(o), **i} for _, _, o, i in jobs],
        'change': 'transcripts and transcript provenance only',
        'objective': 'soft ordinal cross-entropy; expected 1-5 score',
        'limitation': 'Original revision was main, not an immutable model commit. '
                      'Use the original model cache/environment for closest replication; current trainer hash is recorded.'})
    print(f'Validated {len(jobs)} jobs; plan: {args.output / "replay_plan.json"}', flush=True)
    if not args.run:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / 'queue.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'Existing queue lock: {lock}; check its PID before restarting')
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        for config, manifest, out, identity in jobs:
            marker = out / 'replay_identity.json'
            done = out / 'final_summary.json'
            if marker.exists() and read(marker) != identity:
                raise ValueError(f'Output belongs to different input/config/code: {out}')
            if done.exists():
                if not marker.exists():
                    raise ValueError(f'Unverified existing completion: {out}')
                print(f'SKIP {out.name} {out.parent.name}', flush=True)
                continue
            write(marker, identity)
            log = out / 'training.log'
            print(f'START {out.parent.name}/{out.name}', flush=True)
            with log.open('a', encoding='utf-8') as handle:
                subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                    '--worker', '--config', str(config), '--master-root', str(manifest),
                    '--output', str(out)], cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)
            if not done.exists():
                raise RuntimeError(f'Missing final_summary.json: {out}')
            print(f'COMPLETED {out.parent.name}/{out.name}', flush=True)
        metrics = []
        for tag in args.models:
            root = args.output / f'llm_wd_ordinal_{tag}_{args.cohort_tag}_cv'
            oof = root / 'oof_predictions.csv'
            subprocess.run([sys.executable, str(REPO / 'scripts/analysis/combine_wd_cv_predictions.py'),
                            '--fold-root', str(root), '--output', str(oof)], cwd=REPO, check=True)
            import csv
            with oof.open(encoding='utf-8-sig', newline='') as f:
                predictions = list(csv.DictReader(f))
            errors = [float(r['WD_prediction']) - float(r['WD_P_mean']) for r in predictions]
            metrics.append({'model': tag, 'N': len(errors), 'MAE': sum(abs(e) for e in errors)/len(errors),
                            'RMSE': math.sqrt(sum(e*e for e in errors)/len(errors))})
        write(args.output / 'cohere_ordinal_metrics.json', metrics)
        print('All requested replay experiments completed', flush=True)
    finally:
        lock.unlink()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--master-root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=REPO / 'output/wd_cohere_ordinal_replay')
    p.add_argument('--models', nargs='+', choices=['qwen3_8b', 'qwen3_14b'], default=['qwen3_14b', 'qwen3_8b'])
    p.add_argument('--run', action='store_true')
    p.add_argument('--cohort-tag', default='expanded_cohere_ft', choices=['expanded_cohere_ft','cohere_available_new','cohere_available_original'])
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--config', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        worker(args)
    else:
        main(args)
