"""Prepare covered Cohere minutes and replay saved ordinal experiments."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]


def main(args):
    prepared=args.dataset_root/'cohere_new/replacement_audit.json'
    release=args.release_root
    if not (release/'release.json').exists():
        candidates=list(release.glob('*/release.json'))
        if len(candidates)!=1: raise ValueError('Cannot locate unique extracted release')
        release=candidates[0].parent
    if not prepared.exists():
        command=[sys.executable,'-u',str(REPO/'scripts/data/build_wd_cohere_available_subset.py'),
            '--release-root',str(release),'--original-master',str(args.original_master),
            '--output',str(args.dataset_root),'--turn-style',args.turn_style]
        if args.fill_unknown_roles: command.append('--fill-unknown-roles')
        subprocess.run(command,cwd=REPO,check=True)
    import hashlib
    audit=json.loads(prepared.read_text(encoding='utf-8'))
    expected_fill='same_role_bracket_v1' if args.fill_unknown_roles else 'none'
    if audit.get('role_fill_method','none')!=expected_fill or audit.get('turn_style','plain')!=args.turn_style:
        raise ValueError('Prepared dataset uses different role-fill/format settings; choose a fresh DatasetRoot')
    if audit['source_release_sha256']!=hashlib.sha256((release/'release.json').read_bytes()).hexdigest():
        raise ValueError('Prepared dataset belongs to another release')
    for fold in range(1,6):
        source=args.original_master/f'fold_{fold}/master_manifest.jsonl'
        if audit['original_fold_sha256'][str(fold)]!=hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError('Original folds changed since preparation')
    conditions=[('cohere_new','cohere_available_new')]
    if args.with_original_control: conditions.append(('original_subset','cohere_available_original'))
    for condition,tag in conditions:
        subprocess.run([sys.executable,'-u',str(REPO/'scripts/orchestration/run_wd_cohere_ordinal_replay.py'),
            '--master-root',str(args.dataset_root/condition),'--output',str(args.queue_root/condition),
            '--cohort-tag',tag,'--run'],cwd=REPO,check=True)
    print('Cohere available-subset suite finished',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-root',type=Path,required=True)
    p.add_argument('--original-master',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--queue-root',type=Path,required=True)
    p.add_argument('--with-original-control',action='store_true')
    p.add_argument('--fill-unknown-roles',action='store_true')
    p.add_argument('--turn-style',choices=['plain','timestamped_cues'],default='plain')
    main(p.parse_args())
