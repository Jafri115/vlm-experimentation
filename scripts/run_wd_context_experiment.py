"""Controlled transcript-context experiments; sequential GPU jobs in fresh folders."""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from wd_presentation_common import exclusive_lock, sha, write_json, load_folds


def main(args):
    import finetune_qwen3_8b_wd_text as training
    from prepare_wd_context_experiment import prepare, export_review
    from wd_context_inputs import CONTEXT_INSTRUCTION, encode_row
    root=args.output.resolve(); master=args.master_root.resolve()
    repo=Path(__file__).resolve().parents[1]
    root.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(root/'queue.lock'):
        def log(message):
            line=f'[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {message}'
            print(line,flush=True)
            with (root/'queue.log').open('a',encoding='utf-8') as f:f.write(line+'\n')
        sources=[]
        allowed={a.dest for a in training.make_parser()._actions if a.dest!='help'}
        for fold in range(1,6):
            p=args.source_root/f'fold_{fold}'/'run_config.json'
            c=json.loads(p.read_text(encoding='utf-8-sig'))
            c.setdefault('context_input',False);c.setdefault('pooling','mean_all')
            if c['mode']!='consensus' or c.get('system_prompt')!=training.SYSTEM_PROMPT:
                raise ValueError(f'{p}: expected original consensus model settings')
            if allowed-set(c):raise ValueError(f'{p}: missing settings {allowed-set(c)}')
            sources.append({k:v for k,v in c.items() if k in allowed})
        # Refuse changed inputs/settings before regenerating any existing experiment.
        _,folds=load_folds(master)
        signature={'sources':sources,'max_length':args.max_length,'seed':args.seed,
            'manifests':[sha(p) for _,_,p in folds],
            'code':{name:sha(repo/'scripts'/name) for name in [
                'run_wd_context_experiment.py','prepare_wd_context_experiment.py',
                'wd_context_inputs.py','finetune_qwen3_8b_wd_text.py','run_wd_tfidf_baselines.py']}}
        fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        provenance=root/'experiment.json'
        if provenance.exists() and json.loads(provenance.read_text())['fingerprint']!=fingerprint:
            raise ValueError('Experiment settings or code changed. Use a NEW --output directory.')
        write_json(provenance,{'fingerprint':fingerprint,**signature})
        log('Preparing paired inputs and TF-IDF baselines')
        prepare(master,root)
        export_review(master,root,args.source_root)
        if args.prepare_only:
            log('CPU preparation complete; no model loaded');return
        from transformers import AutoTokenizer
        variants=[('target_only','target_only','mean_all'),('previous_context','previous_context','mean_all')]
        if args.patient_pooling:variants.append(('previous_context_patient','previous_context','target_patient'))
        jobs=[];tokenizers={};audit=[]
        for name,dataset_variant,pooling in variants:
            for fold,source in enumerate(sources,1):
                c=dict(source)
                c.update(dataset=str(root/'datasets'/dataset_variant/f'fold_{fold}'/'master_manifest.jsonl'),
                    output=str(root/'llm'/name/f'fold_{fold}'),context_input=True,pooling=pooling,
                    max_length=args.max_length,eval_batch_size=1,seed=args.seed,prepare_only=False)
                key=(c['model'],c['revision'])
                if key not in tokenizers:tokenizers[key]=AutoTokenizer.from_pretrained(*key[:1],revision=key[1])
                rows=training.prepare_rows(training.read_jsonl(Path(c['dataset'])),'consensus')
                for row in rows:
                    enc=encode_row(tokenizers[key],row,training.SYSTEM_PROMPT+CONTEXT_INSTRUCTION,args.max_length,pooling)
                    audit.append({'variant':name,'fold':fold,'segment_uid':row['segment_uid'],
                        'split':row['split'],'tokens':enc['token_count'],'pooling_fallback':enc['pooling_fallback']})
                jobs.append((name,fold,c))
        pd.DataFrame(audit).to_csv(root/'token_preflight.csv',index=False)
        log('All token lengths and pooling masks passed; starting sequential training')
        logs=root/'logs';logs.mkdir(exist_ok=True)
        for name,fold,c in jobs:
            out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
            marker=out/'complete.json'
            required=['test_predictions.csv','final_summary.json','best_head.pt','best_adapter/adapter_config.json']
            if marker.exists() and all((out/p).exists() for p in required):
                if json.loads(marker.read_text())['fingerprint']!=fingerprint:raise ValueError('Stale completion marker')
                log(f'SKIP {name} fold {fold}');continue
            request=out/'request.json';write_json(request,c)
            log(f'START {name} fold {fold}')
            with (logs/f'{name}_fold_{fold}.log').open('a',encoding='utf-8') as f:
                proc=subprocess.run([sys.executable,'-u',str(repo/'scripts'/'train_wd_context_fold.py'),str(request)],
                    cwd=repo,stdout=f,stderr=subprocess.STDOUT)
            if proc.returncode:raise RuntimeError(f'{name} fold {fold} failed; see {logs}. Queue stopped.')
            if not all((out/p).exists() for p in required):raise RuntimeError('Training returned without expected artifacts')
            write_json(marker,{'fingerprint':fingerprint});log(f'COMPLETED {name} fold {fold}')
        for name,_,_ in variants:
            frames=[]
            for fold in range(1,6):
                d=pd.read_csv(root/'llm'/name/f'fold_{fold}'/'test_predictions.csv')
                d['outer_fold']=fold;frames.append(d)
            d=pd.concat(frames,ignore_index=True)
            if d.segment_uid.duplicated().any():raise ValueError('Duplicate OOF predictions')
            d.to_csv(root/'llm'/name/'oof_predictions.csv',index=False)
        from report_wd_context_experiment import report
        report(master,root)
        log('Finished; report.md and metrics.csv are ready')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--master-root',type=Path,default=Path('output/wd_multimodal_master_repaired'))
    p.add_argument('--source-root',type=Path,default=Path('output/llm_wd_consensus_repaired_cv'))
    p.add_argument('--output',type=Path,default=Path('output/wd_context_experiment'))
    p.add_argument('--max-length',type=int,default=4096)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--patient-pooling',action='store_true')
    main(p.parse_args())
