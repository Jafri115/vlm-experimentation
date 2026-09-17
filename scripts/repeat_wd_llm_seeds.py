"""Controlled soft-label LLM repeats; uses each existing fold's saved settings."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from wd_presentation_common import load_folds, sha, write_json, exclusive_lock


def main(args):
    import finetune_qwen3_8b_wd_text as training
    load_folds(args.master_root)
    allowed={a.dest for a in training.make_parser()._actions if a.dest!='help'}
    jobs=[]
    for seed in args.seeds:
        for fold in range(1,6):
            source=args.source_root/f'fold_{fold}'/'run_config.json'
            config=json.loads(source.read_text(encoding='utf-8-sig'))
            config.setdefault('context_input',False)
            config.setdefault('pooling','mean_all')
            if config['mode']!='soft' or config.get('system_prompt')!=training.SYSTEM_PROMPT:
                raise ValueError(f'{source}: requires original soft run and same system prompt')
            missing=allowed-set(config)
            if missing:
                raise ValueError(f'{source}: missing saved settings: {missing}')
            config={k:v for k,v in config.items() if k in allowed}
            output=args.output/f'seed_{seed}'/f'fold_{fold}'
            dataset=args.master_root/f'fold_{fold}'/'master_manifest.jsonl'
            config.update(seed=seed,dataset=dataset.resolve(),output=output.resolve(),prepare_only=False)
            # Also validate the actual JSONL consumed by training; CSV validation alone is insufficient.
            prepared=training.prepare_rows(training.read_jsonl(dataset),'soft')
            csv=pd.read_csv(args.master_root/f'fold_{fold}'/'master_manifest.csv')
            items=pd.DataFrame(prepared)
            for split in ['train','val','test']:
                if set(items.loc[items.split==split,'segment_uid'])!=set(csv.loc[csv.split==split,'segment_uid']):
                    raise ValueError(f'{dataset}: CSV/JSONL split mismatch')
            for col in ['WD_P_rater1','WD_P_rater2','WD_soft','WD_P_mean']:
                aligned=items[['segment_uid',col]].merge(csv[['segment_uid',col]],on='segment_uid',suffixes=('_json','_csv'),validate='one_to_one')
                if not (pd.to_numeric(aligned[col+'_json'])==pd.to_numeric(aligned[col+'_csv'])).all():
                    raise ValueError(f'{dataset}: CSV/JSONL {col} differs')
            fingerprint=hashlib.sha256(json.dumps({'config':config,'dataset_sha':sha(dataset),
                'training_code_sha':sha(Path(training.__file__))},sort_keys=True,default=str).encode()).hexdigest()
            marker=output/'repeat_complete.json'
            complete=marker.exists() and json.loads(marker.read_text())['fingerprint']==fingerprint
            required=['test_predictions.csv','final_summary.json','best_head.pt','best_adapter/adapter_config.json']
            complete=complete and all((output/p).exists() for p in required)
            jobs.append((config,marker,fingerprint,complete))
    if args.dry_run:
        for config,_,_,complete in jobs:
            print(('SKIP' if complete else 'RUN'),config['output'])
        return
    args.output.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(args.output/'training.lock'):
        for config,marker,fingerprint,complete in jobs:
            if complete:
                print('SKIP',config['output'],flush=True);continue
            if marker.exists():
                raise ValueError(f'{marker}: settings changed; use a new output directory')
            print('TRAIN',config['output'],flush=True)
            training.main(argparse.Namespace(**config))
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            write_json(marker,{'fingerprint':fingerprint,'torch_seeded':True,'seed':config['seed']})
        for seed in args.seeds:
            frames=[]
            for fold in range(1,6):
                p=pd.read_csv(args.output/f'seed_{seed}'/f'fold_{fold}'/'test_predictions.csv')
                p['outer_fold']=fold;frames.append(p)
            all_rows=pd.concat(frames,ignore_index=True)
            if all_rows.segment_uid.duplicated().any():
                raise ValueError('Duplicate OOF IDs')
            all_rows.to_csv(args.output/f'seed_{seed}'/'oof_predictions.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--master-root',type=Path,default=Path('output/wd_multimodal_master_repaired'))
    p.add_argument('--source-root',type=Path,default=Path('output/llm_wd_soft_repaired_cv'))
    p.add_argument('--output',type=Path,default=Path('output/wd_presentation_3day/seeds'))
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--dry-run',action='store_true')
    main(p.parse_args())
