"""Run zero-shot or 3+3 few-shot WD_P inference on a VLM-aligned test split."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from llm.finetune_qwen3_8b_wd_text import binary_metrics, finite, read_jsonl
from run_qwen3_8b_wd_3rs_zeroshot import PROMPT, parse_generated


def select(rows, split, consensus_only):
    selected = [r for r in rows if str(r.get('split', '')).lower() == split]
    if consensus_only:
        selected = [r for r in selected if finite(r.get('WD_consensus'))]
    if not selected:
        raise ValueError(f'No eligible rows for split={split}')
    return selected


def choose_demos(rows, count, seed):
    pool = [r for r in rows if str(r.get('split', '')).lower() == 'train' and finite(r.get('WD_consensus'))]
    negative = [r for r in pool if int(float(r['WD_consensus'])) == 0]
    positive = [r for r in pool if int(float(r['WD_consensus'])) == 1]
    rng = random.Random(seed); rng.shuffle(negative); rng.shuffle(positive)
    if len(negative) < count or len(positive) < count:
        raise ValueError(f'Need at least {count} consensus examples from each class in train')
    demos = negative[:count]+positive[:count]; rng.shuffle(demos)
    return demos


def messages(row, demos):
    result = [{'role': 'system', 'content': PROMPT}]
    for demo in demos:
        mean = float(demo['WD_P_mean'])
        score = 1 if int(float(demo['WD_consensus'])) == 0 else max(2, min(5, int(mean+0.5)))
        result.extend([
            {'role': 'user', 'content': 'Example transcript:\n'+demo['transcript_text']},
            {'role': 'assistant', 'content': json.dumps({'wd_p_score': score})},
        ])
    result.append({'role': 'user', 'content': 'Transcript to classify:\n'+row['transcript_text']})
    return result


def main(args):
    rows = read_jsonl(args.dataset)
    selected = select(rows, args.split, args.consensus_only)
    demos = choose_demos(rows, args.examples_per_class, args.seed) if args.shot == 'few' else []
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    config = {'model': args.model, 'revision': args.revision, 'shot': args.shot, 'split': args.split,
              'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              'consensus_only': args.consensus_only, 'examples_per_class': len(demos)//2,
              'demonstration_segment_uids': [r['segment_uid'] for r in demos],
              'selected_segment_uids_sha256': hashlib.sha256('\n'.join(r['segment_uid'] for r in selected).encode()).hexdigest(),
              'selected_rows': len(selected), 'system_prompt': PROMPT, 'temperature': 0.0,
              'decision_rule': 'wd_p_score > 1', 'seed': args.seed}
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    config={**config,'config_hash':config_hash}; config_path=output/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')).get('config_hash')!=config_hash:
        raise ValueError('Existing output has a different configuration; choose another --output directory')
    config_path.write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    (output/'system_prompt.txt').write_text(PROMPT+'\n', encoding='utf-8')
    if demos:
        (output/'demonstrations.json').write_text(json.dumps(demos, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in config.items() if k!='system_prompt'},indent=2),flush=True)
    if args.prepare_only:
        print('PREPARE-ONLY complete; no model loaded.',flush=True); return

    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    if args.dtype=='bfloat16' and not torch.cuda.is_bf16_supported(): raise RuntimeError('Use --dtype float16')
    set_seed(args.seed); dtype=torch.bfloat16 if args.dtype=='bfloat16' else torch.float16
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision,padding_side='left')
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,torch_dtype=dtype,
                                              device_map={'':0},attn_implementation=args.attention).eval()
    checkpoint=output/'predictions.jsonl'; predictions={}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row.get('config_hash')!=config_hash: raise ValueError('Prediction checkpoint configuration mismatch')
            predictions[row['segment_uid']]=row
    pending=[r for r in selected if predictions.get(r['segment_uid'],{}).get('status')!='OK']
    generation={'max_new_tokens':args.max_new_tokens,'do_sample':False,
                'pad_token_id':tokenizer.pad_token_id,'eos_token_id':tokenizer.eos_token_id}
    with checkpoint.open('a',encoding='utf-8') as log:
        for start in range(0,len(pending),args.batch_size):
            batch=pending[start:start+args.batch_size]
            rendered=[tokenizer.apply_chat_template(messages(r,demos),tokenize=False,add_generation_prompt=True,
                                                     enable_thinking=False) for r in batch]
            lengths=[len(tokenizer(x,add_special_tokens=False)['input_ids']) for x in rendered]
            if max(lengths)>args.max_input_tokens:
                raise ValueError(f'Input exceeds token limit: {[r["segment_uid"] for r,n in zip(batch,lengths) if n>args.max_input_tokens]}')
            inputs=tokenizer(rendered,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda:0')
            torch.cuda.synchronize(); began=time.monotonic()
            with torch.inference_mode(): generated=model.generate(**inputs,**generation)
            torch.cuda.synchronize(); elapsed=time.monotonic()-began
            decoded=tokenizer.batch_decode(generated[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
            for source,raw in zip(batch,decoded):
                record={k:source.get(k) for k in ('segment_uid','patient_id','session_id','segment_id','split','transcript_provider','WD_P_mean','WD_soft','WD_consensus')}
                record.update(raw_response=raw,status='ERROR',shot=args.shot,config_hash=config_hash)
                try:
                    score=parse_generated(raw); record.update(status='OK',wd_p_score=score,
                                                              primary_label='WD_P' if score>1 else 'NO_WD_P',
                                                              WD_probability=float(score>1))
                except Exception as exc: record['error']=f'{type(exc).__name__}: {exc}'
                predictions[source['segment_uid']]=record; log.write(json.dumps(record,ensure_ascii=False)+'\n')
            log.flush(); print(f'{sum(r.get("status")=="OK" for r in predictions.values())}/{len(selected)} complete; batch {elapsed:.1f}s',flush=True)
    ordered=[predictions[r['segment_uid']] for r in selected if r['segment_uid'] in predictions]
    pd.DataFrame(ordered).to_csv(output/'predictions.csv',index=False,encoding='utf-8-sig')
    successful=[r for r in ordered if r['status']=='OK']; consensus=[r for r in successful if finite(r.get('WD_consensus'))]
    metrics=binary_metrics([int(float(r['WD_consensus'])) for r in consensus],[r['WD_probability'] for r in consensus],.5)
    summary={'shot':args.shot,'selected':len(selected),'successful':len(successful),'errors':len(ordered)-len(successful),
             'consensus_evaluation_metrics':metrics,'demonstrations':[r['segment_uid'] for r in demos]}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8'); print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--shot',choices=['zero','few'],required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['train','val','test'],default='test'); p.add_argument('--consensus-only',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--examples-per-class',type=int,default=3); p.add_argument('--model',default='Qwen/Qwen3-8B'); p.add_argument('--revision',default='main')
    p.add_argument('--dtype',choices=['bfloat16','float16'],default='bfloat16'); p.add_argument('--attention',choices=['sdpa','eager','flash_attention_2'],default='sdpa')
    p.add_argument('--batch-size',type=int,default=4); p.add_argument('--max-input-tokens',type=int,default=8192); p.add_argument('--max-new-tokens',type=int,default=32)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--prepare-only',action='store_true'); main(p.parse_args())
