"""Qwen3-8B zero-shot text classification for a single 32 GB NVIDIA GPU.

Uses the official, unquantized Qwen/Qwen3-8B checkpoint in BF16, with thinking
disabled. Nothing runs until this script is invoked. See docs/qwen3_8b_32gb.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from run_qwen3_8b_text_zeroshot import (
    PROMPT, ROOT, digest, export_results, parse_label, select_rows, stable_hash, write_csv,
)


def parse_generated(text):
    text = text.strip()
    # Some checkpoints wrap otherwise valid JSON in one Markdown fence.
    if text.startswith('```json\n') and text.endswith('```'):
        text = text[8:-3].strip()
    elif text.startswith('```\n') and text.endswith('```'):
        text = text[4:-3].strip()
    return parse_label(text, 'stop')


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.dataset.read_text(encoding='utf-8').splitlines()]
    selected, excluded = select_rows(rows, args.scope)
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError('No eligible segments')
    config = {'model_id': args.model, 'revision': args.revision, 'dtype': args.dtype,
              'quantization': None, 'enable_thinking': False, 'zero_shot': True,
              'dataset_sha256': digest(args.dataset), 'scope': args.scope,
              'selected_ids_sha256': stable_hash([r['segment_uid'] for r in selected]),
              'selected_segments': len(selected), 'system_prompt': PROMPT,
              'batch_size': args.batch_size, 'max_input_tokens': args.max_input_tokens,
              'max_new_tokens': args.max_new_tokens, 'temperature': args.temperature,
              'top_p': 0.8, 'top_k': 20, 'seed': args.seed, 'attention': 'sdpa'}
    fingerprint = stable_hash(config)
    config_path = output/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8'))['config_hash'] != fingerprint:
        raise ValueError('Inputs or settings changed. Choose a different --output directory.')
    config_path.write_text(json.dumps({**config, 'config_hash': fingerprint}, indent=2)+'\n', encoding='utf-8')
    (output/'system_prompt.txt').write_text(PROMPT+'\n', encoding='utf-8')
    write_csv(output/'selected_segments.csv', [{k: row[k] for k in ('segment_uid', 'segment_idx', 'patient_id', 'session_id',
              'start_sec', 'end_sec', 'transcript_provider', 'llm_ready', 'transcript_status', 'review_flags')} for row in selected])
    write_csv(output/'excluded_segments.csv', excluded)
    if args.prepare_only:
        print(json.dumps({'selected_segments': len(selected), 'scope': args.scope, 'output': str(output), 'model_loaded': False}, indent=2))
        return

    log_path = output/'predictions.jsonl'
    results = {}
    if log_path.exists():
        for line in log_path.read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            if record['config_hash'] != fingerprint:
                raise ValueError('Prediction checkpoint configuration mismatch')
            results[record['segment_uid']] = record
    pending = [r for r in selected if results.get(r['segment_uid'], {}).get('status') != 'OK']
    if not pending:
        print(json.dumps(export_results(output, results, selected, fingerprint), indent=2))
        return

    # Heavy dependencies and model loading occur only for an explicit inference run.
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required. Install a CUDA-enabled PyTorch build.')
    if args.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU does not support BF16. Use --dtype float16.')
    set_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=dtype,
        device_map={'': 0}, attn_implementation='sdpa',
    ).eval()
    runtime = {'torch': torch.__version__, 'transformers': transformers.__version__,
               'gpu': torch.cuda.get_device_name(0), 'resolved_model_revision': getattr(model.config, '_commit_hash', None),
               'dtype': str(dtype), 'cuda': torch.version.cuda}
    runtime_path = output/'runtime.json'
    if runtime_path.exists():
        old_runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
        if old_runtime.get('resolved_model_revision') != runtime['resolved_model_revision']:
            raise ValueError('Resolved model revision changed; use the original --revision or a new output.')
    runtime_path.write_text(json.dumps(runtime, indent=2)+'\n', encoding='utf-8')
    generation = {'max_new_tokens': args.max_new_tokens, 'do_sample': args.temperature > 0,
                  'pad_token_id': tokenizer.pad_token_id, 'eos_token_id': tokenizer.eos_token_id}
    if args.temperature > 0:
        generation.update(temperature=args.temperature, top_p=0.8, top_k=20)
    print(f"Loaded {args.model} on {runtime['gpu']}; {len(pending)} segments pending", flush=True)

    with log_path.open('a', encoding='utf-8') as log:
        for position in range(0, len(pending), args.batch_size):
            batch = pending[position:position+args.batch_size]
            prompts = [tokenizer.apply_chat_template(
                [{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': row['transcript_text']}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ) for row in batch]
            # Never silently truncate a transcript to fit the context.
            lengths = [len(tokenizer(p, add_special_tokens=False)['input_ids']) for p in prompts]
            if any(n > args.max_input_tokens for n in lengths):
                offending = [r['segment_uid'] for r, n in zip(batch, lengths) if n > args.max_input_tokens]
                raise ValueError(f'Input exceeds --max-input-tokens: {offending}; increase the limit or review those rows.')
            if max(lengths)+args.max_new_tokens > model.config.max_position_embeddings:
                raise ValueError('Input plus generation exceeds model context capacity')
            inputs = tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda:0')
            torch.cuda.synchronize()
            start = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation)
            torch.cuda.synchronize()
            elapsed = time.monotonic()-start
            width = inputs['input_ids'].shape[1]
            output_ids = generated[:, width:]
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            for row, raw, ids, input_tokens in zip(batch, outputs, output_ids.tolist(), lengths):
                record = {k: row[k] for k in ('segment_uid', 'segment_idx', 'patient_id', 'session_id', 'start_sec', 'end_sec',
                                              'transcript_provider', 'llm_ready', 'transcript_status', 'review_flags')}
                record.update(config_hash=fingerprint, model_id=args.model, primary_label=None, status='ERROR', raw_response=raw,
                              input_tokens=input_tokens, batch_inference_sec=round(elapsed, 3), batch_size_actual=len(batch))
                try:
                    if not any(token in eos_ids for token in ids):
                        raise ValueError('Generation hit the output limit without EOS')
                    label = parse_generated(raw)
                    record.update(primary_label=label, status='OK', pred_wd_p=int(label in ('WD_P', 'MIXED_P')),
                                  pred_cf_p=int(label in ('CF_P', 'MIXED_P')), pred_rupture=int(label != 'NO_RUPTURE'))
                except (ValueError, TypeError) as exc:
                    record['error'] = str(exc)
                log.write(json.dumps(record, ensure_ascii=False)+'\n')
                results[row['segment_uid']] = record
            log.flush()
            summary = export_results(output, results, selected, fingerprint)
            print(f"{summary['successful_predictions']}/{len(selected)} successful; {summary['failed_predictions']} errors; batch {elapsed:.1f}s", flush=True)
    print(json.dumps(export_results(output, results, selected, fingerprint), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/amberscript_llm/llm_segments_all.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT/'output/qwen3_8b_bf16_zeroshot/inventory_ready')
    parser.add_argument('--scope', choices=['inventory-ready', 'all-ready', 'inventory-nonempty'], default='inventory-ready')
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--revision', default='main', help='Use a commit SHA to pin the checkpoint.')
    parser.add_argument('--dtype', choices=['bfloat16', 'float16'], default='bfloat16')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-input-tokens', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=0.0, help='0 gives deterministic greedy decoding; >0 enables sampling.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_input_tokens < 1 or args.max_new_tokens < 1 or args.temperature < 0 or (args.limit is not None and args.limit < 1):
        parser.error('Batch size, token limits and limit must be positive; temperature must be nonnegative.')
    main(args)
