"""Causal transcript context construction and token pooling masks; no GPU imports."""
import re

import numpy as np
import pandas as pd

CONTEXT_INSTRUCTION = ('\nPrevious-segment text, when supplied, is context only. '
                       'Predict patient withdrawal in the TARGET SEGMENT only. '
                       'Do not transfer a withdrawal judgment from the previous segment to the target.')
TARGET_HEADER = 'TARGET SEGMENT - RATE THIS SEGMENT ONLY\n'
PREVIOUS_HEADER = 'PREVIOUS SEGMENT - CONTEXT ONLY\n'
ROLE = re.compile(r'^\s*(?:\[[^\]\n]+\]\s*)?([TP])\s*:\s*')


def patient_spans(text):
    spans=[];position=0;role=None
    for line in text.splitlines(keepends=True):
        match=ROLE.match(line)
        if match:role=match.group(1)
        if role=='P':
            start=position+(match.end() if match else 0)
            end=position+len(line.rstrip())
            if end>start:spans.append((start,end))
        position+=len(line)
    return spans


def build_context(frame, include_previous, max_gap=1.0):
    """Never bridge absent minutes, overlaps, sessions, patients, or splits."""
    d=frame.copy()
    if d.empty or not d.index.is_unique or not np.isfinite(max_gap) or max_gap<0:
        raise ValueError('Context needs nonempty unique rows and a finite nonnegative gap')
    if d[['patient_id','session_id','split']].isna().any().any():
        raise ValueError('Context identifiers and splits must be present')
    for col in ['start_sec','end_sec']:
        d[col]=pd.to_numeric(d[col],errors='raise')
    if not np.isfinite(d[['start_sec','end_sec']].to_numpy()).all() or (d.end_sec<=d.start_sec).any():
        raise ValueError('Context needs finite, increasing segment timings')
    if d.duplicated(['patient_id','session_id','start_sec']).any():
        raise ValueError('Ambiguous segment ordering within session')
    if d.transcript_text.isna().any() or not d.transcript_text.str.strip().ne('').all():
        raise ValueError('Missing target transcript')
    result={}
    for _,group in d.groupby(['patient_id','session_id'],sort=False):
        group=group.sort_values('start_sec')
        if group['split'].nunique()!=1:
            raise ValueError('A session crosses data splits')
        previous=None
        for idx,row in group.iterrows():
            reason='first_available_segment';available=False
            if previous is not None:
                gap=float(row.start_sec-previous.end_sec)
                available=abs(gap)<=1e-6 or (0<=gap<=max_gap)
                reason='contiguous' if available else 'gap_or_overlap'
            target=str(row.transcript_text)
            prefix=(PREVIOUS_HEADER+str(previous.transcript_text)+'\n\n') if include_previous and available else ''
            prefix+=TARGET_HEADER
            body=prefix+target
            result[idx]={'transcript_text':body,'target_text':target,
                'target_start_char':len(prefix),'target_end_char':len(body),
                'target_patient_spans':[[len(prefix)+a,len(prefix)+b] for a,b in patient_spans(target)],
                'context_available':bool(available),'context_used':bool(include_previous and available),
                'context_reason':reason,'context_source_uid':str(previous.segment_uid) if available else None}
            previous=row
    for key in next(iter(result.values())):
        d[key]=[result[idx][key] for idx in d.index]
    return d


def encode_row(tokenizer,row,system_prompt,max_length,pooling='mean_all'):
    body=row['transcript_text']
    text=tokenizer.apply_chat_template([{'role':'system','content':system_prompt},
        {'role':'user','content':body}],tokenize=False,add_generation_prompt=False,enable_thinking=False)
    if text.count(body)!=1:
        raise ValueError('Cannot locate unique input body in chat template')
    encoded=tokenizer(text,add_special_tokens=False,truncation=False,return_offsets_mapping=True)
    if len(encoded['input_ids'])>max_length:
        raise ValueError(f"{row['segment_uid']}: {len(encoded['input_ids'])} tokens exceeds {max_length}; "
                         'no text was silently truncated. Increase max length equally for both variants.')
    base=text.index(body)
    target=(base+int(row['target_start_char']),base+int(row['target_end_char']))
    spans=[(base+int(a),base+int(b)) for a,b in row.get('target_patient_spans',[])]
    fallback=pooling=='target_patient' and not spans
    selected=[target] if fallback else spans
    if pooling=='mean_all':mask=list(encoded['attention_mask'])
    else:
        mask=[int(end>start and any(start<b and end>a for a,b in selected))
              for start,end in encoded['offset_mapping']]
        if not any(mask):raise ValueError(f"{row['segment_uid']}: pooling selects no tokens")
    return {'input_ids':encoded['input_ids'],'attention_mask':encoded['attention_mask'],
            'pool_mask':mask,'pooling_fallback':fallback,'token_count':len(mask)}
