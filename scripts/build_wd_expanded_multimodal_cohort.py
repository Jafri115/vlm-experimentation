#!/usr/bin/env python
"""Build a strict expanded WD_P cohort shared by transcript and visual models.

Audit mode needs only labels/transcripts. Finalize mode additionally resolves raw
videos and patient sides, builds every frame cache entry, removes failures from
both modalities, and freezes identical patient-disjoint folds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from build_llm_wd_aligned_dataset import load_transcripts,segment_uid
from build_wd_multimodal_master_cohorts import (
    clean_json,describe,patient_folds,validation_patients,write_json,write_table,
)
from wd_presentation_common import exclusive_lock

ROOT=Path(__file__).resolve().parents[1]
TARGET_COLUMNS={
    'coder_1','coder_2','WD_P_rater1','WD_P_rater2','WD_P_mean','WD_P_min','WD_P_max',
    'WD_P_absolute_difference','WD_binary_rater1','WD_binary_rater2','WD_soft',
    'WD_consensus','WD_binary_disagreement','WD_hard_mean','WD_absolute_rater_difference',
}


def normalize_id(value):
    text=str(value).strip()
    return text[:-2] if text.endswith('.0') else text


def parse_time(value):
    if pd.isna(value):return float('nan')
    try:return float(value)
    except ValueError:pass
    parts=str(value).strip().split(':')
    if len(parts)==3:return float(parts[0])*3600+float(parts[1])*60+float(parts[2])
    if len(parts)==2:return float(parts[0])*60+float(parts[1])
    raise ValueError(f'Invalid segment time: {value!r}')


def build_targets(path,threshold):
    raw=pd.read_csv(path,encoding='utf-8-sig',low_memory=False)
    required={'video','patient_id','session_id','coder','segment_id','segment_start','segment_end','WD_P'}
    if required-set(raw):raise ValueError(f'Labels missing {sorted(required-set(raw))}')
    raw=raw.dropna(subset=['video','patient_id','session_id','coder','segment_id','WD_P']).copy()
    raw['patient_id']=raw.patient_id.map(normalize_id);raw['session_id']=raw.session_id.map(normalize_id)
    raw['video']=raw.video.astype(str).str.strip()
    raw['coder']=raw.coder.astype(str).str.strip().replace({'segments Alex':'Alex'})
    raw['segment_id']=pd.to_numeric(raw.segment_id,errors='raise').astype(int)
    raw['WD_P']=pd.to_numeric(raw.WD_P,errors='raise')
    key=['patient_id','session_id','video','segment_id']
    counts=raw.groupby(key,dropna=False).agg(rows=('coder','size'),coders=('coder','nunique')).reset_index()
    keys=counts[(counts.rows==2)&(counts.coders==2)][key]
    two=raw.merge(keys,on=key,validate='many_to_one')
    rows=[]
    for values,g in two.groupby(key,sort=True,dropna=False):
        g=g.sort_values('coder').reset_index(drop=True);a,b=float(g.loc[0,'WD_P']),float(g.loc[1,'WD_P'])
        ba,bb=int(a>=threshold),int(b>=threshold);start=parse_time(g.loc[0,'segment_start']);end=parse_time(g.loc[0,'segment_end'])+1
        if not math.isfinite(start):start=(int(values[3])-1)*60.
        if not math.isfinite(end) or end<=start:end=start+60.
        uid=segment_uid(values[0],values[1],values[3])
        rows.append({'sample_id':uid,'segment_uid':uid,'patient_id':values[0],'session_id':values[1],
            'video':values[2],'segment_id':int(values[3]),'segment_start':g.loc[0,'segment_start'],
            'segment_end':g.loc[0,'segment_end'],'start_sec':start,'end_sec':end,
            'coder_1':g.loc[0,'coder'],'coder_2':g.loc[1,'coder'],'WD_P_rater1':a,'WD_P_rater2':b,
            'WD_P_mean':(a+b)/2,'WD_P_min':min(a,b),'WD_P_max':max(a,b),
            'WD_P_absolute_difference':abs(a-b),'WD_absolute_rater_difference':abs(a-b),
            'WD_binary_rater1':ba,'WD_binary_rater2':bb,'WD_soft':(ba+bb)/2,
            'WD_consensus':float(ba) if ba==bb else np.nan,'WD_binary_disagreement':int(ba!=bb),
            'WD_hard_mean':int((a+b)/2>=threshold),'split':'train'})
    result=pd.DataFrame(rows)
    if result.empty or result.segment_uid.duplicated().any():raise ValueError('Invalid exactly-two-rater targets')
    return result


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def attach_strict_transcripts(targets,transcript_path):
    transcripts=load_transcripts(transcript_path);included=[];excluded=[]
    for row in targets.to_dict('records'):
        uid=segment_uid(row['patient_id'],row['session_id'],row['segment_id'])
        transcript=transcripts.get(uid)
        reason=None
        if transcript is None:reason='NO_TRANSCRIPT_ROW'
        elif not str(transcript.get('transcript_text','') or '').strip():reason='EMPTY_TRANSCRIPT'
        elif not bool(transcript.get('llm_ready')):reason='TRANSCRIPT_NOT_READY'
        if reason:
            excluded.append({'segment_uid':uid,'patient_id':row['patient_id'],'video':row['video'],
                             'reason':reason,'transcript_status':transcript.get('transcript_status','') if transcript else ''})
            continue
        included.append({**row,'segment_uid':uid,
            'transcript_text':str(transcript['transcript_text']).strip(),
            'transcript_text_plain':transcript.get('transcript_text_plain',''),
            'transcript_provider':transcript.get('transcript_provider',''),
            'transcript_status':transcript.get('transcript_status',''),
            'review_flags':transcript.get('review_flags',''),'llm_ready':True})
    result=pd.DataFrame(included)
    if result.empty or result.segment_uid.duplicated().any():raise ValueError('No unique transcript-ready candidates')
    return result,pd.DataFrame(excluded)


def write_long_labels(frame,path):
    rows=[]
    for r in frame.to_dict('records'):
        common={k:r[k] for k in ['video','patient_id','session_id','segment_id','segment_start','segment_end']}
        rows.extend([{**common,'coder':r['coder_1'],'WD_P':r['WD_P_rater1']},
                     {**common,'coder':r['coder_2'],'WD_P':r['WD_P_rater2']}])
    pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig')


def freeze_folds(paired,output,n_folds,inner_folds,seed):
    output.mkdir(parents=True,exist_ok=True)
    assignments=patient_folds(paired,n_folds,seed)
    assignment=[];all_patients=sorted(paired.patient_id.astype(str).unique())
    for patient,fold in sorted(assignments.items()):
        g=paired[paired.patient_id.astype(str)==patient];c=g[g.WD_consensus.notna()]
        assignment.append({'patient_id':patient,'outer_fold':fold,'rows':len(g),'consensus_rows':len(c),
                           'consensus_negative':int((c.WD_consensus==0).sum()),
                           'consensus_positive':int((c.WD_consensus==1).sum())})
    pd.DataFrame(assignment).to_csv(output/'paired_cv_patient_assignments.csv',index=False)
    summaries=[]
    for fold in range(1,n_folds+1):
        test=sorted(p for p,f in assignments.items() if f==fold)
        val=validation_patients(paired,test,inner_folds,seed,fold)
        train=sorted(set(all_patients)-set(test)-set(val))
        d=paired.copy();d['split']='train';patients=d.patient_id.astype(str)
        d.loc[patients.isin(val),'split']='val';d.loc[patients.isin(test),'split']='test'
        if set(train)&set(val) or set(train)&set(test) or set(val)&set(test):raise RuntimeError('Patient leakage')
        folder=output/f'fold_{fold}';folder.mkdir(parents=True,exist_ok=True)
        write_table(d,folder/'master_manifest.csv')
        d.drop(columns=[c for c in TARGET_COLUMNS if c in d]).to_csv(folder/'vlm_manifest.csv',index=False,encoding='utf-8-sig')
        write_table(d[d.WD_consensus.notna()].copy(),folder/'consensus_manifest.csv')
        info={'outer_fold':fold,'train_patients':train,'val_patients':val,'test_patients':test,
              'counts':{s:describe(d[d.split==s]) for s in ['train','val','test']}}
        write_json(folder/'patient_split.json',info);summaries.append(info)
    return summaries


def build(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    targets=build_targets(args.labels_csv,args.positive_threshold)
    candidates,transcript_excluded=attach_strict_transcripts(targets,args.transcripts)
    write_table(candidates,output/'candidate_label_transcript_ready.csv')
    transcript_excluded.to_csv(output/'excluded_transcripts.csv',index=False,encoding='utf-8-sig')
    audit={'stage':'candidate_only','exactly_two_rater':describe(targets.assign(transcript_provider='')),
           'label_transcript_ready':describe(candidates),'exclusion_counts':dict(Counter(transcript_excluded.reason)),
           'sources':{'labels':str(args.labels_csv.resolve()),'labels_sha256':sha(args.labels_csv),
                      'transcripts':str(args.transcripts.resolve()),'transcripts_sha256':sha(args.transcripts)}}
    write_json(output/'candidate_summary.json',audit)
    print(json.dumps(clean_json(audit),indent=2),flush=True)
    if args.audit_only:return
    for path,label in [(args.video_root,'video root'),(args.role_cache,'patient role cache'),(args.yunet_model,'YuNet model')]:
        if not path.exists():raise FileNotFoundError(f'Missing {label}: {path}')
    from build_qwen3vl_wd_full_manifest import build_frame_cache,resolve_manifest
    training,media,missing,unresolved,video_audit,evidence=resolve_manifest(
        candidates,args.video_root,args.role_cache,True)
    video_audit.to_csv(output/'video_role_audit.csv',index=False);evidence.to_csv(output/'patient_role_evidence.csv',index=False)
    missing.to_csv(output/'excluded_missing_video.csv',index=False);unresolved.to_csv(output/'excluded_patient_side.csv',index=False)
    write_table(training,output/'precache_visual_ready.csv')
    cache_summary=build_frame_cache(training,args.frame_cache,args.yunet_model,args.num_frames,
                                    args.frame_width,'all',args.positive_threshold,output)
    try:
        failures=pd.read_csv(output/'frame_cache_failures.csv')
    except pd.errors.EmptyDataError:
        failures=pd.DataFrame()
    failed=set(failures.sample_id.astype(str)) if 'sample_id' in failures else set()
    paired=training[~training.sample_id.astype(str).isin(failed)].copy().reset_index(drop=True)
    paired['visual_ready']=True;paired['paired_ready']=True;paired['frame_cache_ready']=True
    if paired.empty:raise RuntimeError('No rows survived visual cache construction')
    write_table(paired,output/'paired_master_soft.csv')
    write_table(paired[paired.WD_consensus.notna()].copy(),output/'paired_master_consensus.csv')
    write_long_labels(paired,output/'frozen_rater_labels_long.csv')
    folds=freeze_folds(paired,output,args.cv_folds,args.inner_folds,args.seed)
    summary={'purpose':'expanded identical physical-segment cohort for VLM/LLM WD_P',
             'positive_rule':f'each rater WD_P >= {args.positive_threshold:g}',
             'candidate_label_transcript_ready':describe(candidates),'precache_visual_ready':describe(training),
             'paired_master':describe(paired),'videos':int(paired.video.nunique()),
             'frame_cache':cache_summary,
             'exclusions':{'transcript':dict(Counter(transcript_excluded.reason)),
                 'missing_video_rows':len(missing),'unresolved_patient_side_rows':len(unresolved),'frame_cache_failures':len(failed)},
             'cv':{'outer_folds':args.cv_folds,'inner_folds':args.inner_folds,'seed':args.seed},'folds':folds,
             'sources':audit['sources']|{'video_root':str(args.video_root.resolve()),
                 'role_cache':str(args.role_cache.resolve()),'role_cache_sha256':sha(args.role_cache)}}
    write_json(output/'master_cohort_summary.json',summary)
    print('\nFINAL SHARED COHORT\n'+json.dumps(clean_json(summary['paired_master']|{'videos':summary['videos']}),indent=2),flush=True)


def main(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(output/'build.lock'):
        build(args)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels-csv',type=Path,default=ROOT/'data/completed_segments_merged.csv')
    p.add_argument('--transcripts',type=Path,default=ROOT/'data/amberscript_llm/llm_segments_all_role_repaired.jsonl')
    p.add_argument('--video-root',type=Path,default=Path(r'C:\Data\Sequence_model\Memopsy_videos\CONVERTED'))
    p.add_argument('--role-cache',type=Path,default=ROOT/'output/qwen3vl_visual_experiment_v5/patient_role_cache.json')
    p.add_argument('--yunet-model',type=Path,default=ROOT/'models/face_detection_yunet/face_detection_yunet_2026may.onnx')
    p.add_argument('--output',type=Path,default=ROOT/'output/wd_multimodal_master_expanded')
    p.add_argument('--frame-cache',type=Path,default=ROOT/'output/wd_multimodal_master_expanded/frame_cache_16')
    p.add_argument('--positive-threshold',type=float,default=2.0)
    p.add_argument('--num-frames',type=int,default=16);p.add_argument('--frame-width',type=int,default=224)
    p.add_argument('--cv-folds',type=int,default=5);p.add_argument('--inner-folds',type=int,default=4)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--audit-only',action='store_true')
    main(p.parse_args())
