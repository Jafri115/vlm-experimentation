"""Paired comparison of VLM and transcript-LLM WD_P predictions."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from finetune_qwen3_8b_wd_text import binary_metrics


def norm_session(value):
    match=re.fullmatch(r'[sS]?(\d+)',str(value).strip())
    if not match: raise ValueError(f'Invalid session: {value!r}')
    return str(int(match.group(1)))


def add_uid(frame):
    frame=frame.copy()
    if 'segment_uid' in frame: return frame
    if 'sample_id' in frame:
        frame['segment_uid']=frame['sample_id'].astype(str); return frame
    required={'patient_id','session_id','segment_id'}
    if not required.issubset(frame.columns): raise ValueError('Need segment_uid, sample_id, or patient/session/segment columns')
    frame['segment_uid']=[f'{str(p).removesuffix(".0")}_S{norm_session(s)}_seg{int(float(n)):03d}'
                          for p,s,n in zip(frame.patient_id,frame.session_id,frame.segment_id)]
    return frame


def exact_mcnemar(discordant_vlm_correct, discordant_llm_correct):
    from scipy.stats import binomtest
    n=discordant_vlm_correct+discordant_llm_correct
    if not n: return 1.0
    return float(binomtest(discordant_llm_correct, n=n, p=0.5, alternative='two-sided').pvalue)


def main(args):
    vlm=add_uid(pd.read_csv(args.vlm_predictions,encoding='utf-8-sig'))
    llm=add_uid(pd.read_csv(args.llm_predictions,encoding='utf-8-sig'))
    for name,frame,column in [('VLM',vlm,args.vlm_probability_column),('LLM',llm,args.llm_probability_column)]:
        if column not in frame: raise ValueError(f'{name} predictions lack {column!r}')
        if frame.segment_uid.duplicated().any(): raise ValueError(f'{name} has duplicate segment IDs')
    keep=['segment_uid',args.vlm_probability_column]
    vlm=vlm[keep].rename(columns={args.vlm_probability_column:'vlm_probability'})
    llm_truth='WD_consensus' if 'WD_consensus' in llm else args.truth_column
    required=['segment_uid',args.llm_probability_column,llm_truth,'patient_id']
    missing=[x for x in required if x not in llm]
    if missing: raise ValueError(f'LLM predictions lack {missing}')
    llm=llm[required].rename(columns={args.llm_probability_column:'llm_probability',llm_truth:'truth'})
    paired=llm.merge(vlm,on='segment_uid',how='inner',validate='one_to_one')
    paired=paired[pd.to_numeric(paired.truth,errors='coerce').notna()].copy()
    paired['truth']=paired.truth.astype(float).astype(int)
    paired['vlm_pred']=(paired.vlm_probability.astype(float)>=args.threshold).astype(int)
    paired['llm_pred']=(paired.llm_probability.astype(float)>=args.threshold).astype(int)
    paired['vlm_correct']=(paired.vlm_pred==paired.truth).astype(int)
    paired['llm_correct']=(paired.llm_pred==paired.truth).astype(int)
    if paired.empty: raise ValueError('No shared consensus rows')
    y=paired.truth.to_numpy(); vp=paired.vlm_probability.to_numpy(float); lp=paired.llm_probability.to_numpy(float)
    vm=binary_metrics(y,vp,args.threshold); lm=binary_metrics(y,lp,args.threshold)
    v_only=int(((paired.vlm_correct==1)&(paired.llm_correct==0)).sum())
    l_only=int(((paired.vlm_correct==0)&(paired.llm_correct==1)).sum())
    rng=np.random.default_rng(args.seed); patients=paired.patient_id.astype(str).unique(); differences=[]
    for _ in range(args.bootstrap_repetitions):
        sampled=rng.choice(patients,size=len(patients),replace=True)
        parts=[paired[paired.patient_id.astype(str)==patient] for patient in sampled]
        boot=pd.concat(parts,ignore_index=True); by=boot.truth.to_numpy();
        differences.append(binary_metrics(by,boot.llm_probability.to_numpy(float),args.threshold)['balanced_accuracy']-
                           binary_metrics(by,boot.vlm_probability.to_numpy(float),args.threshold)['balanced_accuracy'])
    low,high=np.quantile(differences,[.025,.975])
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True); paired.to_csv(output/'paired_predictions.csv',index=False,encoding='utf-8-sig')
    summary={'shared_consensus_rows':len(paired),'shared_patients':len(patients),'threshold':args.threshold,
             'vlm_metrics':vm,'llm_metrics':lm,
             'llm_minus_vlm':{k:lm[k]-vm[k] for k in ('accuracy','balanced_accuracy','precision','recall','specificity','f1','auprc','auroc') if lm[k] is not None and vm[k] is not None},
             'mcnemar':{'vlm_only_correct':v_only,'llm_only_correct':l_only,'exact_two_sided_p':exact_mcnemar(v_only,l_only)},
             'patient_cluster_bootstrap_balanced_accuracy_difference_95ci':[float(low),float(high)],
             'note':'Inference is paired on identical segments. The bootstrap resamples patients, preserving within-patient dependence.'}
    (output/'comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8'); print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--vlm-predictions',type=Path,required=True); p.add_argument('--llm-predictions',type=Path,required=True)
    p.add_argument('--vlm-probability-column',default='WD_probability'); p.add_argument('--llm-probability-column',default='WD_probability')
    p.add_argument('--truth-column',default='WD_consensus'); p.add_argument('--threshold',type=float,default=.5); p.add_argument('--bootstrap-repetitions',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--output',type=Path,required=True); main(p.parse_args())
