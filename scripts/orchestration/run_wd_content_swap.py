"""Within-patient input-swap diagnostic using independent saved OOF predictions.

No model is loaded. For independent deterministic inference f(x), exchanging inputs
is equivalent to exchanging their saved f(x). This is not a new inference run.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.wd_presentation_common import load_folds, load_oof, metrics, paired_interval, sha, write_json


def donors(d, rng):
    """Random cycles: no self-donors; every donor used once within patient/fold."""
    result=np.full(len(d),-1,dtype=int)
    for _,positions in d.groupby(['patient_id','fold']).indices.items():
        if len(positions)<2:
            continue
        order=rng.permutation(positions)
        result[order]=np.roll(order,1)
    return result


def main(args):
    if args.shuffles<1 or args.bootstrap<1:
        raise ValueError('Shuffles and bootstrap must be positive')
    master,_=load_folds(args.master_root)
    sources={'LLM':args.llm_predictions}
    if args.vlm_predictions is not None:
        sources['VLM']=args.vlm_predictions
    loaded={name:load_oof(path,master,'WD_probability') for name,path in sources.items()}
    common=set.intersection(*(set(d.sample_id) for d in loaded.values()))
    cohort=master[master.sample_id.isin(common)].sort_values('sample_id').reset_index(drop=True)
    eligible=cohort.groupby(['patient_id','fold']).sample_id.transform('size')>=2
    exclusions=cohort.loc[~eligible,['sample_id','patient_id','fold']]
    cohort=cohort[eligible].reset_index(drop=True)
    if cohort.empty:
        raise ValueError('No eligible within-patient swaps')
    mappings=[donors(cohort,np.random.default_rng(args.seed+i)) for i in range(args.shuffles)]
    if any((x<0).any() or (x==np.arange(len(cohort))).any() for x in mappings):
        raise AssertionError('Invalid donor mapping')
    args.output.mkdir(parents=True,exist_ok=True)
    exclusions.to_csv(args.output/'excluded_singletons.csv',index=False)
    records,summary,gains=[],[],[]
    for name,frame in loaded.items():
        pred=cohort[['sample_id']].merge(frame[['sample_id','WD_probability']],on='sample_id',validate='one_to_one').WD_probability.to_numpy()
        truth=cohort.soft_target.to_numpy()
        summary.append({'model':name,'condition':'correct','replicate':0,**metrics(cohort,pred,'soft')})
        swapped_errors=[]
        for replicate,idx in enumerate(mappings,1):
            swapped=pred[idx]
            summary.append({'model':name,'condition':'within_patient_swap','replicate':replicate,**metrics(cohort,swapped,'soft')})
            swapped_errors.append((truth-swapped)**2)
            records.append(pd.DataFrame({'model':name,'replicate':replicate,
                'sample_id':cohort.sample_id,'patient_id':cohort.patient_id,'outer_fold':cohort.fold,
                'donor_sample_id':cohort.sample_id.to_numpy()[idx],
                'donor_session_id':cohort.session_id.to_numpy()[idx], 'target_session_id':cohort.session_id,
                'WD_soft':truth,'WD_consensus':cohort.b1.where(cohort.b1==cohort.b2),
                'correct_probability':pred,'swapped_probability':swapped}))
        improvement=np.mean(swapped_errors,axis=0)-(truth-pred)**2
        gains.append({'model':name,'definition':'mean swapped Brier minus correct Brier; positive favors correct content',
                      **paired_interval(cohort,improvement,np.random.default_rng(args.seed),args.bootstrap)})
    pd.concat(records,ignore_index=True).to_csv(args.output/'swap_predictions.csv',index=False)
    pd.DataFrame(summary).to_csv(args.output/'metrics.csv',index=False)
    pd.DataFrame(gains).to_csv(args.output/'content_gain.csv',index=False)
    write_json(args.output/'provenance.json',{'method':'saved-prediction random-cycle exchange, not fresh inference',
        'seed':args.seed,'shuffles':args.shuffles,'eligible_N':len(cohort),
        'common_before_singleton_exclusion':len(common),'source_rows':{k:len(v) for k,v in loaded.items()},
        'files':{k:{'path':str(v.resolve()),'sha256':sha(v)} for k,v in sources.items()},
        'assumption':'segment-independent evaluation, same checkpoint within fold; fixed saved numerical outputs',
        'CI_scope':'patients resampled; conditions on these shuffle mappings and fitted models'})
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--master-root',type=Path,default=Path('output/wd_multimodal_master_repaired'))
    p.add_argument('--llm-predictions',type=Path,default=Path('output/llm_wd_soft_repaired_cv/oof_predictions.csv'))
    p.add_argument('--vlm-predictions',type=Path,default=None)
    p.add_argument('--output',type=Path,default=Path('output/wd_presentation_3day/content_swap'))
    p.add_argument('--shuffles',type=int,default=3)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=2026)
    main(p.parse_args())
