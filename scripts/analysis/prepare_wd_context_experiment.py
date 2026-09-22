"""Build target-only/previous-context inputs on the frozen cohort; fit paired TF-IDF."""
import argparse
from pathlib import Path

import pandas as pd

from llm.wd_context_inputs import build_context
from wd_presentation_common import load_folds,sha,write_json
from run_wd_tfidf_baselines import fit_fold


def prepare(master_root,output,max_gap=1.0,fit_baselines=True):
    master,folds=load_folds(master_root)
    audit=[]
    for variant in ['target_only','previous_context']:
        frames=[]
        for fold,frame,path in folds:
            context=build_context(frame,variant=='previous_context',max_gap)
            directory=output/'datasets'/variant/f'fold_{fold}'
            directory.mkdir(parents=True,exist_ok=True)
            context.to_csv(directory/'master_manifest.csv',index=False)
            context.to_json(directory/'master_manifest.jsonl',orient='records',lines=True,force_ascii=False)
            for split,g in context.groupby('split'):
                audit.append({'variant':variant,'fold':fold,'split':split,'N':len(g),
                    'context_available':int(g.context_available.sum()),'context_used':int(g.context_used.sum()),
                    'no_patient_span':sum(not spans for spans in g.target_patient_spans)})
            write_json(directory/'provenance.json',{'source':str(path.resolve()),'source_sha256':sha(path),
                'max_gap_seconds':max_gap,'target_labels_unchanged':True,'context_source':'frozen paired cohort only'})
            if fit_baselines:
                print(f'TF-IDF {variant}, fold {fold}',flush=True)
                predictions,summary,validation=fit_fold(context,'consensus',return_validation=True)
                predictions['outer_fold']=fold;validation['outer_fold']=fold
                out=output/'tfidf'/variant/f'fold_{fold}';out.mkdir(parents=True,exist_ok=True)
                predictions.to_csv(out/'test_predictions.csv',index=False)
                validation.to_csv(out/'val_predictions.csv',index=False)
                summary['manifest_sha256']=sha(directory/'master_manifest.csv')
                write_json(out/'summary.json',summary);frames.append(predictions)
        if frames:
            combined=pd.concat(frames,ignore_index=True)
            expected=set(master.loc[master.b1==master.b2,'sample_id'])
            if combined.sample_id.duplicated().any() or set(combined.sample_id)!=expected:
                raise ValueError('Context TF-IDF OOF coverage differs from fixed consensus cohort')
            combined.to_csv(output/'tfidf'/variant/'oof_predictions.csv',index=False)
    pd.DataFrame(audit).to_csv(output/'context_coverage.csv',index=False)
    return master


def export_review(master_root,output,llm_source,fold=1,n=30):
    """Only this fold's validation examples, never its test predictions."""
    if not 1<=fold<=5:raise ValueError('Review fold must be 1..5')
    frame=pd.read_csv(output/'datasets'/'previous_context'/f'fold_{fold}'/'master_manifest.csv')
    frame=frame[(frame['split']=='val')&(frame.WD_P_rater1.ge(2)==frame.WD_P_rater2.ge(2))].copy()
    frame['label']=frame.WD_P_rater1.ge(2).astype(int)
    tf=pd.read_csv(output/'tfidf'/'target_only'/f'fold_{fold}'/'val_predictions.csv')
    frame=frame.merge(tf[['segment_uid','WD_probability']].rename(columns={'WD_probability':'tfidf_probability'}),on='segment_uid',validate='one_to_one')
    source=llm_source/f'fold_{fold}'/'val_predictions.csv'
    if not source.exists():
        raise ValueError(f'Missing validation predictions for review: {source}')
    llm=pd.read_csv(source)
    if 'split' in llm and not llm['split'].eq('val').all():raise ValueError('Review input contains non-validation predictions')
    if llm.segment_uid.duplicated().any():raise ValueError('Duplicate validation predictions')
    if not set(frame.segment_uid).issubset(set(llm.segment_uid)):raise ValueError('Validation coverage mismatch')
    frame=frame.merge(llm[['segment_uid','WD_probability']].rename(columns={'WD_probability':'llm_probability'}),on='segment_uid',validate='one_to_one')
    for name in ['tfidf','llm']:
        p=pd.to_numeric(frame[name+'_probability'],errors='raise')
        if not p.between(0,1).all():raise ValueError('Invalid validation probability')
        frame[name+'_error']=(p.ge(.5)!=frame.label)
    errors=frame[frame.llm_error|frame.tfidf_error].copy()
    errors['error_type']=['LLM_wrong_TFIDF_correct' if a and not b else 'both_wrong' if a else 'TFIDF_wrong_LLM_correct'
        for a,b in zip(errors.llm_error,errors.tfidf_error)]
    errors['llm_error_direction']=['false_positive' if y==0 and e else 'false_negative' if e else 'correct'
        for y,e in zip(errors.label,errors.llm_error)]
    # Balanced deterministic sampling over error categories; no test-based ranking.
    buckets=[g.sample(frac=1,random_state=42) for _,g in errors.groupby(['error_type','llm_error_direction'])]
    selected=[];i=0
    while len(selected)<n and any(i<len(g) for g in buckets):
        for g in buckets:
            if i<len(g) and len(selected)<n:selected.append(g.iloc[i])
        i+=1
    cols=['sample_id','segment_uid','patient_id','session_id','WD_P_rater1','WD_P_rater2',
          'label','tfidf_probability','llm_probability','error_type','llm_error_direction',
          'context_available','context_source_uid','transcript_text']
    review=pd.DataFrame(selected,columns=errors.columns).reindex(columns=cols)
    review['review_reason']='';review['notes']=''
    directory=output/'validation_review';directory.mkdir(parents=True,exist_ok=True)
    review.to_csv(directory/f'fold_{fold}_review.csv',index=False,encoding='utf-8-sig')
    write_json(directory/'scope.json',{'fold':fold,'split':'val','selected_N':len(review),
        'available_error_N':len(errors),'source':str(source),'source_sha256':sha(source),
        'caution':'Validation patients for this fold may be test patients in another fold. Treat adaptations after review as exploratory.',
        'suggested_reasons':['speaker_or_transcript_error','missing_context','ordinary_brevity_false_positive',
                             'indirect_withdrawal_missed','ambiguous_rating','other']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--master-root',type=Path,default=Path('output/wd_multimodal_master_repaired'))
    p.add_argument('--output',type=Path,default=Path('output/wd_context_experiment'))
    p.add_argument('--max-gap',type=float,default=1.)
    p.add_argument('--manifests-only',action='store_true')
    p.add_argument('--review-source',type=Path,default=Path('output/llm_wd_consensus_repaired_cv'))
    args=p.parse_args()
    prepare(args.master_root,args.output,args.max_gap,not args.manifests_only)
    if not args.manifests_only:export_review(args.master_root,args.output,args.review_source)
