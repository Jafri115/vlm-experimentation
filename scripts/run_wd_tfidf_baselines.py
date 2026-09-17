"""TF-IDF consensus logistic and severity ridge baselines on frozen patient folds."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from wd_presentation_common import load_folds, metrics, sha, write_json


def fit_fold(frame, task):
    d = frame.copy()
    if task == 'consensus':
        d = d[d.b1 == d.b2].copy()
    splits = {s:d[d['split'].str.lower()==s].copy() for s in ['train','val','test']}
    if any(x.empty for x in splits.values()):
        raise ValueError('Empty task-specific train/validation/test split')
    for s,x in splits.items():
        if x.transcript_text.isna().any() or not x.transcript_text.astype(str).str.strip().ne('').all():
            raise ValueError(f'Empty transcript in {s}; do not silently change the cohort')
    # Small grid specified in advance; validation only chooses regularization.
    grid = [.1, 1., 10.] if task == 'consensus' else [1., 10., 100.]
    target = 'b1' if task == 'consensus' else 'target'
    best, best_score, candidates = None, np.inf, []
    for strength in grid:
        estimator = (LogisticRegression(C=strength, solver='liblinear', class_weight=None,
                                       random_state=42, max_iter=2000)
                     if task == 'consensus' else Ridge(alpha=strength, solver='lsqr'))
        model = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2,
                          max_features=30000, sublinear_tf=True, strip_accents=None)), ('model', estimator)])
        train,val = splits['train'],splits['val']
        model.fit(train.transcript_text, train[target])
        pred = model.predict_proba(val.transcript_text)[:,1] if task=='consensus' else model.predict(val.transcript_text).clip(1,5)
        score = float(np.mean((pred-val[target].to_numpy())**2)) if task=='consensus' else float(np.mean(abs(pred-val[target].to_numpy())))
        candidates.append({'strength':strength,'validation_loss':score})
        if score < best_score:
            best, best_score = model, score
    test=splits['test']
    pred=best.predict_proba(test.transcript_text)[:,1] if task=='consensus' else best.predict(test.transcript_text).clip(1,5)
    records=test[['sample_id','segment_uid','patient_id','session_id','WD_P_rater1','WD_P_rater2']].copy()
    records['WD_P_mean']=test.target;records['WD_soft']=test.soft_target
    records['WD_consensus']=test.b1.where(test.b1==test.b2)
    records['WD_probability' if task=='consensus' else 'WD_prediction']=pred
    records['split']='test'
    return records, dict(task=task, train_N=len(train),val_N=len(val),test_N=len(test),
                        selection='val_Brier' if task=='consensus' else 'val_MAE',
                        candidates=candidates, selected_strength=best.named_steps['model'].get_params()['C' if task=='consensus' else 'alpha'],
                        vocabulary_size=len(best.named_steps['tfidf'].vocabulary_),
                        test_metrics=metrics(test,pred,task))


def main(args):
    master, folds=load_folds(args.master_root)
    args.output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for task in ['consensus','regression']:
        frames=[]
        for fold,frame,path in folds:
            print(f'TF-IDF {task} fold {fold}',flush=True)
            predictions,summary=fit_fold(frame,task)
            predictions['outer_fold']=fold;frames.append(predictions)
            directory=args.output/task/f'fold_{fold}';directory.mkdir(parents=True,exist_ok=True)
            predictions.to_csv(directory/'test_predictions.csv',index=False)
            summary.update(fold=fold,manifest_sha256=sha(path))
            write_json(directory/'summary.json',summary);summaries.append(summary)
        combined=pd.concat(frames,ignore_index=True)
        expected=set(master.loc[master.b1==master.b2,'sample_id']) if task=='consensus' else set(master.sample_id)
        if combined.sample_id.duplicated().any() or set(combined.sample_id)!=expected:
            raise ValueError('OOF cohort coverage mismatch')
        combined.to_csv(args.output/task/'oof_predictions.csv',index=False)
    write_json(args.output/'summary.json',summaries)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--master-root',type=Path,default=Path('output/wd_multimodal_master_repaired'))
    p.add_argument('--output',type=Path,default=Path('output/wd_presentation_3day/baselines'))
    main(p.parse_args())
