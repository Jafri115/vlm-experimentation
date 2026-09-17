"""Shared, CPU-only validation and statistics for the presentation experiments."""
import contextlib
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_folds(root):
    from evaluate_wd_learning import read_master, baseline_rows
    root = Path(root)
    master = read_master(root/'paired_master_soft.csv')
    assignments, _ = baseline_rows(root, master, 'regression')
    paths = sorted(root.glob('fold_*/master_manifest.csv'))
    if len(paths) != 5:
        raise ValueError('Expected the frozen five-fold cohort')
    master = master.merge(assignments[['sample_id', 'fold']], on='sample_id', validate='one_to_one')
    folds = [(int(p.parent.name.split('_')[-1]), read_master(p), p) for p in paths]
    return master, folds


def load_oof(path, master, column):
    p = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    if 'status' in p:
        p = p[p.status.astype(str).str.lower().isin(['ok', 'success'])].copy()
    key = 'sample_id' if 'sample_id' in p else 'segment_uid'
    if key not in p or p[key].isna().any() or p[key].duplicated().any():
        raise ValueError(f'{path}: missing/duplicate IDs')
    if not p[key].isin(master[key]).all():
        raise ValueError(f'{path}: IDs outside frozen cohort')
    if 'outer_fold' not in p:
        raise ValueError(f'{path}: outer_fold required to verify the held-out checkpoint')
    # Older VLM regressors export WD_P_pred, while the text model and
    # TF-IDF baseline export WD_prediction. Normalize only prediction fields.
    aliases = {'WD_prediction': ['WD_P_pred', 'prediction'],
               'WD_probability': ['probability']}
    source = column
    if source not in p:
        candidates = [name for name in aliases.get(column, []) if name in p]
        if not candidates:
            raise ValueError(f'{path}: missing prediction column {column!r}; '
                             f'accepted alternatives: {aliases.get(column, [])}; '
                             f'available columns: {list(p.columns)}')
        source = candidates[0]
        for other in candidates[1:]:
            if not pd.to_numeric(p[source], errors='coerce').equals(
                    pd.to_numeric(p[other], errors='coerce')):
                raise ValueError(f'{path}: conflicting prediction columns {candidates}')
        print(f'{path}: using {source} as {column}', flush=True)
    values = pd.to_numeric(p[source], errors='coerce')
    if not np.isfinite(values).all():
        raise ValueError(f'{path}: invalid predictions')
    if column == 'WD_probability' and not values.between(0, 1).all():
        raise ValueError(f'{path}: probability outside [0,1]')
    p[column] = values
    d = master.merge(p[[key, 'outer_fold', column]], on=key, validate='one_to_one')
    if not (d.fold == pd.to_numeric(d.outer_fold)).all():
        raise ValueError(f'{path}: OOF folds do not match frozen folds')
    return d


def metrics(d, values, task):
    p = np.asarray(values, float)
    if task == 'regression':
        y = d.target.to_numpy()
        corr = pd.Series(y).rank().corr(pd.Series(p).rank()) if np.std(y)>0 and np.std(p)>0 else np.nan
        return dict(N=len(d), MAE=float(np.mean(abs(y-p))), RMSE=float(np.sqrt(np.mean((y-p)**2))), Spearman=corr)
    soft = d.soft_target.to_numpy()
    c = d.b1.to_numpy() == d.b2.to_numpy()
    t, q = d.b1.to_numpy()[c], p[c]
    h = q >= .5
    tp, tn = int(((t==1)&h).sum()), int(((t==0)&~h).sum())
    fp, fn = int(((t==0)&h).sum()), int(((t==1)&~h).sum())
    recall = tp/(tp+fn) if tp+fn else np.nan
    specificity = tn/(tn+fp) if tn+fp else np.nan
    precision = tp/(tp+fp) if tp+fp else 0.
    return dict(N=len(d), consensus_N=int(c.sum()), Brier=float(np.mean((soft-p)**2)),
                balanced_accuracy=(recall+specificity)/2, recall=recall, specificity=specificity,
                precision=precision, F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                AUROC=float(roc_auc_score(t,q)) if len(np.unique(t))==2 else np.nan,
                average_precision=float(average_precision_score(t,q)) if len(np.unique(t))==2 else np.nan,
                TP=tp,TN=tn,FP=fp,FN=fn)


def paired_interval(d, gains, rng, repetitions=2000):
    from evaluate_wd_learning import bootstrap_gain
    return bootstrap_gain(d, np.asarray(gains), repetitions, rng)


@contextlib.contextmanager
def exclusive_lock(path):
    """OS releases the lock even on a crash; the harmless file can remain."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        if path.stat().st_size == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'Another queue owns {path}; do not launch a second one') from exc
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)
