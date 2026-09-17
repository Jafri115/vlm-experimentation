"""Audit saved OOF predictions against fold-training-only baselines. No model execution."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate_wd_pairwise_reliability import coefficients, load_predictions


def read_master(path):
    d = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    d['h1'], d['h2'] = pd.to_numeric(d.WD_P_rater1), pd.to_numeric(d.WD_P_rater2)
    if d.sample_id.isna().any() or d.sample_id.duplicated().any() or d.patient_id.isna().any():
        raise ValueError(f'{path}: invalid identifiers')
    if not d[['h1', 'h2']].isin(range(1, 6)).all().all():
        raise ValueError(f'{path}: ratings must be integers 1-5')
    d['b1'], d['b2'] = (d.h1 >= 2).astype(int), (d.h2 >= 2).astype(int)
    d['target'] = (d.h1 + d.h2) / 2
    d['soft_target'] = (d.b1 + d.b2) / 2
    return d


def baseline_rows(root, labels, experiment):
    paths = sorted(root.glob('fold_*/master_manifest.csv'))
    if not paths:
        raise ValueError(f'No fold manifests under {root}')
    rows, audit = [], []
    for path in paths:
        d = read_master(path)
        split = d['split'].astype(str).str.lower()
        if not split.isin(['train', 'val', 'test']).all():
            raise ValueError(f'{path}: invalid split')
        patients = {s: set(d.loc[split == s, 'patient_id']) for s in ['train', 'val', 'test']}
        if any(patients[a] & patients[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
            raise ValueError(f'{path}: patient leakage')
        # Verify the frozen ratings, not just the sample identifiers.
        checked = d[['sample_id', 'h1', 'h2', 'patient_id']].merge(
            labels[['sample_id', 'h1', 'h2', 'patient_id']], on='sample_id', suffixes=('', '_master'), validate='one_to_one')
        if len(checked) != len(d) or any(not checked[c].equals(checked[c+'_master']) for c in ['h1', 'h2', 'patient_id']):
            raise ValueError(f'{path}: labels differ from master')
        train = d[split == 'train']
        # Consensus-trained tasks use consensus-only training baselines.
        if experiment in ['consensus fine-tuning', 'zero-shot', '3+3 few-shot']:
            train = train[train.b1 == train.b2]
        if train.empty:
            raise ValueError(f'{path}: empty training subset')
        pooled = pd.concat([train.h1, train.h2])
        constants = {'train_mean': train.target.mean(), 'train_median': train.target.median(),
                     'train_mode': float(pooled.mode().min()),
                     'train_prevalence': train.soft_target.mean()}
        freq = pooled.value_counts(normalize=True).reindex(range(1, 6), fill_value=0)
        fold = int(path.parent.name.split('_')[-1])
        for uid in d.loc[split == 'test', 'sample_id']:
            rows.append({'sample_id': uid, 'fold': fold, **constants,
                         **{f'train_freq_{i}': freq[i] for i in range(1, 6)}})
        audit.append({'experiment': experiment, 'fold': fold, 'training_N': len(train),
                      'test_N': int((split == 'test').sum()), **constants})
    result = pd.DataFrame(rows)
    if result.sample_id.duplicated().any() or set(result.sample_id) != set(labels.sample_id):
        raise ValueError('Fold test sets must cover master exactly once')
    return result, audit


def rank_corr(a, b):
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    return a.corr(b) if a.nunique() > 1 and b.nunique() > 1 else np.nan


def ordinal_metrics(d, pred):
    p, y = np.asarray(pred, float), d.target.to_numpy(float)
    h1, h2 = d.h1.to_numpy(), d.h2.to_numpy()
    r = np.floor(p + .5)
    variance = np.var(y)
    result = {'N': len(d), 'MAE_vs_human_mean': np.mean(abs(p-y)),
              'RMSE_vs_human_mean': np.sqrt(np.mean((p-y)**2)),
              'R2_vs_human_mean': 1-np.mean((p-y)**2)/variance if variance else np.nan,
              'Spearman_vs_human_mean': rank_corr(y, p),
              'prediction_mean': p.mean(), 'prediction_sd': p.std(),
              'target_sd': y.std(), 'sd_ratio': p.std()/y.std() if y.std() else np.nan,
              'bias_vs_human_mean': np.mean(p-y), 'prediction_min': p.min(), 'prediction_max': p.max(),
              'overestimate_human_mean_rate': np.mean(p > y),
              'underestimate_human_mean_rate': np.mean(p < y),
              'rounded_pairwise_exact': np.mean((r == h1).astype(float)+(r == h2))/2,
              'rounded_match_either': np.mean((r == h1)|(r == h2)),
              'rounded_match_both': np.mean((r == h1)&(r == h2))}
    same = h1 == h2
    result['exact_agreement_macro_score_recall'] = np.mean([
        np.mean(r[same & (h1 == score)] == score) for score in np.unique(h1[same])
    ]) if same.any() else np.nan
    centered = pd.DataFrame({'patient': d.patient_id.to_numpy(), 'y': y, 'p': p})
    centered[['y', 'p']] -= centered.groupby('patient')[['y', 'p']].transform('mean')
    result['within_patient_centered_Spearman'] = rank_corr(centered.y, centered.p)
    if len(d) >= 2:
        pairs = [coefficients(p, h, 'ordinal') for h in [h1, h2]]
        human = coefficients(h1, h2, 'ordinal')
        for metric in ['AC2_quadratic', 'ICC_A1', 'MAE', 'within_one_point']:
            result['AI_human_mean_'+metric] = np.mean([z[metric] for z in pairs])
            result['human_human_'+metric] = human[metric]
    return result


def binary_metrics(d, probability):
    p = np.asarray(probability, float)
    y = d.soft_target.to_numpy()
    hard = p >= .5
    h1, h2 = d.b1.to_numpy(), d.b2.to_numpy()
    result = {'N': len(d), 'Brier_vs_soft_target': np.mean((p-y)**2),
              'pairwise_binary_agreement': np.mean((hard == h1).astype(float)+(hard == h2))/2,
              'predicted_positive_rate': hard.mean(), 'probability_sd': p.std()}
    c = h1 == h2
    result['consensus_N'] = int(c.sum())
    if c.any():
        t, h = h1[c], hard[c]
        tp, tn = int(((t == 1)&h).sum()), int(((t == 0)&~h).sum())
        fp, fn = int(((t == 0)&h).sum()), int(((t == 1)&~h).sum())
        recall = tp/(tp+fn) if tp+fn else np.nan
        specificity = tn/(tn+fp) if tn+fp else np.nan
        result.update(TP=tp, TN=tn, FP=fp, FN=fn, sensitivity=recall, specificity=specificity,
                      balanced_accuracy=(recall+specificity)/2,
                      accuracy=(tp+tn)/len(t),
                      AUROC=roc_auc_score(t, p[c]) if len(np.unique(t)) == 2 else np.nan,
                      average_precision=average_precision_score(t, p[c]) if len(np.unique(t)) == 2 else np.nan)
    return result


def bootstrap_gain(d, gains, repetitions, rng):
    # Paired patient-cluster resampling; positive gain means lower model error.
    z = pd.DataFrame({'patient': d.patient_id.to_numpy(), 'gain': gains})
    agg = z.groupby('patient').gain.agg(['sum', 'count']).to_numpy()
    draws = rng.integers(0, len(agg), (repetitions, len(agg)))
    values = agg[draws, 0].sum(axis=1)/agg[draws, 1].sum(axis=1)
    low, high = np.percentile(values, [2.5, 97.5])
    return {'gain': np.mean(gains), 'ci_low': low, 'ci_high': high,
            'patients': len(agg), 'N': len(d)}


def main(args):
    if args.bootstrap < 1:
        raise ValueError('--bootstrap must be positive')
    specs = json.loads(args.specs.read_text(encoding='utf-8-sig'))
    missing = [s['path'] for s in specs if not Path(s['path']).exists()]
    if missing:
        raise SystemExit('Need row-level OOF files on the experiment machine:\n'+'\n'.join(missing))
    labels = read_master(args.master_root/'paired_master_soft.csv')
    loaded, coverage, training = [], [], []
    for spec in specs:
        d, audit = load_predictions(spec, labels)
        base, tr = baseline_rows(args.master_root, labels, spec['experiment'])
        training.extend([{**r, 'model': spec['model']} for r in tr])
        d = d.merge(base, on='sample_id', validate='one_to_one')
        raw = pd.read_csv(spec['path'], encoding='utf-8-sig')
        if 'status' in raw:
            raw = raw[raw.status.astype(str).str.lower().isin(['ok', 'success'])]
        key = 'sample_id' if 'sample_id' in raw else 'segment_uid'
        audit['oof_fold_verified'] = 'outer_fold' in raw
        if 'outer_fold' in raw:
            checked = d[[key, 'fold']].merge(raw[[key, 'outer_fold']], on=key, validate='one_to_one')
            if not (checked.fold == pd.to_numeric(checked.outer_fold, errors='raise')).all():
                raise ValueError(f"{spec['path']}: prediction fold differs from frozen test fold")
        if spec['kind'] != 'regression':
            col = next(c for c in ['WD_probability', 'probability'] if c in raw)
            raw['ai_probability'] = pd.to_numeric(raw[col], errors='coerce')
            d = d.merge(raw[[key, 'ai_probability']], on=key, validate='one_to_one')
        loaded.append((spec, d)); coverage.append(audit)
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    metrics, gains, details, rating_rows, calibration = [], [], [], [], []
    for number, (spec, d) in enumerate(loaded, 1):
        print(f"{number}/{len(loaded)} {spec['experiment']} {spec['model']}", flush=True)
        peers = [p for s, p in loaded if s['experiment'] == spec['experiment']]
        for scale, column in [('ordinal', 'ai_score'), ('binary', 'ai_probability')]:
            if any(column not in p for p in peers):
                continue
            common = set.intersection(*(set(p.dropna(subset=[column]).sample_id) for p in peers))
            selected = d[d.sample_id.isin(common)].copy()
            meta = {'experiment': spec['experiment'], 'model': spec['model'], 'scale': scale}
            if selected.empty:
                raise ValueError(f'No matched rows: {meta}')
            detail_cols = ['sample_id', 'patient_id', 'session_id', 'fold', 'h1', 'h2', 'target',
                           'b1', 'b2', column, 'train_mean', 'train_median', 'train_mode', 'train_prevalence']
            details.append(selected[detail_cols].assign(**meta))
            subsets = [('all', selected), ('exact_agreement', selected[selected.h1 == selected.h2]),
                       ('exact_disagreement', selected[selected.h1 != selected.h2])]
            subsets += [(f'fold_{f}', z) for f, z in selected.groupby('fold')]
            subsets += [(f'patient_{p}', z) for p, z in selected.groupby('patient_id')]
            for group, z in subsets:
                if z.empty:
                    continue
                candidates = {'AI': z[column]}
                if scale == 'ordinal':
                    candidates.update({b: z[b] for b in ['train_mean', 'train_median', 'train_mode']})
                    candidates.update(always_1=np.ones(len(z)), always_2=np.full(len(z), 2.))
                    func = ordinal_metrics
                else:
                    candidates.update(train_prevalence=z.train_prevalence,
                                      train_majority=(z.train_prevalence >= .5).astype(float))
                    func = binary_metrics
                for name, pred in candidates.items():
                    metrics.append({**meta, 'group': group, 'predictor': name, **func(z, pred)})
                if scale == 'ordinal':
                    freq = z[[f'train_freq_{v}' for v in range(1, 6)]].to_numpy()
                    support = np.arange(1, 6)[None, :]
                    expected_mae = (freq * abs(support-z.target.to_numpy()[:, None])).sum(axis=1)
                    expected_mse = (freq * (support-z.target.to_numpy()[:, None])**2).sum(axis=1)
                    metrics.append({**meta, 'group': group, 'predictor': 'random_training_rating_expected',
                                    'N': len(z), 'MAE_vs_human_mean': expected_mae.mean(),
                                    'RMSE_vs_human_mean': np.sqrt(expected_mse.mean())})
                if not group.startswith(('patient_', 'fold_')):
                    y = z.target.to_numpy() if scale == 'ordinal' else z.soft_target.to_numpy()
                    for baseline in (['train_mean', 'train_median', 'train_mode'] if scale == 'ordinal' else ['train_prevalence']):
                        for loss in (['absolute_error', 'squared_error'] if scale == 'ordinal' else ['squared_error']):
                            error = lambda v: np.abs(np.asarray(v)-y) if loss == 'absolute_error' else (np.asarray(v)-y)**2
                            gains.append({**meta, 'group': group, 'baseline': baseline, 'loss': loss,
                                          **bootstrap_gain(z, error(z[baseline])-error(z[column]), args.bootstrap, rng)})
                    if scale == 'ordinal':
                        gains.append({**meta, 'group': group, 'baseline': 'random_training_rating_expected',
                                      'loss': 'squared_error', **bootstrap_gain(z, expected_mse-(z[column].to_numpy()-y)**2, args.bootstrap, rng)})
            # Every ordered human-rating pair: where did the AI put these segments?
            for (h1, h2), z in selected.groupby(['h1', 'h2']):
                p = z[column].to_numpy()
                decisions = np.floor(p+.5).astype(int) if scale == 'ordinal' else (p >= .5).astype(int)
                row = {**meta, 'human1': h1, 'human2': h2, 'N': len(z),
                       'prediction_mean': p.mean(), 'prediction_sd': p.std(),
                       'prediction_q10': np.quantile(p, .1), 'prediction_median': np.median(p),
                       'prediction_q90': np.quantile(p, .9)}
                for value in (range(1, 6) if scale == 'ordinal' else range(2)):
                    row[f'predicted_{value}_n'] = int((decisions == value).sum())
                    row[f'predicted_{value}_rate'] = float((decisions == value).mean())
                rating_rows.append(row)
            if scale == 'binary':
                selected['bin'] = np.minimum((selected[column]*10).astype(int), 9)
                for b, z in selected.groupby('bin'):
                    calibration.append({**meta, 'bin_low': b/10, 'bin_high': (b+1)/10, 'N': len(z),
                                        'mean_probability': z[column].mean(), 'mean_human_positive_fraction': z.soft_target.mean()})
            if not args.no_plots:
                plot(selected, column, scale, args.output/f'{number:02d}_{scale}.png', spec)
    for filename, records in [('metrics.csv', metrics), ('baseline_gains.csv', gains),
                              ('human_pair_prediction_distributions.csv', rating_rows),
                              ('calibration.csv', calibration), ('coverage.csv', coverage),
                              ('training_baselines.csv', training)]:
        pd.DataFrame(records).to_csv(args.output/filename, index=False)
    pd.concat(details, ignore_index=True).to_csv(args.output/'row_level_audit.csv', index=False)
    lines = ['# Does the model beat simple baselines?', '',
             'Positive error reduction means the AI beats the baseline. CIs resample patients; they do not include repeated-training variability.',
             'The ordinal target below is the mean human score; binary Brier error uses the fraction of humans calling withdrawal present.',
             'Read per-score distributions and per-patient results alongside pooled metrics. Exact-agreement subsets exclude difficult human disagreements.', '',
             '| Experiment | Model | Scale | Group | Baseline | Loss | N | Error reduction | 95% CI |',
             '|---|---|---|---|---|---|---:|---:|---|']
    for r in gains:
        lines.append(f"| {r['experiment']} | {r['model']} | {r['scale']} | {r['group']} | {r['baseline']} | {r['loss']} | {r['N']} | {r['gain']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] |")
    lines += ['', 'See docs/wd_learning_evaluation.md for interpretation and limits. No test-set threshold optimization or model training was performed.']
    lines += ['', '## Ordinal performance and constant baselines', '',
              '| Experiment | Model | Group | Predictor | N | MAE to human mean | Spearman | Prediction SD | Mean AI-human ICC |',
              '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in metrics:
        if r['scale'] != 'ordinal' or r['group'] not in ['all', 'exact_agreement']:
            continue
        values = ' | '.join(f"{r.get(k, np.nan):.3f}" for k in ['MAE_vs_human_mean', 'Spearman_vs_human_mean', 'prediction_sd', 'AI_human_mean_ICC_A1'])
        lines.append(f"| {r['experiment']} | {r['model']} | {r['group']} | {r['predictor']} | {r['N']} | {values} |")
    (args.output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Results: {args.output.resolve()}')


def plot(d, column, scale, path, spec):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    same = d[d.h1 == d.h2]
    rounded = np.floor(same[column]+.5).astype(int) if scale == 'ordinal' else (same[column] >= .5).astype(int)
    cats = list(range(1, 6)) if scale == 'ordinal' else [0, 1]
    counts = pd.crosstab(same.h1, rounded).reindex(index=range(1, 6), columns=cats, fill_value=0)
    totals = counts.sum(axis=1)
    rates = counts.div(totals.replace(0, np.nan), axis=0)
    axes[0].imshow(rates, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    for i in range(5):
        for j in range(len(cats)):
            v = rates.iloc[i, j]
            axes[0].text(j, i, f'{v:.0%}' if np.isfinite(v) else 'N/A', ha='center', va='center', color='white' if v>.6 else 'black')
    axes[0].set(xticks=range(len(cats)), xticklabels=cats, yticks=range(5),
                yticklabels=[f'{i} (N={totals.loc[i]})' for i in range(1, 6)],
                xlabel='AI rounded score' if scale == 'ordinal' else 'AI binary label',
                ylabel='Identical human score', title='Where do AI predictions go?')
    if scale == 'ordinal':
        axes[1].scatter(d.target, d[column], alpha=.15, s=12)
        axes[1].plot([1, 5], [1, 5], '--', color='black')
        axes[1].set(xlim=(.9, 5.1), ylim=(.9, 5.1), xlabel='Mean human score', ylabel='AI score', title='Prediction spread (all matched segments)')
    else:
        b = np.minimum((d[column]*10).astype(int), 9)
        cal = d.assign(bin=b).groupby('bin').agg(p=(column, 'mean'), observed=('soft_target', 'mean'))
        axes[1].plot(cal.p, cal.observed, 'o-')
        axes[1].plot([0, 1], [0, 1], '--', color='black')
        axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel='AI probability', ylabel='Human positive fraction', title='Calibration (descriptive)')
    fig.suptitle(f"{spec['experiment']} / {spec['model']}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--master-root', type=Path, default=Path('output/wd_multimodal_master_repaired'))
    p.add_argument('--specs', type=Path, default=Path(__file__).with_name('wd_pairwise_reliability_specs.json'))
    p.add_argument('--output', type=Path, default=Path('output/wd_learning_evaluation'))
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-plots', action='store_true')
    main(p.parse_args())
