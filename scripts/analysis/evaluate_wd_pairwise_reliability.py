"""Pairwise WD agreement with matched human baselines and patient bootstrap CIs.

Run from the repository root. Requires only numpy and pandas; no GPU.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def coefficients(a, b, scale):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if scale == 'binary':
        observed = np.mean(a == b)
        p = (a.mean() + b.mean()) / 2
        expected = 2 * p * (1 - p)
        return {'agreement': observed, 'AC1': (observed - expected) / (1 - expected)}
    # Fixed five-category scale; round half upward only for categorical AC2.
    ar, br = np.floor(a + .5).astype(int), np.floor(b + .5).astype(int)
    w = 1 - ((np.arange(5)[:, None] - np.arange(5)[None, :]) / 4) ** 2
    margins = np.bincount(np.r_[ar, br], minlength=6)[1:6] / (2 * len(a))
    # Gwet's irrCAC::gwet.ac1.raw, fixed category universe 1..5.
    expected = w.sum() * np.sum(margins * (1 - margins)) / (5 * 4)
    observed = np.mean(w[ar - 1, br - 1])
    x = np.column_stack([a, b])
    n, k = x.shape
    grand, row, col = x.mean(), x.mean(axis=1), x.mean(axis=0)
    msr = k * np.square(row - grand).sum() / (n - 1)
    msc = n * np.square(col - grand).sum() / (k - 1)
    mse = np.square(x - row[:, None] - col[None, :] + grand).sum() / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return {'AC2_quadratic': (observed - expected) / (1 - expected),
            'ICC_A1': (msr - mse) / den if den else np.nan,
            'MAE': np.mean(abs(a - b)), 'within_one_point': np.mean(abs(a - b) <= 1)}


def load_predictions(spec, labels):
    path = Path(spec['path'])
    p = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    total = len(p)
    if 'status' in p:
        p = p[p.status.astype(str).str.lower().isin(['ok', 'success'])].copy()
    key = 'sample_id' if 'sample_id' in p else 'segment_uid'
    if p[key].isna().any() or p[key].duplicated().any():
        raise ValueError(f'{path}: missing or duplicate identifiers; rebuild OOF predictions')
    if not p[key].isin(labels[key]).all():
        raise ValueError(f'{path}: prediction IDs outside the master cohort')
    ordinal = None
    if spec['kind'] == 'regression':
        ordinal = next(c for c in ['WD_prediction', 'WD_P_pred', 'prediction'] if c in p)
        source = ordinal
        threshold, low, high = 2, 1, 5
    else:
        source = next(c for c in ['WD_probability', 'probability'] if c in p)
        ordinal = next((c for c in ['wd_p_score', 'WD_P_score'] if c in p), None)
        threshold, low, high = .5, 0, 1
    values = pd.to_numeric(p[source], errors='coerce')
    valid = values.between(low, high) & np.isfinite(values)
    p = p.loc[valid].copy()
    p['ai_binary'] = (values.loc[valid] >= threshold).astype(int)
    if ordinal:
        p['ai_score'] = pd.to_numeric(p[ordinal], errors='coerce')
        p.loc[~p.ai_score.between(1, 5), 'ai_score'] = np.nan
    cols = [key, 'ai_binary'] + (['ai_score'] if ordinal else [])
    d = labels.merge(p[cols], on=key, validate='one_to_one')
    return d, {'experiment': spec['experiment'], 'model': spec['model'],
               'source': str(path), 'input_rows': total, 'valid_binary_rows': len(d),
               'excluded_rows': total - len(d)}


def evaluate(d, scale, meta, repetitions, rng):
    cols = ['h1', 'h2', 'ai_score'] if scale == 'ordinal' else ['b1', 'b2', 'ai_binary']
    d = d.dropna(subset=cols).reset_index(drop=True)
    if len(d) < 2:
        return []
    x = d[cols].to_numpy(float)
    groups = [np.flatnonzero(d.patient_id.to_numpy() == p) for p in d.patient_id.unique()]

    def calc(z):
        pairs = {'human1_vs_human2': coefficients(z[:, 0], z[:, 1], scale),
                 'ai_vs_human1': coefficients(z[:, 2], z[:, 0], scale),
                 'ai_vs_human2': coefficients(z[:, 2], z[:, 1], scale)}
        baseline = pairs['human1_vs_human2']
        pairs['mean_ai_minus_human_baseline'] = {
            m: (pairs['ai_vs_human1'][m] + pairs['ai_vs_human2'][m]) / 2 - v
            for m, v in baseline.items()}
        return {(pair, m): v for pair, metrics in pairs.items() for m, v in metrics.items()}

    point = calc(x)
    boot = {key: [] for key in point}
    for _ in range(repetitions):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for key, value in calc(x[idx]).items():
            if np.isfinite(value):
                boot[key].append(value)
    rows = []
    for (pair, metric), value in point.items():
        samples = boot[pair, metric]
        ci = np.percentile(samples, [2.5, 97.5]) if samples else [np.nan, np.nan]
        rows.append({**meta, 'scale': scale, 'comparison': pair, 'metric': metric,
                     'N': len(d), 'patients': len(groups),
                     'human_disagreement_rows': int((d.b1 != d.b2).sum()),
                     'estimate': value, 'ci_low': ci[0], 'ci_high': ci[1],
                     'valid_bootstrap_replicates': len(samples)})
    return rows


def main(args):
    if args.bootstrap < 0:
        raise ValueError('--bootstrap must be nonnegative')
    labels = pd.read_csv(args.labels, encoding='utf-8-sig', low_memory=False)
    labels['h1'] = pd.to_numeric(labels.WD_P_rater1, errors='raise')
    labels['h2'] = pd.to_numeric(labels.WD_P_rater2, errors='raise')
    if labels.patient_id.isna().any() or not labels[['h1', 'h2']].isin(range(1, 6)).all().all():
        raise ValueError('Master labels need patient IDs and integer human ratings 1–5')
    labels['b1'], labels['b2'] = (labels.h1 >= 2).astype(int), (labels.h2 >= 2).astype(int)
    specs = json.loads(args.specs.read_text(encoding='utf-8-sig'))
    loaded, audit = [], []
    for spec in specs:
        d, info = load_predictions(spec, labels)
        loaded.append((spec, d)); audit.append(info)
    result = []
    rng = np.random.default_rng(args.seed)
    for spec, d in loaded:
        print(f"Evaluating {spec['experiment']} / {spec['model']}: {len(d)} rows", flush=True)
        meta = {'experiment': spec['experiment'], 'model': spec['model'], 'population': 'available'}
        for scale in ['binary', 'ordinal']:
            if scale == 'ordinal' and 'ai_score' not in d:
                continue
            result += evaluate(d, scale, meta, args.bootstrap, rng)
        # Common successful IDs per experiment allow fair modality comparison.
        peers = [peer for other, peer in loaded if other['experiment'] == spec['experiment']]
        for scale in ['binary', 'ordinal']:
            if scale == 'ordinal' and any('ai_score' not in peer for peer in peers):
                continue
            common = set.intersection(*(set(peer.dropna(subset=['ai_score']).sample_id)
                       if scale == 'ordinal' else set(peer.sample_id) for peer in peers))
            matched = d[d.sample_id.isin(common)]
            result += evaluate(matched, scale, {**meta, 'population': 'common_models'}, args.bootstrap, rng)
    args.output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(result)
    table.to_csv(args.output / 'pairwise_reliability.csv', index=False)
    for scale in ['binary', 'ordinal']:
        table[table.scale == scale].to_csv(args.output / f'{scale}_reliability.csv', index=False)
    pd.DataFrame(audit).to_csv(args.output / 'coverage.csv', index=False)
    lines = ['# Pairwise AI–human reliability', '',
             'Each human baseline uses the exact same rows as its AI comparisons. CIs resample patients.',
             'Consensus-only human binary agreement is perfect by selection, not an independent benchmark.',
             'AC2 rounds continuous predictions half upward; ICC, MAE and within-one-point use original scores.',
             'Binary decisions follow saved probabilities >=0.5 (regression uses score >=2).',
             'The baseline difference is mean(AI–human1, AI–human2) minus human–human. Negative MAE difference favors AI.',
             'Intervals containing zero do not establish equivalence or human-level performance.', '',
             '| Experiment | Model | Population | Scale | Comparison | Metric | N | Estimate | 95% CI |',
             '|---|---|---|---|---|---|---:|---:|---|']
    for r in result:
        lines.append('| ' + ' | '.join(str(r[k]) for k in ['experiment', 'model', 'population', 'scale', 'comparison', 'metric', 'N']) +
                     f" | {r['estimate']:.3f} | [{r['ci_low']:.3f}, {r['ci_high']:.3f}] |")
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Results: {args.output.resolve()}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, default=Path('output/wd_multimodal_master_repaired/paired_master_soft.csv'))
    p.add_argument('--specs', type=Path, default=Path(__file__).with_name('wd_pairwise_reliability_specs.json'))
    p.add_argument('--output', type=Path, default=Path('output/wd_pairwise_reliability'))
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    main(p.parse_args())
