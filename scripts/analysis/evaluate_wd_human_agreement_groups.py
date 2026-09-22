"""Compare AI ratings on exact human agreement and disagreement groups; CPU only."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.evaluate_wd_pairwise_reliability import load_predictions


def summarize(d, scale):
    if scale == 'ordinal':
        d = d.dropna(subset=['ai_score'])
        a = d.ai_score.to_numpy(float)
        decision = np.floor(a + .5)
        h1, h2 = d.h1.to_numpy(), d.h2.to_numpy()
    else:
        a = decision = d.ai_binary.to_numpy()
        h1, h2 = d.b1.to_numpy(), d.b2.to_numpy()
    n = len(d)
    m1, m2 = decision == h1, decision == h2
    result = {'N': n, 'patients': d.patient_id.nunique(),
              'human_exact_agreement_n': int((d.h1 == d.h2).sum()),
              'human_binary_agreement_n': int((d.b1 == d.b2).sum()),
              'match_human1_n': int(m1.sum()), 'match_human2_n': int(m2.sum()),
              'match_both_n': int((m1 & m2).sum()),
              'match_either_n': int((m1 | m2).sum()),
              'match_neither_n': int((~m1 & ~m2).sum())}
    for name in ['match_human1', 'match_human2', 'match_both', 'match_either', 'match_neither']:
        result[name + '_rate'] = result[name + '_n'] / n if n else np.nan
    result['mean_pairwise_agreement'] = (m1.sum() + m2.sum()) / (2*n) if n else np.nan
    if scale == 'ordinal':
        result['mean_pairwise_MAE'] = (np.abs(a-h1).mean() + np.abs(a-h2).mean())/2 if n else np.nan
        result['within_one_of_both_rate'] = float(((abs(a-h1) <= 1) & (abs(a-h2) <= 1)).mean()) if n else np.nan
    return result


def groups(d):
    same = d.h1 == d.h2
    yield 'exact_agreement', d[same]
    yield 'exact_disagreement', d[~same]
    yield 'different_scores_same_binary', d[(~same) & (d.b1 == d.b2)]
    yield 'different_binary', d[d.b1 != d.b2]
    for score in range(1, 6):
        yield f'both_score_{score}', d[same & (d.h1 == score)]


def main(args):
    labels = pd.read_csv(args.labels, encoding='utf-8-sig', low_memory=False)
    labels['h1'] = pd.to_numeric(labels.WD_P_rater1, errors='raise')
    labels['h2'] = pd.to_numeric(labels.WD_P_rater2, errors='raise')
    if (labels.sample_id.isna().any() or labels.sample_id.duplicated().any()
            or labels.patient_id.isna().any()
            or not labels[['h1', 'h2']].isin(range(1, 6)).all().all()):
        raise ValueError('Master needs unique sample_id, patient_id and integer ratings 1-5')
    labels['b1'], labels['b2'] = (labels.h1 >= 2).astype(int), (labels.h2 >= 2).astype(int)
    args.output.mkdir(parents=True, exist_ok=True)
    counts = [{'group': name, 'N': len(d), 'patients': d.patient_id.nunique()} for name, d in groups(labels)]
    pd.DataFrame(counts).to_csv(args.output/'human_group_counts.csv', index=False)
    if args.human_counts_only:
        print(pd.DataFrame(counts).to_string(index=False))
        return
    specs = json.loads(args.specs.read_text(encoding='utf-8-sig'))
    missing = [s['path'] for s in specs if not Path(s['path']).exists()]
    if missing:
        raise SystemExit('Row-level predictions are required; summaries cannot recover these groups.\nMissing:\n' + '\n'.join(missing))
    loaded, audit = [], []
    for spec in specs:
        d, coverage = load_predictions(spec, labels)
        loaded.append((spec, d))
        audit.append(coverage)
    rows = []
    for spec, d in loaded:
        peers = [p for s, p in loaded if s['experiment'] == spec['experiment']]
        for scale in ['binary', 'ordinal']:
            if scale == 'ordinal' and 'ai_score' not in d:
                continue
            valid = d.dropna(subset=['ai_score']) if scale == 'ordinal' else d
            populations = [('available', valid)]
            if scale == 'binary' or all('ai_score' in p for p in peers):
                common = set.intersection(*(set((p.dropna(subset=['ai_score']) if scale == 'ordinal' else p).sample_id) for p in peers))
                populations.append(('common_models', valid[valid.sample_id.isin(common)]))
            for population, selected in populations:
                for group, subset in groups(selected):
                    rows.append({'experiment': spec['experiment'], 'model': spec['model'],
                                 'population': population, 'scale': scale, 'group': group,
                                 **summarize(subset, scale)})
    table = pd.DataFrame(rows)
    table.to_csv(args.output/'agreement_by_human_group.csv', index=False)
    pd.DataFrame(audit).to_csv(args.output/'coverage.csv', index=False)
    lines = ['# AI agreement by human agreement group', '',
             'Exact agreement means identical original 1-5 scores, e.g. 3 and 3. Disagreement includes 2 and 3 even though both indicate withdrawal.',
             'Tables use common successful model IDs within each experiment. Cohorts can differ across experiments.',
             'Ordinal matching rounds AI scores half upward; MAE retains continuous scores. Probability-only classifiers have binary results only.',
             'On disagreement rows there is no agreed human answer: matching either human is descriptive, not accuracy.',
             'If humans disagree on binary labels, any valid binary AI decision necessarily matches one human. Either-human agreement is then 100% and average pairwise agreement 50% by definition.',
             'On exact-agreement rows, matching either human and both humans are identical. Subgroups overlap: both_score_N subdivides exact_agreement; the two disagreement types subdivide exact_disagreement.', '']
    for scale in ['ordinal', 'binary']:
        lines += [f'## {scale.title()} agreement', '',
                  '| Experiment | Model | Human group | N | Match H1 | Match H2 | Match both | Match either |',
                  '|---|---|---|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['population'] != 'common_models' or r['scale'] != scale:
                continue
            fmt = lambda key: f'{r[key]:.1%}' if r['N'] else 'N/A'
            lines.append(f"| {r['experiment']} | {r['model']} | {r['group']} | {r['N']} | " +
                         ' | '.join(fmt(k) for k in ['match_human1_rate', 'match_human2_rate', 'match_both_rate', 'match_either_rate']) + ' |')
        lines.append('')
    (args.output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Results: {args.output.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=Path('output/wd_multimodal_master_repaired/paired_master_soft.csv'))
    parser.add_argument('--specs', type=Path, default=Path(__file__).with_name('wd_pairwise_reliability_specs.json'))
    parser.add_argument('--output', type=Path, default=Path('output/wd_human_agreement_groups'))
    parser.add_argument('--human-counts-only', action='store_true')
    main(parser.parse_args())
