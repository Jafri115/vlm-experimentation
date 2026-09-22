"""Assemble paired baseline comparisons, content checks, seeds and presentation figures."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.wd_presentation_common import load_folds,load_oof,metrics,paired_interval,write_json


def main(args):
    master,_=load_folds(args.master_root)
    specs=json.loads(args.specs.read_text(encoding='utf-8-sig'))
    table,gains=[],[]
    args.output.mkdir(parents=True,exist_ok=True)
    for task,experiment in [('consensus','consensus fine-tuning'),('regression','regression')]:
        column='WD_probability' if task=='consensus' else 'WD_prediction'
        sources={'TF-IDF':args.root/'baselines'/task/'oof_predictions.csv'}
        sources.update({s['model']:Path(s['path']) for s in specs if s['experiment']==experiment})
        loaded={name:load_oof(path,master,column) for name,path in sources.items()}
        common=set.intersection(*(set(d.sample_id) for d in loaded.values()))
        cohort=master[master.sample_id.isin(common)].copy()
        if task=='consensus':cohort=cohort[cohort.b1==cohort.b2]
        cohort=cohort.sort_values('sample_id').reset_index(drop=True)
        if cohort.empty:raise ValueError('No paired evaluation rows')
        predictions={name:cohort[['sample_id']].merge(d[['sample_id',column]],on='sample_id',validate='one_to_one')[column].to_numpy() for name,d in loaded.items()}
        from evaluate_wd_learning import baseline_rows
        baseline,_=baseline_rows(args.master_root,master,experiment)
        base_column='train_prevalence' if task=='consensus' else 'train_mean'
        predictions['Training constant']=cohort[['sample_id']].merge(baseline,on='sample_id',validate='one_to_one')[base_column].to_numpy()
        y=cohort.soft_target.to_numpy() if task=='consensus' else cohort.target.to_numpy()
        loss=lambda p:(p-y)**2 if task=='consensus' else abs(p-y)
        for name,pred in predictions.items():
            table.append({'task':task,'model':name,**metrics(cohort,pred,task)})
            for comparator in ['TF-IDF','Training constant']:
                if name==comparator:continue
                gains.append({'task':task,'model':name,'baseline':comparator,
                              'loss':'Brier' if task=='consensus' else 'MAE',
                              **paired_interval(cohort,loss(predictions[comparator])-loss(pred),np.random.default_rng(42),args.bootstrap)})
    pd.DataFrame(table).to_csv(args.output/'model_comparison.csv',index=False)
    pd.DataFrame(gains).to_csv(args.output/'paired_gains.csv',index=False)
    seeds=[]
    for path in sorted((args.root/'seeds').glob('seed_*/oof_predictions.csv')):
        d=load_oof(path,master,'WD_probability')
        if set(d.sample_id)!=set(master.sample_id):raise ValueError(f'{path}: incomplete seed cohort')
        seeds.append({'seed':int(path.parent.name.split('_')[-1]),**metrics(d,d.WD_probability,'soft')})
    pd.DataFrame(seeds).to_csv(args.output/'seed_metrics.csv',index=False)
    lines=['# Three-day presentation experiments','','## Same-cohort model comparison','',
           'Binary decisions use fixed threshold 0.5 and binary-consensus rows. Severity targets use mean human ratings. TF-IDF regularization was selected on validation only.','',
           '| Task | Model | N | Balanced accuracy | AUROC | Brier | MAE | RMSE |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        vals=' | '.join(f'{r[k]:.3f}' if k in r and np.isfinite(r[k]) else '—' for k in ['balanced_accuracy','AUROC','Brier','MAE','RMSE'])
        lines.append(f"| {r['task']} | {r['model']} | {r['N']} | {vals} |")
    lines+=['','## Improvement over baselines','','Positive values mean the named model has lower error. Intervals resample patients and are exploratory, not adjusted for multiple comparisons.','',
            '| Task | Model | Baseline | Loss | Improvement | Patient-bootstrap 95% CI |','|---|---|---|---|---:|---|']
    for r in gains:
        lines.append(f"| {r['task']} | {r['model']} | {r['baseline']} | {r['loss']} | {r['gain']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] |")
    swap_dir=args.root/'content_swap'
    if not (swap_dir/'metrics.csv').exists():raise ValueError('Content-swap diagnostic missing')
    swap=pd.read_csv(swap_dir/'metrics.csv')
    content=pd.read_csv(swap_dir/'content_gain.csv')
    lines+=['','## Does the matching content matter?','','These are saved-prediction input exchanges within patient and held-out fold, not fresh inference. This is equivalent for independent segment inference under the same checkpoint.','',
            '| Model | Condition | Replicate | N | Balanced accuracy | Brier |','|---|---|---:|---:|---:|---:|']
    for _,r in swap.iterrows():
        lines.append(f'| {r.model} | {r.condition} | {r.replicate} | {r.N} | {r.balanced_accuracy:.3f} | {r.Brier:.4f} |')
    for _,r in content.iterrows():
        lines+=['',f'{r.model}: mean swapped-minus-correct Brier = {r.gain:.4f}, patient-bootstrap interval [{r.ci_low:.4f}, {r.ci_high:.4f}]. Positive favors the correct content.']
    lines+=['','Swapping preserves patient identity but may change session. Similar withdrawal ratings across segments can weaken this diagnostic. No formal chance-test p-value is claimed. The intervals condition on these shuffles, not all possible mappings.']
    if seeds:
        lines+=['','## Controlled LLM soft-label seeds','','| Seed | N | Balanced accuracy | Brier |','|---:|---:|---:|---:|']
        for r in seeds:lines.append(f"| {r['seed']} | {r['N']} | {r['balanced_accuracy']:.3f} | {r['Brier']:.4f} |")
        for key in ['balanced_accuracy','Brier']:
            v=np.array([r[key] for r in seeds])
            lines+=['',f'{key}: mean {v.mean():.4f}; range {v.min():.4f}–{v.max():.4f}. Seeds are repeated fits on the same data, not independent cohorts.']
    else:lines+=['','Controlled seed repeats were not requested or have not completed.']
    lines+=['','## Suggested slide order','','1. Frozen patient folds and paired evaluation counts.','2. TF-IDF versus LLM/VLM and training-constant baselines.','3. Correct versus swapped content, with patient-level uncertainty.','4. Existing severity-collapse plot; seed stability if available.','','Model selection was not changed based on test scores. Results remain exploratory because this cohort has already been examined. See docs/wd_three_day_experiments.md for methods and limitations.']
    (args.output/'presentation_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if not args.no_plots:plot_results(pd.DataFrame(table),swap,args.output,seeds)


def plot_results(table,swap,output,seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,task,key,title in [(axes[0],'consensus','balanced_accuracy','Binary withdrawal (higher is better)'),(axes[1],'regression','MAE','Severity MAE (lower is better)')]:
        d=table[table.task==task]
        ax.bar(d.model,d[key]);ax.set_title(title);ax.tick_params(axis='x',rotation=20)
        if task=='consensus':ax.set_ylim(0,1);ax.axhline(.5,ls='--',color='gray')
        for i,v in enumerate(d[key]):ax.text(i,v,f'{v:.3f}',ha='center',va='bottom')
    fig.tight_layout();fig.savefig(output/'model_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,key,title in [(axes[0],'balanced_accuracy','Consensus balanced accuracy'),(axes[1],'Brier','All-row Brier error (lower is better)')]:
        for j,(model,d) in enumerate(swap.groupby('model',sort=False)):
            correct=d[d.condition=='correct'][key].iloc[0]
            shuffled=d[d.condition!='correct'][key]
            ax.plot([j*3,j*3+1],[correct,shuffled.mean()],'-',color='gray')
            ax.scatter([j*3],[correct],label=f'{model}: correct')
            ax.scatter(np.full(len(shuffled),j*3+1),shuffled,marker='x',label=f'{model}: swaps')
        ax.set_title(title);ax.set_xticks([]);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'content_swap.png',dpi=180);plt.close(fig)
    if seeds:
        fig,ax=plt.subplots(figsize=(6,4))
        ax.plot([str(r['seed']) for r in seeds],[r['balanced_accuracy'] for r in seeds],'o-')
        ax.set(xlabel='Training seed',ylabel='Balanced accuracy',ylim=(0,1),title='Controlled LLM soft-label repeats')
        fig.tight_layout();fig.savefig(output/'seed_stability.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--master-root',type=Path,default=Path('output/wd_multimodal_master_repaired'))
    p.add_argument('--root',type=Path,default=Path('output/wd_presentation_3day'))
    p.add_argument('--output',type=Path,default=Path('output/wd_presentation_3day/report'))
    p.add_argument('--specs',type=Path,default=Path(__file__).with_name('wd_pairwise_reliability_specs.json'))
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--no-plots',action='store_true')
    main(p.parse_args())
