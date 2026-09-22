"""Same-cohort context comparison with paired patient-bootstrap uncertainty."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.wd_presentation_common import load_folds,load_oof,metrics,paired_interval


def markdown_table(frame, float_digits=4):
    """Render Markdown without pandas' optional tabulate dependency."""
    def display(value):
        if pd.isna(value):
            return ''
        if isinstance(value,(float,np.floating)):
            return f'{float(value):.{float_digits}f}'
        return str(value).replace('|','\\|').replace('\n',' ')
    columns=[str(column) for column in frame.columns]
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    lines.extend('| '+' | '.join(display(value) for value in row)+' |'
                 for row in frame.itertuples(index=False,name=None))
    return '\n'.join(lines)


def report(master_root,root):
    master,_=load_folds(master_root)
    expected=set(master.loc[master.b1==master.b2,'sample_id'])
    loaded={};rows=[]
    for model in ['tfidf','llm']:
        for variant in ['target_only','previous_context','previous_context_patient']:
            path=root/model/variant/'oof_predictions.csv'
            if not path.exists():continue
            d=load_oof(path,master,'WD_probability').sort_values('sample_id').reset_index(drop=True)
            if set(d.sample_id)!=expected:raise ValueError(f'{path}: incomplete consensus coverage')
            name=f'{model}/{variant}';loaded[name]=d
            rows.append({'model':model,'variant':variant,**metrics(d,d.WD_probability,'consensus')})
    table=pd.DataFrame(rows);table.to_csv(root/'metrics.csv',index=False)
    comparisons=[]
    pairs=[('llm/previous_context','llm/target_only'),('tfidf/previous_context','tfidf/target_only'),
           ('llm/previous_context','tfidf/previous_context'),
           ('llm/previous_context_patient','llm/previous_context')]
    for better,baseline in pairs:
        if better not in loaded or baseline not in loaded:continue
        d=loaded[better];other=loaded[baseline]
        gains=(other.WD_probability-d.soft_target)**2-(d.WD_probability-d.soft_target)**2
        interval=paired_interval(d,gains,np.random.default_rng(42))
        comparisons.append({'model':better,'baseline':baseline,'Brier_improvement':float(gains.mean()),
            'patient_bootstrap_95CI':str(interval)})
    comparison=pd.DataFrame(comparisons);comparison.to_csv(root/'comparisons.csv',index=False)
    cols=['model','variant','N','balanced_accuracy','AUROC','Brier','F1','recall','specificity']
    text='# Transcript context experiment\n\nFixed 0.5 threshold; identical consensus targets and frozen patient folds.\n\n'
    text+=markdown_table(table[cols]) if len(table) else 'No complete predictions yet.'
    text+='\n\nPositive Brier improvement favors the named model. Intervals resample patients, not segments.\n\n'
    if len(comparison):text+=markdown_table(comparison)
    text+='\n\nThese are exploratory comparisons on an already inspected cohort. Context benefit is not guaranteed. '
    text+='The optional patient pooling contrast changes pooling as well as the input representation; it is reported separately.\n'
    (root/'report.md').write_text(text,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if len(table):
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        labels=table.model+'/'+table.variant
        for ax,col in zip(axes,['balanced_accuracy','Brier']):
            ax.barh(labels,table[col]);ax.set_title(col);ax.set_xlim(0,1 if col=='balanced_accuracy' else max(.3,table[col].max()*1.1))
        axes[0].axvline(.5,color='gray',linestyle='--')
        fig.tight_layout();fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--master-root',type=Path,default=Path('output/wd_multimodal_master_repaired'))
    p.add_argument('--root',type=Path,default=Path('output/wd_context_experiment'))
    a=p.parse_args();report(a.master_root,a.root)
