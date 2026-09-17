"""Synthetic CPU integration checks; no model downloads or GPU execution."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wd_presentation_common import load_folds,load_oof,exclusive_lock
from run_wd_tfidf_baselines import main as baseline_main
from run_wd_content_swap import donors,main as swap_main
from report_wd_presentation import main as report_main


class PresentationTests(unittest.TestCase):
    def test_prediction_column_aliases(self):
        master=pd.DataFrame({'sample_id':['a','b'],'fold':[1,2]})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pred.csv'
            for alias in ['WD_prediction','WD_P_pred','prediction']:
                pd.DataFrame({'sample_id':['a','b'],'outer_fold':[1,2],alias:[1.2,2.3]}).to_csv(path,index=False)
                result=load_oof(path,master,'WD_prediction')
                np.testing.assert_allclose(result.WD_prediction,[1.2,2.3])
            pd.DataFrame({'sample_id':['a','b'],'outer_fold':[1,2],
                          'WD_P_mean':[1.2,2.3]}).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'available columns'):
                load_oof(path,master,'WD_prediction')
            pd.DataFrame({'sample_id':['a','b'],'outer_fold':[1,2],
                          'WD_P_pred':[1.2,2.3],'prediction':[2.,2.]}).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'conflicting prediction'):
                load_oof(path,master,'WD_prediction')

    def test_derangements_and_singletons(self):
        d=pd.DataFrame({'patient_id':[1,1,1,2,2,3],'fold':[1,1,1,2,2,3]})
        idx=donors(d,np.random.default_rng(42))
        self.assertEqual(idx[-1],-1)
        self.assertFalse((idx[:-1]==np.arange(5)).any())
        self.assertEqual(set(idx[:-1]),set(range(5)))
        self.assertTrue((d.patient_id.to_numpy()[:-1]==d.patient_id.to_numpy()[idx[:-1]]).all())

    def test_lock_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with exclusive_lock(Path(tmp)/'test.lock'):
                with self.assertRaises(RuntimeError):
                    with exclusive_lock(Path(tmp)/'test.lock'):pass
            with exclusive_lock(Path(tmp)/'test.lock'):pass

    def test_full_cpu_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);master_root=root/'master';master_root.mkdir()
            d=pd.DataFrame({'sample_id':[f's{i}' for i in range(20)],'segment_uid':[f'u{i}' for i in range(20)],
                'patient_id':np.repeat(range(10),2),'session_id':1,
                'WD_P_rater1':[1,3]*10,'WD_P_rater2':[1,3]*10,
                'transcript_text':['engaged cooperative responsive','avoidance withdrawal distant']*10})
            d.to_csv(master_root/'paired_master_soft.csv',index=False)
            for f in range(5):
                folder=master_root/f'fold_{f+1}';folder.mkdir()
                frame=d.copy()
                frame['split']=['test' if p//2==f else 'val' if p//2==(f+1)%5 else 'train' for p in frame.patient_id]
                frame.to_csv(folder/'master_manifest.csv',index=False)
            out=root/'results'
            baseline_main(argparse.Namespace(master_root=master_root,output=out/'baselines'))
            master,_=load_folds(master_root)
            pred=d[['sample_id']].copy();pred['outer_fold']=d.patient_id//2+1
            pred['WD_probability']=[.1,.9]*10;pred['WD_prediction']=[1.1,2.9]*10
            pred.to_csv(root/'pred.csv',index=False)
            swap_main(argparse.Namespace(master_root=master_root,llm_predictions=root/'pred.csv',
                vlm_predictions=root/'pred.csv',output=out/'content_swap',shuffles=3,bootstrap=20,seed=42))
            maps=pd.read_csv(out/'content_swap/swap_predictions.csv')
            self.assertTrue((maps.sample_id!=maps.donor_sample_id).all())
            gains=pd.read_csv(out/'content_swap/content_gain.csv')
            self.assertTrue((gains.gain>0).all())
            self.assertTrue((gains.ci_low>0).all())
            specs=[{'experiment':task,'model':model,'path':str(root/'pred.csv')}
                for task in ['consensus fine-tuning','regression'] for model in ['LLM','VLM']]
            # Exercise the historical VLM export format through the full report.
            pred.rename(columns={'WD_prediction':'WD_P_pred'}).to_csv(root/'vlm.csv',index=False)
            for spec in specs:
                if spec['model']=='VLM':spec['path']=str(root/'vlm.csv')
            (root/'specs.json').write_text(json.dumps(specs))
            report_main(argparse.Namespace(master_root=master_root,specs=root/'specs.json',root=out,
                output=out/'report',bootstrap=20,no_plots=False))
            self.assertTrue((out/'report/presentation_report.md').exists())
            self.assertTrue((out/'report/content_swap.png').exists())
            self.assertEqual(len(pd.read_csv(out/'report/model_comparison.csv')),8)
            # Wrong held-out checkpoint must never be accepted.
            pred['outer_fold']=1
            pred.to_csv(root/'bad.csv',index=False)
            with self.assertRaisesRegex(ValueError,'OOF folds'):
                load_oof(root/'bad.csv',master,'WD_probability')


if __name__=='__main__':unittest.main()
