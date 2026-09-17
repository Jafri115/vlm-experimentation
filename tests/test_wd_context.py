"""CPU tests for context boundaries and target-only pooling."""
import sys
import unittest
import tempfile
import numpy as np
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wd_context_inputs import build_context,encode_row,TARGET_HEADER


class CharacterTokenizer:
    def apply_chat_template(self,messages,**kwargs):
        return 'SYSTEM:'+messages[0]['content']+'\nUSER:'+messages[1]['content']+'\nEND'
    def __call__(self,text,**kwargs):
        return {'input_ids':list(range(len(text))),'attention_mask':[1]*len(text),
                'offset_mapping':[(i,i+1) for i in range(len(text))]}


class ContextTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'segment_uid':['a','b','c','d'],'patient_id':[1]*4,
            'session_id':[1,1,1,2],'split':['train']*4,'start_sec':[0,60,180,0],
            'end_sec':[60,120,240,60],'WD_consensus':[0,1,0,1],
            'transcript_text':['[00:00.0] P: previous','[00:00.0] T: question\n[00:02.0] P: target\ncontinued',
                               '[00:00.0] T: no patient','[00:00.0] P: session two']})

    def test_boundaries_and_preserved_targets(self):
        f=self.frame();d=build_context(f,True)
        self.assertEqual(d.context_used.tolist(),[False,True,False,False])
        self.assertEqual(d.WD_consensus.tolist(),f.WD_consensus.tolist())
        self.assertIn('previous',d.iloc[1].transcript_text)
        self.assertNotIn('previous',build_context(f,False).iloc[1].transcript_text)
        self.assertEqual(d.iloc[2].transcript_text,TARGET_HEADER+f.iloc[2].transcript_text)
        f.loc[1,'split']='test'
        with self.assertRaisesRegex(ValueError,'crosses'):build_context(f,True)

    def test_target_patient_mask_and_fallback(self):
        d=build_context(self.frame(),True);tok=CharacterTokenizer()
        for idx,expected in [(1,'targetcontinued'),(2,d.iloc[2].target_text)]:
            row=d.iloc[idx].to_dict();enc=encode_row(tok,row,'instruction',10000,'target_patient')
            text=tok.apply_chat_template([{'content':'instruction'},{'content':row['transcript_text']}])
            chosen=''.join(ch for ch,m in zip(text,enc['pool_mask']) if m)
            self.assertEqual(chosen,expected)
            self.assertEqual(enc['pooling_fallback'],idx==2)
        with self.assertRaisesRegex(ValueError,'silently truncated'):encode_row(tok,row,'instruction',5)

    def test_overlap_and_ambiguous_time(self):
        f=self.frame();f.loc[1,'start_sec']=59
        self.assertFalse(build_context(f,True).iloc[1].context_used)
        f.loc[1,'start_sec']=0
        with self.assertRaisesRegex(ValueError,'Ambiguous'):build_context(f,True)

    def test_preparation_review_and_report(self):
        from prepare_wd_context_experiment import prepare,export_review
        from report_wd_context_experiment import report
        from finetune_qwen3_8b_wd_text import read_jsonl,prepare_rows
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);master=root/'master';master.mkdir();out=root/'out';source=root/'source'
            d=pd.DataFrame({'sample_id':[f's{i}' for i in range(20)],'segment_uid':[f'u{i}' for i in range(20)],
                'patient_id':np.repeat(range(10),2),'session_id':1,'start_sec':[0,60]*10,'end_sec':[60,120]*10,
                'WD_P_rater1':[1,3]*10,'WD_P_rater2':[1,3]*10,'WD_consensus':[0,1]*10,
                'WD_soft':[0,1]*10,'WD_P_mean':[1,3]*10,
                'transcript_text':['[00:00.0] P: engaged cooperative responsive','[00:00.0] P: avoidance withdrawal distant']*10})
            d.to_csv(master/'paired_master_soft.csv',index=False)
            for fold in range(1,6):
                folder=master/f'fold_{fold}';folder.mkdir();f=d.copy()
                f['split']=['test' if p//2==fold-1 else 'val' if p//2==fold%5 else 'train' for p in f.patient_id]
                f.to_csv(folder/'master_manifest.csv',index=False)
                val=f[f.split=='val'].copy();val['WD_probability']=.6
                s=source/f'fold_{fold}';s.mkdir(parents=True);val.to_csv(s/'val_predictions.csv',index=False)
            prepare(master,out);export_review(master,out,source)
            review=pd.read_csv(out/'validation_review/fold_1_review.csv')
            self.assertTrue(set(review.patient_id).issubset({2,3}))
            rows=prepare_rows(read_jsonl(out/'datasets/previous_context/fold_1/master_manifest.jsonl'),'consensus')
            self.assertEqual(len(rows),20)
            self.assertIsInstance(rows[0]['target_patient_spans'],list)
            for variant in ['target_only','previous_context']:
                destination=out/'llm'/variant;destination.mkdir(parents=True)
                pred=pd.read_csv(out/'tfidf'/variant/'oof_predictions.csv')
                pred.to_csv(destination/'oof_predictions.csv',index=False)
            report(master,out)
            self.assertEqual(len(pd.read_csv(out/'metrics.csv')),4)
            self.assertEqual(len(pd.read_csv(out/'comparisons.csv')),3)
            self.assertTrue((out/'comparison.png').exists())


if __name__=='__main__':unittest.main()
