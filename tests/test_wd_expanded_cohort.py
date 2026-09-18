"""CPU-only checks for expanded cohort target and transcript construction."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_wd_expanded_multimodal_cohort import build_targets,attach_strict_transcripts,freeze_folds


class ExpandedCohortTests(unittest.TestCase):
    def test_strict_candidate_and_shared_folds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);labels=[]
            for patient in range(10):
                for segment in [1,2]:
                    for coder in ['A','B']:
                        labels.append({'video':f'{patient}_S1.mp4','patient_id':patient,'session_id':'S1',
                            'coder':coder,'segment_id':segment,'segment_start':(segment-1)*60,
                            'segment_end':segment*60-1,'WD_P':1 if segment==1 else 3})
            label_path=root/'labels.csv';pd.DataFrame(labels).to_csv(label_path,index=False)
            targets=build_targets(label_path,2);self.assertEqual(len(targets),20)
            transcript_path=root/'text.jsonl'
            with transcript_path.open('w',encoding='utf-8') as f:
                for row in targets.itertuples():
                    ready=row.segment_id==1 or str(row.patient_id)!='0'
                    f.write(json.dumps({'segment_uid':row.segment_uid,'transcript_text':'P: text',
                        'llm_ready':ready,'transcript_provider':'test','transcript_status':'READY' if ready else 'REVIEW'})+'\n')
            candidate,excluded=attach_strict_transcripts(targets,transcript_path)
            self.assertEqual(len(candidate),19);self.assertEqual(excluded.reason.tolist(),['TRANSCRIPT_NOT_READY'])
            candidate['video_path']='video.mp4';candidate['patient_side']='left';candidate['visual_ready']=True
            candidate['paired_ready']=True;candidate['frame_cache_ready']=True
            folds=freeze_folds(candidate,root/'out',5,4,42)
            self.assertEqual(len(folds),5)
            tests=[]
            for fold in folds:tests.extend(fold['test_patients'])
            self.assertEqual(len(tests),len(set(tests)))


if __name__=='__main__':unittest.main()
