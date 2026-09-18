"""Validation tests for the small manual patient-side review."""
import json,sys,tempfile,unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wd_patient_side_review import apply

class ReviewTests(unittest.TestCase):
    def test_apply_backs_up_and_updates_only_reviewed_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cohort=root/'cohort';cohort.mkdir()
            pd.DataFrame([{'patient_id':'10','video':'10_S1.mp4'}]).to_csv(cohort/'excluded_patient_side.csv',index=False)
            pd.DataFrame([{'patient_id':'10','video':'10_S1.mp4','video_path':'x.mp4','video_found':True}]).to_csv(cohort/'video_role_audit.csv',index=False)
            decision=root/'decisions.json';decision.write_text(json.dumps({'format':'wd_patient_side_decisions_v1','created_at':'now','decisions':[{'patient_id':'10','video':'10_S1.mp4','patient_side':'left'}]}))
            cache=root/'cache.json';cache.write_text('{}')
            class A:pass
            args=A();args.cohort_root=cohort;args.decisions=decision;args.role_cache=cache
            apply(args);updated=json.loads(cache.read_text())
            self.assertEqual(updated['10_S1.mp4']['patient_side'],'left')
            self.assertEqual(len(list(root.glob('cache.backup_*.json'))),1)

if __name__=='__main__':unittest.main()
