import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_llm_wd_aligned_dataset import load_two_rater_targets, segment_uid
from analysis.compare_vlm_llm_wd_predictions import exact_mcnemar
from llm.finetune_qwen3_8b_wd_text import prepare_rows


class LlmWdExperimentTests(unittest.TestCase):
    def test_uid_normalization(self):
        self.assertEqual(segment_uid('401001.0', 'S02', '3.0'), '401001_S2_seg003')

    def test_two_rater_targets_match_vlm_definitions(self):
        rows = [
            {'patient_id': '1', 'session_id': 'S1', 'segment_id': 1, 'coder': 'A', 'WD_P': 1},
            {'patient_id': '1', 'session_id': 'S1', 'segment_id': 1, 'coder': 'B', 'WD_P': 2},
            {'patient_id': '1', 'session_id': 'S1', 'segment_id': 2, 'coder': 'A', 'WD_P': 3},
            {'patient_id': '1', 'session_id': 'S1', 'segment_id': 2, 'coder': 'B', 'WD_P': 2},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'ratings.csv'; pd.DataFrame(rows).to_csv(path, index=False)
            targets = load_two_rater_targets(path, 2.0)
        disagreement = targets['1_S1_seg001']
        self.assertEqual((disagreement['WD_soft'], disagreement['WD_hard_mean'], disagreement['WD_consensus']), (0.5, 0, None))
        positive = targets['1_S1_seg002']
        self.assertEqual((positive['WD_soft'], positive['WD_hard_mean'], positive['WD_consensus']), (1.0, 1, 1))

    def test_modes_reuse_split_and_consensus_excludes_disagreement(self):
        rows = []
        for split, patient in [('train', '1'), ('val', '2'), ('test', '3')]:
            rows.append({'segment_uid': patient, 'patient_id': patient, 'split': split,
                         'WD_P_mean': 2.0, 'WD_soft': 1.0, 'WD_consensus': 1})
            rows.append({'segment_uid': patient+'x', 'patient_id': patient, 'split': split,
                         'WD_P_mean': 1.5, 'WD_soft': 0.5, 'WD_consensus': None})
        self.assertEqual(len(prepare_rows(rows, 'soft')), 6)
        self.assertEqual(len(prepare_rows(rows, 'consensus')), 3)

    def test_patient_leakage_is_rejected(self):
        rows = [
            {'patient_id': 'same', 'split': 'train', 'WD_P_mean': 1, 'WD_soft': 0, 'WD_consensus': 0},
            {'patient_id': 'same', 'split': 'test', 'WD_P_mean': 1, 'WD_soft': 0, 'WD_consensus': 0},
            {'patient_id': 'v', 'split': 'val', 'WD_P_mean': 1, 'WD_soft': 0, 'WD_consensus': 0},
        ]
        with self.assertRaises(ValueError):
            prepare_rows(rows, 'consensus')

    def test_exact_mcnemar_is_symmetric(self):
        self.assertEqual(exact_mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar(2, 8), exact_mcnemar(8, 2))


if __name__ == '__main__':
    unittest.main()
