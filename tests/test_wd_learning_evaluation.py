"""CPU checks for the WD learning audit: python -m unittest discover -s tests -p test_wd_learning_evaluation.py"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_wd_learning import baseline_rows, ordinal_metrics, read_master, bootstrap_gain


class LearningEvaluationTests(unittest.TestCase):
    def test_perfect_and_constant_predictions(self):
        d = pd.DataFrame({'patient_id': [1, 1, 2, 2], 'h1': [1, 2, 3, 4],
                          'h2': [1, 2, 3, 4], 'target': [1, 2, 3, 4]})
        perfect = ordinal_metrics(d, d.target)
        self.assertEqual(perfect['R2_vs_human_mean'], 1)
        self.assertEqual(perfect['AI_human_mean_ICC_A1'], 1)
        self.assertEqual(perfect['rounded_match_both'], 1)
        constant = ordinal_metrics(d, [2.5]*4)
        self.assertEqual(constant['R2_vs_human_mean'], 0)
        self.assertEqual(constant['prediction_sd'], 0)
        self.assertTrue(np.isnan(constant['Spearman_vs_human_mean']))

    def test_cluster_gain(self):
        d = pd.DataFrame({'patient_id': [1, 1, 2]})
        result = bootstrap_gain(d, np.ones(3), 50, np.random.default_rng(42))
        self.assertEqual(result['gain'], 1)
        self.assertEqual(result['ci_low'], 1)
        self.assertEqual(result['ci_high'], 1)

    def test_training_only_baseline_and_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = pd.DataFrame({'sample_id': ['a', 'b', 'c'], 'patient_id': [1, 2, 3],
                                  'WD_P_rater1': [1, 3, 5], 'WD_P_rater2': [1, 3, 5]})
            frame.to_csv(root/'labels.csv', index=False)
            labels = read_master(root/'labels.csv')
            for f in range(3):
                folder = root/f'fold_{f+1}'
                folder.mkdir()
                fold = frame.copy()
                fold['split'] = ['test' if i == f else 'train' if i == (f+1)%3 else 'val' for i in range(3)]
                fold.to_csv(folder/'master_manifest.csv', index=False)
            b, _ = baseline_rows(root, labels, 'regression')
            self.assertEqual(b.set_index('sample_id').loc['a', 'train_mean'], 3)
            # Same patient appears in train and test: must fail before scoring.
            path = root/'fold_1/master_manifest.csv'
            bad = pd.read_csv(path)
            bad.loc[1, 'patient_id'] = 1
            bad.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'patient leakage'):
                baseline_rows(root, labels, 'regression')


if __name__ == '__main__':
    unittest.main()
