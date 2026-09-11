import unittest
import csv
import tempfile
from pathlib import Path

from run_qwen3_8b_wd_3rs_zeroshot import LABELS, PROMPT, parse_generated
from compare_qwen3_8b_wd_to_ground_truth import evaluate_wd


class WithdrawalZeroShotTests(unittest.TestCase):
    def test_valid_integer_scores(self):
        for score in range(1, 6):
            self.assertEqual(parse_generated('{"wd_p_score":%d}' % score), score)

    def test_single_markdown_fence_is_tolerated(self):
        self.assertEqual(parse_generated('```json\n{"wd_p_score":2}\n```'), 2)

    def test_invalid_schema_or_score_is_rejected(self):
        invalid = [
            '{"wd_p_score":0}', '{"wd_p_score":6}', '{"wd_p_score":2.0}',
            '{"wd_p_score":true}', '{"wd_p_score":"2"}',
            '{"wd_p_score":2,"extra":1}', '{"primary_label":"WD_P"}', 'not json',
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises((ValueError, TypeError)):
                parse_generated(text)

    def test_prompt_matches_binary_target_and_manual_threshold(self):
        self.assertEqual(LABELS, ('NO_WD_P', 'WD_P'))
        self.assertIn('scores 2-5', PROMPT)
        self.assertIn('absent only for score 1', PROMPT)
        self.assertNotIn('CF_P', PROMPT)

    def test_binary_evaluator_uses_score_above_one(self):
        predictions = [
            {'segment_uid': 'a', 'status': 'OK', 'primary_label': 'WD_P',
             'wd_p_score': '2', 'transcript_provider': 'amberscript'},
            {'segment_uid': 'b', 'status': 'OK', 'primary_label': 'NO_WD_P',
             'wd_p_score': '1', 'transcript_provider': 'voxtral'},
        ]
        truth = [
            {'segment_uid': 'a', 'patient_id': '1', 'session_id': '1', 'segment_number': 1,
             'WD_P_binary_mean_gt1': 1, 'strict_wd_two_plus_rater_consensus': True,
             'strict_wd_binary': 1},
            {'segment_uid': 'b', 'patient_id': '1', 'session_id': '1', 'segment_number': 2,
             'WD_P_binary_mean_gt1': 0, 'strict_wd_two_plus_rater_consensus': True,
             'strict_wd_binary': 0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'predictions.csv'
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
                writer.writeheader()
                writer.writerows(predictions)
            summary = evaluate_wd(path, truth, root/'evaluation')
        self.assertEqual(summary['primary_mean_rating_metrics']['accuracy'], 1.0)
        self.assertEqual(summary['strict_wd_consensus_rows'], 2)


if __name__ == '__main__':
    unittest.main()
