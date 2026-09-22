import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.compare_qwen3_8b_to_ground_truth import binary_metrics, multiclass_metrics, type_label


class GroundTruthComparisonTests(unittest.TestCase):
    def test_type_labels(self):
        self.assertEqual(type_label(0, 0), 'NO_RUPTURE')
        self.assertEqual(type_label(1, 0), 'WD_P')
        self.assertEqual(type_label(0, 1), 'CF_P')
        self.assertEqual(type_label(1, 1), 'MIXED_P')

    def test_binary_metrics(self):
        rows = [{'truth': t, 'pred': p} for t, p in [(1, 1), (0, 0), (0, 1), (1, 0)]]
        result = binary_metrics(rows, 'truth', 'pred')
        self.assertEqual((result['TP'], result['TN'], result['FP'], result['FN']), (1, 1, 1, 1))
        self.assertEqual(result['balanced_accuracy'], 0.5)

    def test_multiclass_absent_class_remains_in_macro_average(self):
        rows = [{'truth': 'NO_RUPTURE', 'pred': 'NO_RUPTURE'}, {'truth': 'WD_P', 'pred': 'NO_RUPTURE'}]
        summary, per_class, matrix = multiclass_metrics(rows, 'truth', 'pred')
        self.assertEqual(summary['accuracy'], 0.5)
        self.assertEqual(len(per_class), 4)
        self.assertEqual(len(matrix), 4)


if __name__ == '__main__':
    unittest.main()
