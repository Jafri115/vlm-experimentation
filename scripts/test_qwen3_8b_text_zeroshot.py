import unittest

from run_qwen3_8b_text_zeroshot import parse_label, select_rows


class ZeroShotTests(unittest.TestCase):
    def test_scope_filters_preserve_review_rows_only_when_requested(self):
        rows = [dict(segment_uid=str(i), transcript_text=text, in_audio_inventory=inv,
                     llm_ready=ready, transcript_status='READY' if ready else 'REVIEW')
                for i, (text, inv, ready) in enumerate([('text', True, True), ('text', True, False),
                                                      ('text', False, True), ('', True, False)])]
        self.assertEqual([r['segment_uid'] for r in select_rows(rows, 'inventory-ready')[0]], ['0'])
        self.assertEqual([r['segment_uid'] for r in select_rows(rows, 'all-ready')[0]], ['0', '2'])
        self.assertEqual([r['segment_uid'] for r in select_rows(rows, 'inventory-nonempty')[0]], ['0', '1'])

    def test_duplicate_segment_ids_are_rejected(self):
        row = dict(segment_uid='same', transcript_text='text', in_audio_inventory=True, llm_ready=True, transcript_status='READY')
        with self.assertRaises(ValueError):
            select_rows([row, row], 'inventory-ready')

    def test_valid_labels(self):
        for label in ('NO_RUPTURE', 'WD_P', 'CF_P', 'MIXED_P'):
            self.assertEqual(parse_label('{"primary_label":"'+label+'"}', 'stop'), label)

    def test_invalid_or_truncated_output_is_not_a_negative_prediction(self):
        for text, finish in [('{"primary_label":"INVALID"}', 'stop'), ('{}', 'stop'),
                             ('{"primary_label":"NO_RUPTURE"}', 'length'),
                             ('{"primary_label":"WD_P","extra":1}', 'stop'), ('not JSON', 'stop')]:
            with self.assertRaises(ValueError):
                parse_label(text, finish)


if __name__ == '__main__':
    unittest.main()
