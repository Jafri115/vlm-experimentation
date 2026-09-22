import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_timestamped_llm_dataset import assign_cue, canonical_role, infer_mapping, overlap_scores, timed_dialogue, timestamp, voxtral


class TimestampedDatasetTests(unittest.TestCase):
    def test_timestamp_rounding_carries_into_minutes(self):
        self.assertEqual(timestamp(59.96), '01:00.0')
        self.assertEqual(timestamp(3600.1), '60:00.1')
        self.assertEqual(timestamp(0), '00:00.0')
        with self.assertRaises(ValueError):
            timestamp(-1)

    def test_every_cue_keeps_its_own_timestamp(self):
        us = [{'start_sec': 62.3, 'speaker': 'T', 'text': 'Hallo.'},
              {'start_sec': 65.7, 'speaker': 'T', 'text': 'Wie geht es?'},
              {'start_sec': 66, 'speaker': 'P', 'text': 'Gut.'}]
        self.assertEqual(timed_dialogue(us), '[01:02.3] T: Hallo.\n[01:05.7] T: Wie geht es?\n[01:06.0] P: Gut.')

    def test_global_speaker_numbers_do_not_determine_roles(self):
        turns = [{'start': 0, 'end': 20, 'speaker': 'SPEAKER_01'}, {'start': 20, 'end': 40, 'speaker': 'SPEAKER_00'}]
        result = infer_mapping(turns, [(0, 20, 'T'), (20, 40, 'P')], 'amberscript_session_time_overlap')
        self.assertEqual(result['mapping'], {'SPEAKER_00': 'P', 'SPEAKER_01': 'T'})

    def test_manual_seed_is_rematched_in_new_namespace(self):
        turns = [{'start': 10, 'end': 20, 'speaker': 'NEW_B'}, {'start': 30, 'end': 40, 'speaker': 'NEW_A'}]
        result = infer_mapping(turns, [(11, 18, 'T')], 'manual_therapist_seed_time_overlap')
        self.assertEqual(result['mapping'], {'NEW_B': 'T', 'NEW_A': 'P'})

    def test_conflicting_or_one_sided_evidence_does_not_assign_roles(self):
        turns = [{'start': 0, 'end': 20, 'speaker': 'A'}, {'start': 20, 'end': 40, 'speaker': 'B'}]
        for evidence in [[(0, 40, 'T')], [(0, 40, 'T'), (0, 40, 'P')]]:
            self.assertEqual(infer_mapping(turns, evidence, 'test')['mapping'], {})

    def test_overlapping_duplicate_turns_do_not_double_evidence(self):
        turns = [{'start': 0, 'end': 10, 'speaker': 'A'}, {'start': 2, 'end': 8, 'speaker': 'A'}]
        self.assertEqual(overlap_scores(turns, [(0, 10, 'T')])['A']['T'], 10)

    def test_short_evidence_never_establishes_session_roles(self):
        turns = [{'start': 0, 'end': 1, 'speaker': 'A'}, {'start': 1, 'end': 2, 'speaker': 'B'}]
        self.assertEqual(infer_mapping(turns, [(0, 1, 'T'), (1, 2, 'P')], 'test')['mapping'], {})

    def test_mixed_speaker_cue_and_unassigned_roles_remain_unknown(self):
        session = {'turns': [{'start': 0, 'end': 5, 'speaker': 'A'}, {'start': 5, 'end': 10, 'speaker': 'B'}], 'role_mapping': {'A': 'T', 'B': 'P'}}
        self.assertEqual(assign_cue(0, 10, session)['speaker'], 'UNKNOWN')
        session['role_mapping'] = {}
        result = assign_cue(0, 5, session)
        self.assertEqual(result['speaker'], 'UNKNOWN')
        self.assertEqual(result['global_speaker'], 'A')

    def test_noncanonical_export_labels_are_not_guessed(self):
        self.assertEqual(canonical_role('Patientin'), 'P')
        self.assertIsNone(canonical_role('P2'))
        self.assertIsNone(canonical_role('I'))

    def test_voxtral_offset_cleaning_provider_and_role_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'example.json'
            path.write_text(json.dumps({'model': 'mistralai/Voxtral-Small-24B-2507', 'segments': [
                {'start': 2, 'end': 5, 'speaker': 'T', 'raw_speaker': 'SPEAKER_00', 'text': 'Ja<br> (lacht).'}]}), encoding='utf-8')
            session = {'turns': [{'start': 122, 'end': 125, 'speaker': 'A'}], 'role_mapping': {'A': 'P'}}
            result = voxtral({'start_sec': 120, 'end_sec': 180}, path, session)
            self.assertEqual(result['transcript_provider'], 'voxtral')
            self.assertTrue(result['llm_ready'])
            self.assertEqual(timed_dialogue(result['utterances']), '[02:02.0] P: Ja.')
            self.assertEqual(result['utterances'][0]['original_asr_role'], 'T')


if __name__ == '__main__':
    unittest.main()
