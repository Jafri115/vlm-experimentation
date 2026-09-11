import argparse
import json
import tempfile
import unittest
from pathlib import Path

from complete_llm_dataset_from_memopsy import candidate, resolve_role, union_duration


class FullSessionRoleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / '401001_S1_seg002_000060-000120s.json'
        self.args = argparse.Namespace(min_coverage=0.5, min_purity=0.8)
        self.row = {'segment_uid': '401001_S1_seg002', 'start_sec': '60', 'end_sec': '120', 'therapist_id': 'Example therapist'}
        self.timeline = {'turns': [{'start': 61, 'end': 64, 'role': 'P'}], 'source': 'session.json',
                         'sha256': 'test', 'role_source': 'profiles.csv', 'role_source_sha256': 'test',
                         'role_status': 'PROVISIONAL', 'therapist_id': 'Example therapist', 'provenance': 'test'}

    def tearDown(self):
        self.temp.cleanup()

    def run_candidate(self, segments=None, timeline=True, review=None, metadata=None):
        if segments is None:
            segments = [{'start': 1, 'end': 4, 'speaker': 'T', 'raw_speaker': 'SPEAKER_01', 'text': 'Hallo <br> ja (lacht).'}]
        self.path.write_text(json.dumps({'segments': segments, 'transcription_metadata': metadata or {}}), encoding='utf-8')
        return candidate(self.row, self.path, self.timeline if timeline else None, review, self.args)

    def test_absolute_offset_overrides_incorrect_local_role(self):
        result = self.run_candidate()
        self.assertEqual(result['memopsy_status'], 'READY_FULL_SESSION_ROLES')
        self.assertEqual(result['transcript_text'], 'P: Hallo ja.')
        self.assertEqual(result['changed_role_utterances'], 1)
        self.assertEqual(result['utterances'][0]['start_sec'], 61)
        self.assertEqual(result['utterances'][0]['original_asr_raw_speaker'], 'SPEAKER_01')

    def test_no_full_session_timeline_never_reuses_local_labels(self):
        result = self.run_candidate(timeline=False)
        self.assertEqual(result['memopsy_status'], 'MISSING_FULL_SESSION_DIARIZATION')
        self.assertEqual(result['utterances'][0]['speaker'], 'UNKNOWN')

    def test_ambiguous_simultaneous_speakers_remain_unknown(self):
        turns = [{'start': 1, 'end': 4, 'role': role} for role in ['T', 'P']]
        self.assertEqual(resolve_role(1, 4, turns)['speaker'], 'UNKNOWN')

    def test_small_overlap_does_not_assign_whole_utterance(self):
        result = resolve_role(0, 10, [{'start': 1, 'end': 2, 'role': 'P'}])
        self.assertEqual(result['speaker'], 'UNKNOWN')
        self.assertEqual(result['role_coverage'], 0.1)

    def test_duplicate_turns_do_not_inflate_coverage(self):
        self.assertEqual(union_duration([(1, 4), (2, 3), (3, 5)]), 4)
        result = resolve_role(0, 10, [{'start': 1, 'end': 4, 'role': 'P'}]*4)
        self.assertEqual(result['role_coverage'], 0.3)
        self.assertEqual(result['speaker'], 'UNKNOWN')

    def test_manual_disagreement_blocks_promotion(self):
        for decision in ('False', 'Partial'):
            result = self.run_candidate(review={'global_result_supported': decision})
            self.assertEqual(result['memopsy_status'], 'MANUAL_REVIEW_DISPUTES_GLOBAL_DIARIZATION')

    def test_invalid_segment_local_interval_blocks_promotion(self):
        result = self.run_candidate(segments=[{'start': 61, 'end': 64, 'text': 'Hallo', 'speaker': 'T'}])
        self.assertEqual(result['memopsy_status'], 'INVALID_ASR_TIMESTAMPS')

    def test_flagged_asr_and_empty_text_are_excluded(self):
        self.assertEqual(self.run_candidate(metadata={'prediction_anomalous': True})['memopsy_status'], 'ASR_PREDICTION_ANOMALOUS')
        self.assertEqual(self.run_candidate(segments=[])['memopsy_status'], 'NO_TRANSCRIPT_TEXT')

    def test_missing_one_role_assignment_blocks_entire_candidate(self):
        result = self.run_candidate(segments=[{'start': 1, 'end': 4, 'text': 'Ja'}, {'start': 20, 'end': 21, 'text': 'Nein'}])
        self.assertEqual(result['unresolved_utterances'], 1)
        self.assertEqual(result['memopsy_status'], 'UNRESOLVED_UTTERANCE_ROLES')

    def test_role_namespace_metadata_mismatch_blocks_promotion(self):
        self.timeline['therapist_id'] = 'Different therapist'
        self.assertEqual(self.run_candidate()['memopsy_status'], 'SESSION_THERAPIST_METADATA_MISMATCH')


if __name__ == '__main__':
    unittest.main()
