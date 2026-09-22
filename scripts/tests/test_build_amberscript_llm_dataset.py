"""Regression checks for transcript cleaning and temporal assignment."""
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_amberscript_llm_dataset import (
    align_row, dialogue_text, normalize_space, read_source,
    remove_parentheses, session_key, strip_markup,
)


class TranscriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "segments").mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def srt(self, text):
        path = self.root / "401001_S1.srt"
        path.write_text(text, encoding="utf-8")
        return read_source(path)

    def row(self, number):
        return {"segment_uid": f"401001_S1_seg{number:03d}", "patient_id": "401001", "session_id": "1", "segment_number": number, "start_sec": 60*(number-1), "end_sec": 60*number}

    def test_markup_entities_and_nested_annotations(self):
        text = strip_markup('P: Sch&ouml;n&lt;br /&gt; heute (lacht (leise)). <b>Ja</b> <br')
        cleaned, _ = remove_parentheses(text)
        self.assertEqual(normalize_space(cleaned), 'P: Schön heute. Ja')

    def test_parentheses_cross_cues_and_speaker_carries(self):
        source = self.srt('1\n00:00:00,000 --> 00:00:05,000\nT: Guten Tag (unv.\n\n2\n00:00:05,000 --> 00:00:09,000\nweiter) heute.\n\n3\n00:00:09,000 --> 00:00:12,000\nP: Ja.\n')
        self.assertEqual(dialogue_text(source['utterances']), 'T: Guten Tag heute.\nP: Ja.')

    def test_unmatched_open_does_not_destroy_remaining_session(self):
        cleaned, stats = remove_parentheses('Hallo (unfertig\nP: Guten Tag')
        self.assertIn('P: Guten Tag', cleaned)
        self.assertEqual(stats['unmatched_open_parentheses'], 1)

    def test_malformed_editorial_markers_are_not_model_input(self):
        source = self.srt('1\n00:00:00,000 --> 00:00:05,000\n- #00:00:05-0, Video beginnt)\n\n2\n00:00:05,000 --> 00:00:09,000\nP: Ja #00:00:08#.\n')
        self.assertEqual(dialogue_text(source['utterances']), 'P: Ja.')

    def test_midpoint_assigns_crossing_cue_once_and_point_at_boundary(self):
        source = self.srt('1\n00:00:58,000 --> 00:01:02,000\nP: Genau.\n\n2\n00:01:00,000 --> 00:01:00,000\nT: Ja.\n')
        first, u1 = align_row(self.row(1), source, 1, self.root)
        second, u2 = align_row(self.row(2), source, 2, self.root)
        self.assertFalse(u1)
        self.assertEqual(len(u2), 2)
        self.assertEqual(second['boundary_crossing_cues'], 1)
        self.assertEqual(second['transcript_status'], 'READY')

    def test_reversed_timestamp_flags_only_affected_windows(self):
        source = self.srt('1\n00:00:00,000 --> 00:00:05,000\nP: Hallo.\n\n2\n00:01:10,000 --> 00:01:05,000\nT: Ja.\n\n3\n00:02:05,000 --> 00:02:10,000\nP: Danke.\n')
        statuses = [align_row(self.row(i), source, i, self.root)[0]['transcript_status'] for i in (1, 2, 3)]
        self.assertEqual(statuses, ['READY', 'REVIEW_TIMING', 'READY'])

    def test_docx_uses_endpoint_timestamps_and_excludes_editorial_text(self):
        path = self.root / '401001_S1.docx'
        paragraphs = ['401001_S1.mp4', 'CAVE: (#00:00:00-0#, Video beginnt) #00:00:02-3#', 'T: Hallo (unv. #00:00:03-0#). #00:00:05-4#', 'P: Ja. #00:00:05-4#']
        xml = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + ''.join(f'<w:p><w:r><w:t>{p}</w:t></w:r></w:p>' for p in paragraphs) + '</w:body></w:document>'
        with ZipFile(path, 'w') as archive:
            archive.writestr('word/document.xml', xml)
        source = read_source(path)
        self.assertEqual(source['utterances'][0]['start_sec'], 2.3)
        self.assertEqual(source['utterances'][0]['end_sec'], 5.4)
        self.assertEqual(dialogue_text(source['utterances']), 'T: Hallo.\nP: Ja.')

    def test_html_disguised_as_docx(self):
        path = self.root / '401001_S1.docx'
        path.write_text('<html><body><p><small>00:00:02</small><br /><i>T</i>: Hallo (lacht).</p><p><small>00:00:08</small><br /><i>P</i>: Ja.</p></body></html>', encoding='utf-8')
        source = read_source(path)
        self.assertEqual(source['utterances'][0]['start_sec'], 2)
        self.assertEqual(source['utterances'][0]['end_sec'], 8)
        self.assertEqual(dialogue_text(source['utterances']), 'T: Hallo.\nP: Ja.')

    def test_missing_transcript_is_explicit(self):
        row, utterances = align_row(self.row(1), None, 1, self.root)
        self.assertEqual(row['transcript_status'], 'MISSING_TRANSCRIPT')
        self.assertFalse(row['llm_ready'])
        self.assertFalse(row['role_mapping_is_ground_truth'])
        self.assertFalse(utterances)

    def test_filename_variants_and_no_guessing_141_as_14(self):
        self.assertEqual(session_key('401027_13_converted.docx'), ('401027', '13'))
        self.assertEqual(session_key('401019_S141_transcript.srt'), ('401019', '141'))


if __name__ == '__main__':
    unittest.main()
