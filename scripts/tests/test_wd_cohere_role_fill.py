"""Behavioral tests for bounded role repair and timestamped transcript cues."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('builder',Path(__file__).resolve().parents[1]/'data/build_wd_cohere_available_subset.py')
builder=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def word(index,role,start=None,raw=None):
    start=float(index) if start is None else start
    return {'index':index,'text':f'word{index}','start':start,'end':start+.4,
            'speaker':role,'raw_speaker':raw}


class RoleFillTests(unittest.TestCase):
    def test_same_role_bracket_fills_without_mutating_source(self):
        original=[word(0,'P'),word(1,None),word(2,'P')]
        result,audit=builder.repair_unknown_roles(original)
        self.assertEqual(result[1]['speaker'],'P')
        self.assertIsNone(original[1]['speaker'])
        self.assertEqual(audit[0]['inferred_speaker'],'P')

    def test_cross_speaker_boundary_remains_unknown(self):
        result,audit=builder.repair_unknown_roles([word(0,'P'),word(1,None),word(2,'T')])
        self.assertIsNone(result[1]['speaker']); self.assertFalse(audit)

    def test_long_unknown_run_is_not_filled(self):
        result,audit=builder.repair_unknown_roles([word(0,'P')]+[word(i,None) for i in range(1,5)]+[word(5,'P')])
        self.assertFalse(audit)

    def test_untimed_and_raw_speaker_conflicts_are_barriers(self):
        for unknown in [dict(word(1,None),start=None),word(1,None,raw='B')]:
            result,audit=builder.repair_unknown_roles([word(0,'P',raw='A'),unknown,word(2,'P',raw='A')])
            self.assertFalse(audit)

    def test_gaps_missing_indices_and_edge_words_are_not_filled(self):
        for source in [[word(0,None),word(1,'P')],
                       [word(0,'P'),word(1,None,start=10),word(2,'P',start=11)],
                       [word(0,'P'),word(2,None),word(3,'P')]]:
            _,audit=builder.repair_unknown_roles(source)
            self.assertFalse(audit)

    def test_cues_keep_words_roles_and_absolute_time(self):
        words=[word(0,'P',start=625.1),word(1,'P',start=625.7),word(2,'T',start=632.2)]
        output=builder.format_words(words,'timestamped_cues')
        self.assertEqual(output,'[10:25.1] P: word0 word1\n[10:32.2] T: word2')
        self.assertEqual(builder.timestamp(59.99),'01:00.0')

    def test_contextual_sentence_tail(self):
        original=[dict(word(0,'T',start=0),text='suizidale'),
                  dict(word(1,None,start=.5),text='Phase.',end=3.4),
                  dict(word(2,'P',start=3.8),text='Ich')]
        result,audit,review=builder.repair_contextual_roles(original)
        self.assertEqual(result[1]['speaker'],'T')
        self.assertEqual(audit[0]['method'],'previous_sentence_tail_v2')
        self.assertFalse(audit[0]['confidence_calibrated'])

    def test_contextual_sentence_prefix(self):
        original=[dict(word(0,'P',start=0),text='Ende.'),
                  dict(word(1,None,start=3),text='Und'),
                  dict(word(2,None,start=3.5),text='die'),
                  dict(word(3,'T',start=4),text='restlichen')]
        result,audit,review=builder.repair_contextual_roles(original)
        self.assertEqual(result[1]['speaker'],'T')
        self.assertEqual(audit[0]['method'],'next_sentence_prefix_v2')

    def test_contextual_acknowledgment_not_assigned(self):
        result,audit,review=builder.repair_contextual_roles(
            [word(0,'P'),dict(word(1,None),text='Okay.'),word(2,'T')])
        self.assertIsNone(result[1]['speaker']); self.assertFalse(audit)
        self.assertEqual(review[0]['reason'],'acknowledgment_requires_review')

    def test_contextual_overlap_and_raw_conflict_not_assigned(self):
        for unknown in [dict(word(1,None,start=.5),text='Ende.',raw_speaker='B'),
                        dict(word(1,None,start=.5),text='Ende.',end=2.5)]:
            original=[dict(word(0,'P',start=0,raw='A'),text='ein'),unknown,word(2,'T',start=1.5)]
            result,audit,review=builder.repair_contextual_roles(original)
            self.assertIsNone(result[1]['speaker']); self.assertFalse(audit)


if __name__=='__main__': unittest.main()
