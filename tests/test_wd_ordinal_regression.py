"""CPU checks for the ordinal WD_P target and report helpers."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from finetune_qwen3_8b_wd_text import (
    MANUAL_V2_SYSTEM_PROMPT,
    MANUAL_V3_SYSTEM_PROMPT,
    SYSTEM_PROMPTS,
    prepare_rows,
)
from report_wd_ordinal_regression import report


class OrdinalTargetTests(unittest.TestCase):
    def test_manual_rubric_has_ordinal_anchors_without_binary_rule(self):
        self.assertIs(SYSTEM_PROMPTS["manual_compact_v2"], MANUAL_V2_SYSTEM_PROMPT)
        for phrase in ("Shutting down", "Avoiding", "Masking experience",
                       "at least one clear marker", "1 =", "3 =", "5 ="):
            self.assertIn(phrase, MANUAL_V2_SYSTEM_PROMPT)
        self.assertNotIn("positive", MANUAL_V2_SYSTEM_PROMPT.lower())

    def test_v3_rubric_operationalizes_high_salience_without_marker_counting(self):
        self.assertIs(SYSTEM_PROMPTS["manual_detailed_v3"], MANUAL_V3_SYSTEM_PROMPT)
        for phrase in (
            "CLEARLY ELEVATED SALIENCE",
            "VERY SALIENT / DOMINANT",
            "Do not mechanically count markers or seconds",
            "A weak or ambiguous additional cue must NOT automatically raise a 3 to a 4",
            "Exact agreement about the narrow subtype is not required",
            "Speaker labels may contain errors",
        ):
            self.assertIn(phrase, MANUAL_V3_SYSTEM_PROMPT)

    def test_two_ratings_become_a_distribution(self):
        rows = []
        for patient, split in enumerate(("train", "val", "test"), 1):
            rows.append({"split": split, "patient_id": patient, "WD_P_mean": 2,
                         "WD_P_rater1": 1, "WD_P_rater2": 3})
        prepared = prepare_rows(rows, "ordinal")
        self.assertEqual(len(prepared), 3)
        self.assertEqual(prepared[0]["_target_distribution"], [.5, 0, .5, 0, 0])
        self.assertEqual(sum(prepared[0]["_target_distribution"]), 1)

    def test_exact_human_agreement_is_one_hot(self):
        rows = [{"split": split, "patient_id": index, "WD_P_mean": 4,
                 "WD_P_rater1": 4, "WD_P_rater2": 4}
                for index, split in enumerate(("train", "val", "test"), 1)]
        self.assertEqual(prepare_rows(rows, "ordinal")[0]["_target_distribution"], [0, 0, 0, 1, 0])

    def test_report_validates_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); master_root = root / "master"; master_root.mkdir()
            rows = []
            for patient in range(10):
                for segment in range(2):
                    rating = 1 + ((patient + segment) % 5)
                    rows.append({"sample_id": f"p{patient}s{segment}", "segment_uid": f"u{patient}_{segment}",
                                 "patient_id": patient, "session_id": 1,
                                 "WD_P_rater1": rating, "WD_P_rater2": rating})
            master = pd.DataFrame(rows)
            master.to_csv(master_root / "paired_master_soft.csv", index=False)
            oof = []
            for fold in range(1, 6):
                frame = master.copy()
                frame["split"] = ["test" if patient // 2 == fold-1 else
                                  "val" if patient // 2 == fold % 5 else "train"
                                  for patient in frame.patient_id]
                folder = master_root / f"fold_{fold}"; folder.mkdir()
                frame.to_csv(folder / "master_manifest.csv", index=False)
                test = frame[frame.split == "test"][["sample_id", "segment_uid"]].copy()
                test["outer_fold"] = fold
                test["WD_prediction"] = [1 + (i % 5) for i in range(len(test))]
                oof.append(test)
            prediction = pd.concat(oof, ignore_index=True)
            qwen8 = root / "qwen8.csv"; qwen14 = root / "qwen14.csv"
            prediction.to_csv(qwen8, index=False); prediction.to_csv(qwen14, index=False)
            args = type("Args", (), {"master_root": master_root, "qwen8": qwen8, "qwen14": qwen14,
                                      "extra": [], "output": root / "report", "bootstrap": 10,
                                      "seed": 42, "no_plots": True})()
            report(args)
            self.assertTrue((args.output / "ordinal_regression_report.md").exists())
            self.assertEqual(len(pd.read_csv(args.output / "ordinal_regression_metrics.csv")), 3)


if __name__ == "__main__":
    unittest.main()
