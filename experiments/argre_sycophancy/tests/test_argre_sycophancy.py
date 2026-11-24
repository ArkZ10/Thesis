"""Unit tests for experiments/argre_sycophancy (CPU-only, no model loading)."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import calibrate_answers
import collect_answers
import run as argre_run


class ArgreSycophancyTests(unittest.TestCase):
    def test_vendored_argre_imports_and_paths_exist(self):
        self.assertTrue(argre_run.ARGRE.exists(), f"Missing ARGRE dir: {argre_run.ARGRE}")
        self.assertTrue(argre_run.BROKENMATH.exists(), f"Missing BrokenMath results: {argre_run.BROKENMATH}")
        self.assertTrue(collect_answers.DATASET.exists(), f"Missing BrokenMath sample.json: {collect_answers.DATASET}")
        self.assertTrue(callable(argre_run.PCA))
        self.assertTrue(callable(argre_run.RMTrainer))

    def test_load_pairs_splits_cleanly(self):
        pairs, excluded = argre_run.load_pairs()
        self.assertEqual(len(pairs), 18)
        splits = {row["split"] for row in pairs}
        self.assertEqual(splits, {"train", "heldout"})
        self.assertEqual(sum(row["split"] == "heldout" for row in pairs), 4)
        self.assertEqual(sum(row["split"] == "train" for row in pairs), 14)
        self.assertGreater(len(excluded), 0)
        for row in pairs:
            self.assertEqual(row["original_judge"], "incorrect")
            self.assertTrue(row["chosen"].strip())
            self.assertTrue(row["rejected"].strip())

    def test_padded_pairs_shapes_and_masks(self):
        samples = [
            (torch.ones(3, 8), torch.full((5, 8), 2.0)),
            (torch.full((2, 8), 3.0), torch.full((4, 8), 4.0)),
        ]
        packed = argre_run.padded_pairs(samples)
        self.assertEqual(len(packed), 2)
        self.assertEqual(tuple(packed[0]["hidden_states"].shape), (2, 5, 8))
        self.assertEqual(int(packed[0]["attention_mask"][0].sum()), 3)
        self.assertEqual(int(packed[0]["attention_mask"][1].sum()), 5)
        self.assertEqual(int(packed[1]["attention_mask"][0].sum()), 2)
        self.assertEqual(int(packed[1]["attention_mask"][1].sum()), 4)

    def test_calibrate_metrics_counts_and_balanced_accuracy(self):
        rows = [
            {"label": calibrate_answers.ACCEPTS, "reward": -2.0},
            {"label": calibrate_answers.ACCEPTS, "reward": 1.0},
            {"label": calibrate_answers.CHALLENGES, "reward": 2.0},
            {"label": calibrate_answers.CHALLENGES, "reward": 3.0},
        ]
        stats = calibrate_answers.metrics(rows, cutoff=0.0)
        self.assertEqual(stats["accepts_flagged"], 1)
        self.assertEqual(stats["accepts_missed"], 1)
        self.assertEqual(stats["challenges_flagged"], 0)
        self.assertEqual(stats["challenges_passed"], 2)
        self.assertAlmostEqual(stats["sensitivity"], 0.5)
        self.assertAlmostEqual(stats["specificity"], 1.0)
        self.assertAlmostEqual(stats["balanced_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
