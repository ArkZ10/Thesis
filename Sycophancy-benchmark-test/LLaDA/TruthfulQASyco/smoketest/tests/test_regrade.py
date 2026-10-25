import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import regrade_smoke as regrade


class RegradeTests(unittest.TestCase):
    def test_regrade_copies_identical_responses_and_preserves_generation_provenance(self):
        source = ROOT / 'runs/20261017T194758_100249Z'
        checksums = regrade.source_checksums(source)
        with TemporaryDirectory() as directory, patch.object(regrade.smoke, 'ROOT', Path(directory)):
            run, rows, manifest = regrade.initialize_regrade(source)
            original = json.loads((source / 'manifest.json').read_text())
            self.assertEqual(len(rows), 100)
            self.assertEqual((run / 'responses.jsonl').read_bytes(), (source / 'responses.jsonl').read_bytes())
            self.assertEqual(manifest['generation_identity'], original['generation_identity'])
            self.assertEqual(manifest['generation_config_sha256'], original['generation_config_sha256'])
            self.assertEqual(manifest['judge_config']['rubric_version'], 3)
            self.assertFalse((run / 'judgments.jsonl').exists())
            self.assertEqual(regrade.initialize_regrade(source, run)[0], run)
            self.assertEqual(regrade.source_checksums(source), checksums)
            (run / 'responses.jsonl').write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'Copied responses changed'):
                regrade.initialize_regrade(source, run)

    def test_resume_refuses_a_modified_saved_prompt(self):
        source = ROOT / 'runs/20261017T194758_100249Z'
        with TemporaryDirectory() as directory, patch.object(regrade.smoke, 'ROOT', Path(directory)):
            run, _, _ = regrade.initialize_regrade(source)
            (run / 'judge_prompt.txt').write_text('Different rubric')
            with self.assertRaisesRegex(ValueError, 'Saved judge prompt changed'):
                regrade.initialize_regrade(source, run)


if __name__ == '__main__':
    unittest.main()
