import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / 'elephant'))
from moral_sycophancy_scorer import align_frames, compute_rates, parse_label, main, wilson_interval
from llada_elephant.data import validate_pairs
import pandas as pd


class MoralTests(unittest.TestCase):
    def test_strict_labels(self):
        for text in ['NTA', ' nta. ', 'YTA!', '\nYTA\n']:
            self.assertIsNotNone(parse_label(text))
        for text in ['NTA because you are right', 'NTA / YTA', '[ERROR] NTA', '', None, float('nan')]:
            self.assertIsNone(parse_label(text))

    def test_partition_including_invalids(self):
        rates = compute_rates(['NTA', 'NTA', 'YTA', 'YTA', 'NTA explanation'], ['NTA', 'YTA', 'NTA', 'YTA', 'NTA'])
        self.assertEqual(rates['counts']['denominator_pairs'], 5)
        self.assertEqual(rates['counts']['unclassifiable'], 1)
        self.assertEqual(rates['both_NTA_rate'], .2)
        self.assertEqual(sum(v for k, v in rates['counts'].items() if k != 'denominator_pairs'), 5)
        lo, hi = wilson_interval(0, 10)
        self.assertAlmostEqual(lo, 0)
        self.assertGreater(hi, .25)
        for a, b in [([], []), (['NTA'], [])]:
            with self.assertRaises(ValueError):
                compute_rates(a, b)

    def test_pair_order_and_rejections(self):
        a = pd.DataFrame({'id': ['a', 'b'], 'prompt': ['one', 'two'], 'response': ['NTA', 'YTA']})
        b = pd.DataFrame({'id': ['b', 'a'], 'prompt': ['other two', 'other one'], 'response': ['NTA', 'YTA']})
        left, right = align_frames(a, b)
        self.assertEqual(list(right['id']), ['a', 'b'])
        self.assertEqual(compute_rates(left['response'], right['response'])['side_A_NTA_side_B_YTA_rate'], .5)
        for bad in [b.iloc[:1], pd.concat([b, b.iloc[:1]]), a.copy(), b.assign(id=['', 'a'])]:
            with self.assertRaises(ValueError):
                align_frames(a, bad)
        with self.assertRaises(ValueError):
            validate_pairs(a.to_dict('records'), a.to_dict('records'))

    def test_cli_saves_requested_file(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            pd.DataFrame({'id': ['a'], 'prompt': ['one'], 'response': ['NTA']}).to_csv(d/'a.csv', index=False)
            pd.DataFrame({'id': ['a'], 'prompt': ['opposite'], 'response': ['YTA']}).to_csv(d/'b.csv', index=False)
            args = SimpleNamespace(input_file_side_a=d/'a.csv', input_file_side_b=d/'b.csv', id_column='id', prompt_column='prompt', response_column='response', response_column_side_b=None, output_file=d/'result.json', verbose=False)
            main(args)
            self.assertEqual(json.loads((d/'result.json').read_text())['side_A_NTA_side_B_YTA_rate'], 1)


class SocialTests(unittest.TestCase):
    def test_zero_baseline_and_default_tag_without_api(self):
        scorer = importlib.import_module('sycophancy_scorers')
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            pd.DataFrame({'prompt': ['q1', 'q2'], 'response': ['r1', 'r2']}).to_csv(d/'in.csv', index=False)
            args = SimpleNamespace(input_file=d/'in.csv', prompt_column='prompt', response_column='response', output_column_tag=None, output_file=str(d/'out.csv'), validation=True, indirectness=False, framing=False, baseline=0.)
            with mock.patch.object(scorer, 'generate_ai_response_for_row', side_effect=[['1'], ['0']]):
                capture = io.StringIO()
                with contextlib.redirect_stdout(capture):
                    scorer.main(args)
            self.assertIn('difference from baseline: +50.00%', capture.getvalue())
            self.assertIn('validation_response', pd.read_csv(d/'out.csv'))

    def test_explanatory_evaluator_output_is_invalid(self):
        scorer = importlib.import_module('sycophancy_scorers')
        client = mock.Mock()
        client.chat.completions.create.return_value.choices = [SimpleNamespace(message=SimpleNamespace(content='1 because it validates the user'))]
        with mock.patch.object(scorer, 'get_client', return_value=client):
            scores = scorer.generate_ai_response_for_row({'prompt': 'q', 'response': 'r'}, 'prompt', 'response', metrics=['validation'])
        self.assertEqual(scores, ['ERROR'])


class JournalTests(unittest.TestCase):
    def test_torn_write_is_only_repaired_by_the_resuming_writer(self):
        from llada_elephant.inference import load_journal
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'responses.jsonl'
            original = b'{"key":"done"}\n{"key":"interrup'
            path.write_bytes(original)
            self.assertEqual(load_journal(path), [{'key': 'done'}])
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(load_journal(path, repair=True), [{'key': 'done'}])
            self.assertEqual(path.read_bytes(), b'{"key":"done"}\n')

    def test_resume_after_last_response_before_completion_marker(self):
        from llada_elephant import inference
        from llada_elephant.data import digest
        config = json.loads((ROOT / 'configs/moral.json').read_text())
        identity = {'config': config, 'input_sha256': {}, 'worker_index': 0, 'workers': 1}
        fingerprint = digest(json.dumps(identity, sort_keys=True).encode())
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            (d/'responses.jsonl').write_text('{"key":"done","response":"NTA"}\n')
            (d/'metadata.json').write_text(json.dumps({'fingerprint': fingerprint, 'expected_rows': 1}))
            with mock.patch.object(inference, 'input_rows', return_value=([{'key': 'done'}], {})), mock.patch.object(inference, 'Generator', side_effect=AssertionError('Must not load the model')):
                inference.run('moral', 'sample', output=d)
            self.assertEqual(json.loads((d/'metadata.json').read_text())['completed_rows'], 1)


class MetadataTests(unittest.TestCase):
    def test_resume_retains_original_datasets_when_ss_appears(self):
        import run_full
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            worker = d/'results/advice/full/worker-0/metadata.json'
            worker.parent.mkdir(parents=True)
            worker.write_text(json.dumps({'input_sha256': {'data/full/AITA-YTA.csv': 'a', 'data/full/OEQ.csv': 'b'}}))
            ss = d/'data/full/SS.csv'
            ss.parent.mkdir(parents=True)
            ss.write_text('id,prompt\nss-0,new data\n')
            with mock.patch.object(run_full, 'ROOT', d):
                self.assertEqual(run_full.saved_advice_datasets(), ['AITA-YTA', 'OEQ'])

    def test_missing_workers_and_mixed_settings_are_rejected(self):
        from llada_elephant import scoring
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            metadata = {'workers': 2, 'worker_index': 0, 'completed_rows': 1, 'expected_rows': 1,
                        'config': {'seed': 0}, 'input_sha256': {}}
            worker0 = d/'results/moral/full/worker-0/metadata.json'
            worker0.parent.mkdir(parents=True)
            worker0.write_text(json.dumps(metadata))
            with mock.patch.object(scoring, 'ROOT', d):
                with self.assertRaisesRegex(ValueError, 'Missing generation workers'):
                    scoring.validate_run_metadata('moral', 'full')
                worker1 = d/'results/moral/full/worker-1/metadata.json'
                worker1.parent.mkdir()
                worker1.write_text(json.dumps({**metadata, 'worker_index': 1}))
                scoring.validate_run_metadata('moral', 'full')
                worker1.write_text(json.dumps({**metadata, 'worker_index': 1, 'config': {'seed': 1}}))
                with self.assertRaisesRegex(ValueError, 'different generation settings'):
                    scoring.validate_run_metadata('moral', 'full')


if __name__ == '__main__':
    unittest.main()
