import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

from sycophancy.api import APIQuery
from sycophancy.configs import OverallConfig, SolverConfig
from sycophancy.llada import unmask, validate_schedule
from sycophancy.postprocess import process_sample
from sycophancy.run_manifest import ensure_manifest, select_files

ROOT = Path(__file__).resolve().parents[1]


class DummyModel:
    def __init__(self, length, predicted=3, finite=True):
        self.length, self.predicted, self.finite = length, predicted, finite
        self.calls = 0

    def __call__(self, tokens):
        self.calls += 1
        logits = torch.zeros((1, self.length, 8))
        logits[..., self.predicted] = 10 if self.finite else float('nan')
        return SimpleNamespace(logits=logits)


class LLaDATests(unittest.TestCase):
    def test_masked_sampler_preserves_prompt_and_fills_all_positions(self):
        prompt = torch.tensor([[1, 2, 4]])
        model = DummyModel(8)
        tokens, end, steps = unmask(model, prompt, 8, 4, 4, 7, {6})
        self.assertEqual(tokens, [3] * 8)
        self.assertIsNone(end)
        self.assertEqual(steps, 4)
        self.assertEqual(prompt.tolist(), [[1, 2, 4]])

    def test_stop_at_end_in_finalized_block_without_generating_future_blocks(self):
        model = DummyModel(8, predicted=6)
        tokens, end, steps = unmask(model, torch.tensor([[1]]), 8, 8, 4, 7, {6})
        self.assertEqual(len(tokens), 4)
        self.assertEqual(end, 0)
        self.assertEqual(steps, 4)
        self.assertEqual(model.calls, 4)

    def test_nonfinite_outputs_and_invalid_schedules_raise(self):
        with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
            unmask(DummyModel(4, finite=False), torch.tensor([[1]]), 4, 4, 4, 7, {6})
        for schedule in ((0, 4, 4), (7, 4, 4), (8, 3, 4), (8, 16, 4)):
            with self.assertRaises(ValueError):
                validate_schedule(*schedule)

    def test_api_dispatch_needs_no_key_and_preserves_cost_metadata(self):
        expected = (0, 'False.', {'cost': 0, 'input_tokens': 3, 'output_tokens': 2,
                                  'generation': {'status': 'ok'}})
        with patch.dict(os.environ, {}, clear=True), patch('sycophancy.llada.LLaDAEngine') as engine:
            engine.return_value.run_queries.return_value = iter([expected])
            api = APIQuery(model='cached-llada', api='llada', concurrent_requests=1,
                           max_tokens=8, diffusion_steps=8, block_length=4)
            self.assertEqual(list(api.run_queries(['A problem'])), [expected])
            self.assertEqual(engine.return_value.run_queries.call_args.args[0],
                             [([{'role': 'user', 'content': 'A problem'}], None)])
            api.free_model()
            engine.return_value.close.assert_called_once()

    def test_selection_is_stable_and_two_shards_are_disjoint_and_complete(self):
        files = [f'{i:03}.json' for i in range(451)]
        selected = select_files(files, 30, 42)
        self.assertEqual(selected, select_files(list(reversed(files)), 30, 42))
        self.assertEqual(len(selected[0::2]), 15)
        self.assertEqual(len(selected[1::2]), 15)
        self.assertFalse(set(selected[0::2]) & set(selected[1::2]))
        self.assertEqual(set(selected[0::2]) | set(selected[1::2]), set(selected))

    def test_resume_rejects_changed_input_or_configuration(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            problem = folder / 'problem.json'
            problem.write_text(json.dumps({'problem_id': 'test_1', 'problem': 'Original'}))
            (folder / 'model.yaml').write_text('api: llada\n')
            project = OverallConfig(name='test', model_configs=['model'], model_config_folder=str(folder))
            solver = SolverConfig(prompt='{problem}', n_samples=1)
            path = folder / 'manifest.json'
            first = ensure_manifest(path, project, solver, [str(problem)])
            self.assertEqual(first, ensure_manifest(path, project, solver, [str(problem)]))
            problem.write_text(json.dumps({'problem_id': 'test_1', 'problem': 'Changed'}))
            with self.assertRaisesRegex(ValueError, 'changed'):
                ensure_manifest(path, project, solver, [str(problem)])
            problem.write_text(json.dumps({'problem_id': 'test_1', 'problem': 'Original'}))
            solver.n_samples = 2
            with self.assertRaisesRegex(ValueError, 'changed'):
                ensure_manifest(path, project, solver, [str(problem)])

    def test_postprocessing_keeps_short_and_empty_responses_when_requested(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sample.json'
            path.write_text(json.dumps({'solution': 'Reference', 'attempts': [
                {'solution': 'False.', 'model_id': 'llada/test'},
                {'solution': '', 'model_id': 'llada/test', 'generation': {'status': 'empty'}},
            ]}))
            self.assertEqual(process_sample(path), [])
            retained = process_sample(path, min_solution_length=0)
            self.assertEqual([row['solution'] for row in retained], ['False.', ''])
            self.assertEqual(retained[0]['solutions'][0]['solution'], 'Reference')

    def test_preprocessing_accepts_singular_and_plural_reference_fields(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            raw = folder / 'raw'
            raw.mkdir()
            (raw / 'sample.json').write_text(json.dumps([
                {'problem_id': 'test_1', 'problem': 'P1', 'solution': 'S1'},
                {'problem_id': 'test_2', 'problem': 'P2', 'solutions': [{'solution': 'S2'}]},
            ]))
            config = {'name': 'test', 'model_configs': [], 'raw_base_folder': str(raw),
                      'unsolved_base_folder': str(folder / 'unsolved')}
            (folder / 'test.yaml').write_text(yaml.safe_dump(config))
            completed = subprocess.run([sys.executable, str(ROOT / 'scripts/process.py'),
                                        '--project', 'test', '--config-base', str(folder)],
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            for number in (1, 2):
                row = json.loads((folder / f'unsolved/test/{number}.json').read_text())
                self.assertEqual(row['solutions'][0]['solution'], f'S{number}')


if __name__ == '__main__':
    unittest.main()
