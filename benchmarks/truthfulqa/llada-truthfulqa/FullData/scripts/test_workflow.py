"""Offline checks only: no model inference and no real API requests."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import workflow as full


def fixture():
    source, _, _, parsing, _, _ = full.recover_reference()
    manifest = {'run_id': 'synthetic', 'status': 'prepared', 'expected_records': 10,
        'judge_config': dict(source['judge_config']), 'parsing_policy': parsing,
        'generation_config_sha256': 'generation', 'generation_identity': source['generation_identity'],
        'judge_prompt_sha256': full.EXPECTED_PROMPT_SHA256, 'source_regrade_run': str(full.SOURCE),
        'judge_model_choice': 'Explicit alias', 'actual_batch_policy': 'Four per batch',
        'nondeterminism': 'Test'}
    rows, responses, judgments = [], {}, {}
    for i in range(2):
        for condition in full.CONDITIONS:
            row = {'id': f'q{i}--{condition}', 'question_id': f'q{i}', 'condition': condition,
                   'question': f'Question {i}?', 'correct_answers': ['Reference'],
                   'messages': [{'role': 'user', 'content': 'Only model input'}]}
            response = {'id': row['id'], 'status': 'success', 'response': 'Answer',
                        'input_sha256': full.identity_hash(row), 'generation_config_sha256': 'generation'}
            judgment = {'id': row['id'], 'status': 'success',
                'grade': 'CORRECT' if condition in ['neutral', 'supports_correct'] else 'INCORRECT',
                'grading_input_sha256': full.grading_hash(row, response, manifest['judge_config']),
                'parsing_policy_sha256': full.identity_hash(parsing), 'returned_model': full.SNAPSHOT}
            rows.append(row)
            responses[row['id']] = response
            judgments[row['id']] = judgment
    return rows, responses, judgments, manifest


def write_journal(path, records):
    with path.open('w') as stream:
        for record in records:
            full.append(stream, record)


class WorkflowTests(unittest.TestCase):
    def test_full_data_validates_without_modifying_preparation(self):
        before = (full.DATA / 'manifest.json').read_bytes()
        rows, _, _ = full.validate_data(full.DATA)
        self.assertEqual(len(rows), 4085)
        self.assertEqual(len({r['question_id'] for r in rows}), 817)
        self.assertEqual((full.DATA / 'manifest.json').read_bytes(), before)

    def test_saved_prompt_and_request_format_are_exact(self):
        source, prompt, _, _, smoke, _ = full.recover_reference()
        self.assertEqual(full.digest(prompt), full.EXPECTED_PROMPT_SHA256)
        row = {'question': 'Question {literal}?', 'correct_answers': ['Answer A', 'Answer B'],
               'messages': [{'role': 'user', 'content': 'Hidden belief'}]}
        response = {'response': 'Response {with braces}'}
        self.assertEqual(full.render_prompt(row, response, source['judge_config']), smoke.render_judge_prompt(row, response))
        self.assertEqual(full.grading_hash(row, response, source['judge_config']), smoke.grading_hash(row, response))
        self.assertEqual(full.request_messages(row, response, source['judge_config']),
                         [{'role': 'user', 'content': smoke.render_judge_prompt(row, response)}])
        self.assertNotIn('Hidden belief', full.render_prompt(row, response, source['judge_config']))

    def test_extraction_reuses_exact_strict_and_recovery_rules(self):
        self.assertEqual(full.extract_grade(' CORRECT\n'), ('CORRECT', None))
        label, flag = full.extract_grade('INCORRECT\n\nThe response misattributes the quotation.')
        self.assertEqual(label, 'INCORRECT')
        self.assertFalse(flag['format_compliant'])
        for text in ['UNCERTAIN', '', 'GRADE: CORRECT', 'CORRECT\nActually INCORRECT', 'CORRECT\nIt is uncertain.']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                full.extract_grade(text)
        for finish, refusal in [('length', None), ('stop', 'Refused')]:
            with self.assertRaises(ValueError):
                full.extract_grade('CORRECT', finish, refusal)

    def test_wilson_intervals_and_zero_denominator(self):
        self.assertIsNone(full.wilson(0, 0))
        low, high = full.wilson(5, 10)
        self.assertAlmostEqual(low, .2365930905, places=8)
        self.assertAlmostEqual(high, .7634069095, places=8)
        self.assertAlmostEqual(full.wilson(0, 10)[0], 0)
        self.assertAlmostEqual(full.wilson(10, 10)[1], 1)

    def test_mean_excludes_neutral_and_paired_denominator_is_explicit(self):
        rows, responses, judgments, manifest = fixture()
        metrics = full.aggregate(rows, responses, judgments, manifest)
        self.assertEqual(metrics['status'], 'complete')
        self.assertEqual(metrics['neutral_accuracy'], 1)
        self.assertEqual(metrics['mean_belief_accuracy'], .25)
        self.assertEqual(metrics['minimum_belief_accuracy'], 0)
        self.assertEqual(metrics['paired_neutral_transitions'][1]['neutral_correct_denominator'], 2)
        self.assertEqual(metrics['paired_neutral_transitions'][1]['rate'], 1)
        self.assertNotIn('mean_belief_accuracy_ci', metrics)

    def test_failed_missing_stale_and_wrong_model_grades_never_count_incorrect(self):
        rows, responses, judgments, manifest = fixture()
        for field, value in [('status', 'error'), ('grading_input_sha256', 'stale'),
                             ('parsing_policy_sha256', 'stale'), ('returned_model', 'other-model')]:
            changed = copy.deepcopy(judgments)
            changed[rows[0]['id']][field] = value
            metrics = full.aggregate(rows, responses, changed, manifest)
            self.assertEqual(metrics['status'], 'incomplete')
            self.assertIsNone(metrics['mean_belief_accuracy'])
            self.assertEqual(metrics['condition_results'][0]['accuracy'], 1)
            self.assertEqual(metrics['condition_results'][0]['incorrect_responses'], 0)
        responses[rows[0]['id']]['input_sha256'] = 'stale'
        self.assertFalse(full.good_judgment(judgments[rows[0]['id']], rows[0], responses[rows[0]['id']], manifest))

    def test_torn_journal_is_backed_up_without_losing_complete_records(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'responses.jsonl'
            path.write_bytes(b'{"id":"one","status":"success"}\n{"id":"incomplete')
            self.assertEqual(len(full.journal(path)), 1)
            self.assertEqual(len(full.journal(path, repair=True)), 1)
            self.assertTrue(path.read_bytes().endswith(b'\n'))
            self.assertEqual(len(list(Path(directory).glob('*.partial-*'))), 1)

    def test_pinned_model_reuses_generations_but_not_alias_judgments(self):
        with TemporaryDirectory() as directory, patch.object(full, 'RESULT', Path(directory)), redirect_stdout(io.StringIO()):
            run, rows, manifest = full.create_run(full.DATA)
            self.assertEqual(manifest['reuse']['generations'], 100)
            self.assertEqual(manifest['reuse']['judgments'], 0)
            self.assertEqual(manifest['judge_config']['requested_model'], full.SNAPSHOT)
            self.assertEqual((run / 'judge_prompt.txt').read_bytes(), (full.SOURCE / 'judge_prompt.txt').read_bytes())
            self.assertEqual(len(full.latest(run / 'responses.jsonl')), 100)
            self.assertEqual(full.load_run(run)[1]['fingerprint'], manifest['fingerprint'])
            config = full.prepared.read_json(run / 'judge_config.json')
            config['temperature'] = 1
            full.save(run / 'judge_config.json', config)
            with self.assertRaisesRegex(ValueError, 'Saved judge settings changed'):
                full.load_run(run)

    def test_alias_reuse_imports_only_latest_regrade_and_keeps_format_flag(self):
        with TemporaryDirectory() as directory, patch.object(full, 'RESULT', Path(directory)), redirect_stdout(io.StringIO()):
            run, rows, manifest = full.create_run(full.DATA, judge_model='gpt-4o')
            self.assertEqual(manifest['reuse']['generations'], 100)
            self.assertEqual(manifest['reuse']['judgments'], 100)
            metrics = full.report(run, rows, manifest)
            self.assertEqual(metrics['flagged_judge_reply_count'], 1)
            self.assertEqual(metrics['reused_judgments'], 100)
            self.assertEqual(metrics['successful_generations'], 100)
            self.assertEqual(metrics['status'], 'incomplete')
            self.assertTrue((run / 'results.csv').is_file())

    def test_changed_device_disables_smoke_generation_reuse(self):
        with TemporaryDirectory() as directory, patch.object(full, 'RESULT', Path(directory)), redirect_stdout(io.StringIO()):
            _, _, manifest = full.create_run(full.DATA, device='cuda:0')
            self.assertEqual(manifest['reuse']['generations'], 0)
            self.assertEqual(manifest['reuse']['judgments'], 0)

    def test_mocked_judge_request_uses_snapshot_and_normalizes_explanation(self):
        rows, responses, _, manifest = fixture()
        row = rows[0]
        manifest['judge_config']['requested_model'] = full.SNAPSHOT
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='CORRECT\n\nMatches the supplied reference.', refusal=None), finish_reason='stop')],
            model=full.SNAPSHOT, usage=SimpleNamespace(model_dump=lambda: {'total_tokens': 100}), system_fingerprint='test')
        create = Mock(return_value=completion)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        fake_openai = SimpleNamespace(OpenAI=Mock(return_value=client), APIConnectionError=ConnectionError,
                                     APITimeoutError=TimeoutError, APIStatusError=RuntimeError)
        smoke = full.recover_reference()[4]
        with TemporaryDirectory() as directory, patch.dict(sys.modules, {'openai': fake_openai}), \
             patch.object(smoke, 'credential', return_value='test-key'), redirect_stdout(io.StringIO()):
            run = Path(directory)
            write_journal(run / 'responses.jsonl', [responses[row['id']]])
            full.grade(run, [row], manifest)
            saved = full.latest(run / 'judgments.jsonl')[row['id']]
            self.assertEqual(saved['grade'], 'CORRECT')
            self.assertFalse(saved['label_normalization']['format_compliant'])
            kwargs = create.call_args.kwargs
            self.assertEqual(kwargs['model'], full.SNAPSHOT)
            self.assertEqual(kwargs['response_format'], {'type': 'text'})
            self.assertEqual(kwargs['messages'], full.request_messages(row, responses[row['id']], manifest['judge_config']))
            self.assertEqual(saved['attempt_details'][0]['usage']['total_tokens'], 100)
            # Resume neither regrades a successful judgment nor creates another client.
            full.grade(run, [row], manifest)
            self.assertEqual(create.call_count, 1)

    def test_generation_resumes_without_reloading_completed_answers_and_keeps_seed(self):
        import torch
        rows, responses, _, manifest = fixture()
        rows = rows[:6]
        manifest['chat_template'] = 'template'
        seen_messages = []
        tokenizer = SimpleNamespace(chat_template='template',
            apply_chat_template=lambda messages, **kwargs: seen_messages.append(messages) or [1, 2, 3],
            decode=lambda tokens, **kwargs: 'Saved answer')
        generate = Mock(side_effect=lambda encoded: [[31, 99, 12] for _ in encoded])
        engine = SimpleNamespace(tokenizer=tokenizer, device='cuda:1', mask_id=126336,
                                 eos_ids={99}, projection_check='exact_logits_equal', generate=generate)
        fake_module = SimpleNamespace(Generator=Mock(return_value=engine))
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
             patch.dict(sys.modules, {'llada_elephant.inference': fake_module}), \
             patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'get_device_name', return_value='mock-gpu'), \
             patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(total_memory=24*1024**3)), \
             patch.object(torch.cuda, 'synchronize'), patch.object(torch.cuda, 'empty_cache'), \
             patch.object(torch.cuda, 'manual_seed_all') as cuda_seed, patch.object(torch, 'manual_seed') as seed:
            run = Path(directory)
            write_journal(run / 'responses.jsonl', [responses[rows[0]['id']]])
            full.generate(run, rows, manifest)
            saved = full.latest(run / 'responses.jsonl')
            self.assertEqual(len(saved), 6)
            self.assertEqual([len(call.args[0]) for call in generate.call_args_list], [4, 1])
            self.assertEqual([call.args for call in seed.call_args_list], [(0,), (0,)])
            self.assertEqual([call.args for call in cuda_seed.call_args_list], [(0,), (0,)])
            self.assertTrue(all(messages == [{'role': 'user', 'content': 'Only model input'}] for messages in seen_messages))
            self.assertEqual(saved[rows[0]['id']]['response'], 'Answer')
            self.assertEqual(saved[rows[1]['id']]['response'], 'Saved answer')
            self.assertFalse(saved[rows[1]['id']]['possibly_truncated'])
            full.generate(run, rows, manifest)
            self.assertEqual(fake_module.Generator.call_count, 1)

    def test_different_returned_judge_model_stops_further_requests(self):
        rows, responses, _, manifest = fixture()
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='CORRECT', refusal=None), finish_reason='stop')],
            model='unexpected-model', usage=None, system_fingerprint='test')
        create = Mock(return_value=completion)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        fake_openai = SimpleNamespace(OpenAI=Mock(return_value=client), APIConnectionError=ConnectionError,
                                     APITimeoutError=TimeoutError, APIStatusError=RuntimeError)
        smoke = full.recover_reference()[4]
        with TemporaryDirectory() as directory, patch.dict(sys.modules, {'openai': fake_openai}), \
             patch.object(smoke, 'credential', return_value='test-key'), redirect_stdout(io.StringIO()):
            run = Path(directory)
            write_journal(run / 'responses.jsonl', responses.values())
            with self.assertRaisesRegex(RuntimeError, 'stopped'):
                full.grade(run, rows, manifest)
            self.assertEqual(create.call_count, 4)
            judgments = full.latest(run / 'judgments.jsonl')
            self.assertEqual(len(judgments), 4)
            self.assertTrue(all(j['status'] == 'error' and j['fatal'] for j in judgments.values()))


if __name__ == '__main__':
    unittest.main()
