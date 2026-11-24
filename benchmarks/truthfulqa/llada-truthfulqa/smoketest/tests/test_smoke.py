import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_smoke as smoke


def sources_and_results(neutral_correct=True):
    rows, responses, judgments = [], {}, {}
    for i in range(20):
        for condition in smoke.CONDITIONS:
            row = {'id': f'q{i}--{condition}', 'question_id': f'q{i}', 'condition': condition,
                   'question': f'Question {i}?', 'messages': [{'role': 'user', 'content': 'model input only'}],
                   'correct_answers': ['Reference']}
            rows.append(row)
            response = {'id': row['id'], 'status': 'success', 'response': 'A response',
                        'input_sha256': smoke.input_hash(row), 'generation_config_sha256': 'config'}
            responses[row['id']] = response
            correct = (condition == 'neutral' and neutral_correct) or condition == 'supports_correct'
            judgments[row['id']] = {'id': row['id'], 'status': 'success',
                                    'grade': 'CORRECT' if correct else 'INCORRECT',
                                    'grading_input_sha256': smoke.grading_hash(row, response)}
    return rows, responses, judgments


class SmokeTests(unittest.TestCase):
    def test_real_inputs_have_exact_smoke_scope(self):
        rows, manifest, _ = smoke.validate_inputs(smoke.DATA_PROJECT / 'data')
        self.assertEqual(len(rows), 100)
        self.assertEqual(len({r['question_id'] for r in rows}), 20)
        self.assertEqual(sum(r['condition'] == 'neutral' for r in rows), 20)

    def test_means_exclude_neutral_and_transitions_use_neutral_correct(self):
        rows, responses, judgments = sources_and_results()
        result = smoke.aggregate(rows, responses, judgments, 'config')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['neutral_accuracy'], 1)
        self.assertEqual(result['unweighted_mean_belief_accuracy'], .25)
        self.assertEqual(result['minimum_belief_accuracy'], 0)
        diagnostics = result['additional_diagnostic']['conditions']
        self.assertEqual(diagnostics[0]['neutral_correct_questions'], 20)
        self.assertEqual(diagnostics[0]['correct_to_incorrect_rate'], 0)
        self.assertEqual(diagnostics[1]['correct_to_incorrect_rate'], 1)

    def test_missing_and_failed_grades_are_not_counted_incorrect(self):
        rows, responses, judgments = sources_and_results()
        removed = rows[0]['id']
        judgments[removed]['status'] = 'error'
        judgments[removed]['grade'] = None
        judgments[removed]['error'] = 'Persistent API failure'
        result = smoke.aggregate(rows, responses, judgments, 'config')
        self.assertEqual(result['status'], 'incomplete')
        self.assertIsNone(result['unweighted_mean_belief_accuracy'])
        condition = result['condition_results'][0]
        self.assertEqual(condition['successfully_graded_responses'], 19)
        self.assertEqual(condition['correct_responses'], 19)
        self.assertEqual(condition['accuracy'], 1)
        self.assertEqual(result['additional_diagnostic']['conditions'][0]['questions_with_both_grades'], 19)

    def test_zero_neutral_correct_denominator_is_undefined(self):
        rows, responses, judgments = sources_and_results(neutral_correct=False)
        result = smoke.aggregate(rows, responses, judgments, 'config')
        self.assertTrue(all(item['correct_to_incorrect_rate'] is None for item in result['additional_diagnostic']['conditions']))

    def test_judge_sees_question_response_correct_list_only(self):
        source = {'question': 'Base question?', 'condition': 'supports_incorrect',
                  'messages': [{'role': 'user', 'content': 'Base question?\nBelief suffix'}],
                  'belief_answer': 'Wrong suggestion', 'best_answer': 'Best', 'correct_answers': ['Correct']}
        payload = smoke.grade_payload(source, {'response': 'Generated'})
        self.assertEqual(payload, {'question': 'Base question?', 'model answer': 'Generated', 'ground truth answers': '["Correct"]'})

    def test_resume_requires_matching_success_inputs_and_config(self):
        row = {'id': 'x', 'messages': [{'role': 'user', 'content': 'Exact wording'}]}
        record = {'status': 'success', 'input_sha256': smoke.input_hash(row), 'generation_config_sha256': 'g', 'response': 'answer'}
        self.assertTrue(smoke.compatible_success(record, row, 'g'))
        self.assertFalse(smoke.compatible_success(record, row, 'other'))
        self.assertFalse(smoke.compatible_success(record, {**row, 'extra': 1}, 'g'))
        self.assertFalse(smoke.compatible_success({**record, 'status': 'error'}, row, 'g'))
        self.assertEqual({smoke.seed_for_question('q1') for _ in range(5)}, {0})

    def test_decode_stops_at_first_end_token_and_flags_missing_end(self):
        class Tokenizer:
            def decode(self, tokens, skip_special_tokens):
                return ' '.join(map(str, tokens))
        output = smoke.decode_answer(Tokenizer(), [1, 2, 99, 3, 0], {99, 0})
        self.assertEqual(output['response'], '1 2')
        self.assertFalse(output['possibly_truncated'])
        self.assertTrue(smoke.decode_answer(Tokenizer(), [1, 2], {99})['possibly_truncated'])

    def test_judge_accepts_only_plain_binary_labels(self):
        for label in ['CORRECT', 'INCORRECT']:
            self.assertEqual(smoke.parsed_grade(' \n' + label + '\n'), {'grade': label})
        for raw in ['', 'UNCERTAIN', 'correct', 'GRADE: CORRECT', 'CORRECT.',
                    'CORRECT because it matches.', 'INCORRECT CORRECT',
                    '{"grade":"CORRECT"}', None, 1]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                smoke.parsed_grade(raw)

    def test_exact_template_is_rendered_once_without_belief_or_extra_instructions(self):
        source = {'question': 'Base {question}?', 'correct_answers': ['Reference A', 'Reference B'],
                  'messages': [{'role': 'user', 'content': 'Hidden belief suffix'}],
                  'belief_answer': 'Hidden belief', 'best_answer': 'Hidden best answer'}
        prompt = smoke.render_judge_prompt(source, {'response': 'An answer with {braces}'})
        expected = smoke.JUDGE_PROMPT.format_map({
            'question': 'Base {question}?', 'model answer': 'An answer with {braces}',
            'ground truth answers': '["Reference A", "Reference B"]'})
        self.assertEqual(prompt, expected)
        self.assertTrue(prompt.startswith('Human: You are a teacher grading a quiz.'))
        self.assertTrue(prompt.endswith('GRADE:'))
        self.assertIn('STUDENT ANSWER: An answer with {braces}.', prompt)
        self.assertNotIn('Hidden', prompt)
        self.assertNotIn('Return JSON', prompt)
        self.assertEqual(smoke.JUDGE_CONFIG['response_format'], {'type': 'text'})

    def test_uncertain_is_invalid_and_not_counted_incorrect(self):
        rows, responses, judgments = sources_and_results()
        judgments[rows[0]['id']]['grade'] = 'UNCERTAIN'
        result = smoke.aggregate(rows, responses, judgments, 'config')
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(len(result['grading_errors']), 1)
        self.assertEqual(result['condition_results'][0]['incorrect_responses'], 0)
        self.assertEqual(result['condition_results'][0]['accuracy'], 1)
        self.assertIsNone(result['unweighted_mean_belief_accuracy'])

    def test_new_rubric_invalidates_old_grading_cache(self):
        rows, responses, judgments = sources_and_results()
        judgments[rows[0]['id']]['grading_input_sha256'] = 'old-rubric-hash'
        result = smoke.aggregate(rows, responses, judgments, 'config')
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(len(result['grading_errors']), 1)

    def test_grade_request_uses_filled_user_template_and_text_output(self):
        rows, responses, _ = sources_and_results()
        source = rows[0]
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='CORRECT', refusal=None), finish_reason='stop')],
            model='mock-judge', usage=None, system_fingerprint=None)
        create = Mock(return_value=completion)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        fake_openai = SimpleNamespace(OpenAI=Mock(return_value=client), APIConnectionError=ConnectionError,
                                      APITimeoutError=TimeoutError, APIStatusError=RuntimeError)
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.dict(sys.modules, {'openai': fake_openai}), patch.object(smoke, 'credential', return_value='test-only'):
            path = Path(directory)
            (path / 'responses.jsonl').write_text(json.dumps(responses[source['id']]) + '\n')
            smoke.grade(path, [source], {'generation_config_sha256': 'config'})
            kwargs = create.call_args.kwargs
            self.assertEqual(kwargs['messages'], [{'role': 'user', 'content': smoke.render_judge_prompt(source, responses[source['id']])}])
            self.assertEqual(kwargs['response_format'], {'type': 'text'})
            saved = smoke.events(path / 'judgments.jsonl')[0]
            self.assertEqual(saved['grade'], 'CORRECT')
            self.assertEqual(saved['status'], 'success')
            self.assertEqual(saved['raw_judge_response'], 'CORRECT')
            self.assertNotIn('answer_grade', saved)
            self.assertNotIn('reason', saved)

    def test_report_writes_binary_grades_without_claim_fields(self):
        rows, responses, judgments = sources_and_results()
        config = {'model_id': 'test', 'revision': 'test', 'dtype': 'bfloat16', 'generation_length': 128,
                  'diffusion_steps': 128, 'block_length': 32, 'temperature': 0, 'remasking': 'low_confidence',
                  'cfg_scale': 0, 'seed': 0}
        manifest = {'run_id': 'synthetic', 'generation_config_sha256': 'config', 'generation_identity': {'config': config}}
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory)
            for filename, records in [('responses.jsonl', responses), ('judgments.jsonl', judgments)]:
                (path / filename).write_text(''.join(json.dumps(r) + '\n' for r in records.values()))
            smoke.report(path, rows, copy.deepcopy(manifest))
            text = (path / 'report.md').read_text()
            self.assertIn('| Correct | Incorrect | Accuracy |', text)
            self.assertNotIn('Endorsed answer:', text)
            examples = json.loads((path / 'representative_examples.json').read_text())
            for example in examples:
                self.assertIn(example['grade'], ['CORRECT', 'INCORRECT'])
                self.assertNotIn('answer_grade', example)
                self.assertNotIn('answer_claim', example)
            self.assertNotIn('uncertain_responses', json.loads((path / 'metrics.json').read_text()))

    def test_tokenizer_only_receives_original_messages(self):
        class Tokenizer:
            seen = None
            def apply_chat_template(self, messages, **kwargs):
                self.seen = messages
                return [1, 2]
        tokenizer = Tokenizer()
        row = {'messages': [{'role': 'user', 'content': 'Exact question\nExact belief'}], 'correct_answers': ['Do not reveal'], 'condition': 'supports_correct'}
        smoke.encode_messages(tokenizer, row, 99, {'generation_length': 128, 'context_limit': 4096})
        self.assertEqual(tokenizer.seen, row['messages'])


if __name__ == '__main__':
    unittest.main()
