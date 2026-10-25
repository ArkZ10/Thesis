import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import run_assessment as experiment


class AssessmentTests(unittest.TestCase):
    def test_exact_instruction_appended_without_mutating_original(self):
        source = {'id': 'q--supports_incorrect', 'question': 'Question?', 'correct_answers': ['Reference'],
                  'messages': [{'role': 'user', 'content': 'Question?\nOriginal belief.'}]}
        original = copy.deepcopy(source)
        changed = experiment.intervention([source])[0]
        self.assertEqual(source, original)
        self.assertEqual(changed['messages'], [{'role': 'user', 'content': source['messages'][0]['content'] + '\n\n' + experiment.INSTRUCTION}])
        self.assertEqual(changed['question'], source['question'])
        self.assertEqual(changed['correct_answers'], source['correct_answers'])
        self.assertNotIn('Reference', changed['messages'][0]['content'])
        self.assertNotEqual(experiment.common.identity_hash(changed), experiment.common.identity_hash(source))

    def test_scope_and_conditions_match_saved_baseline(self):
        rows, source, prompt, _, _, _, _ = experiment.validate()
        self.assertEqual(len(rows), 100)
        self.assertEqual(len({r['question_id'] for r in rows}), 20)
        for condition in experiment.common.CONDITIONS:
            self.assertEqual(sum(r['condition'] == condition for r in rows), 20)
        self.assertEqual(prompt, (experiment.BASELINE / 'judge_prompt.txt').read_bytes())
        self.assertEqual(source['judge_config']['requested_model'], 'gpt-4o')

    def test_format_compliance_is_a_separate_structural_check(self):
        self.assertTrue(experiment.format_check('Assessment: Facts.\n\nAnswer: Final.')['compliant'])
        self.assertTrue(experiment.format_check('Assessment:\nFacts.\nAnswer:\nFinal.')['compliant'])
        for text in ['Final.', 'Assessment:\nAnswer: Final.', 'Assessment: Facts.\nAnswer:',
                     'Intro\nAssessment: Facts.\nAnswer: Final.', 'Answer: Final.\nAssessment: Facts.',
                     'Assessment: Facts.\nAnswer: First.\nAnswer: Second.']:
            with self.subTest(text=text):
                self.assertFalse(experiment.format_check(text)['compliant'])

    def test_prepare_is_fresh_and_resume_rejects_tampering(self):
        with TemporaryDirectory() as temporary, patch.object(experiment, 'RESULT', Path(temporary)), redirect_stdout(io.StringIO()):
            directory, rows, manifest = experiment.prepare()
            self.assertEqual(manifest['reused_generations'], 0)
            self.assertEqual(manifest['reused_judgments'], 0)
            self.assertFalse((directory / 'responses.jsonl').exists())
            self.assertEqual(experiment.load(directory)[0], rows)
            self.assertEqual(manifest['generation_identity']['config'],
                             experiment.common.prepared.read_json(experiment.BASELINE / 'generation_config.json'))
            (directory / 'assessment_instruction.txt').write_text('Changed')
            with self.assertRaisesRegex(ValueError, 'instruction changed'):
                experiment.load(directory)

    def test_report_keeps_format_flags_separate_from_factual_grades(self):
        with TemporaryDirectory() as temporary, patch.object(experiment, 'RESULT', Path(temporary)), redirect_stdout(io.StringIO()):
            directory, rows, manifest = experiment.prepare()
            row = rows[0]
            response = {'id': row['id'], 'response': 'An answer without headings.', 'status': 'success',
                'input_sha256': experiment.common.identity_hash(row),
                'generation_config_sha256': manifest['generation_config_sha256'], 'possibly_truncated': False}
            judgment = {'id': row['id'], 'status': 'success', 'grade': 'CORRECT', 'returned_model': experiment.common.SNAPSHOT,
                'grading_input_sha256': experiment.common.grading_hash(row, response, manifest['judge_config']),
                'parsing_policy_sha256': experiment.common.identity_hash(manifest['parsing_policy'])}
            for name, record in [('responses.jsonl', response), ('judgments.jsonl', judgment)]:
                with (directory / name).open('w') as stream:
                    experiment.common.append(stream, record)
            metrics = experiment.summarize(directory, rows, manifest)
            self.assertEqual(metrics['status'], 'incomplete')
            self.assertEqual(metrics['successfully_graded_responses'], 1)
            self.assertEqual(metrics['condition_results'][0]['correct_responses'], 1)
            self.assertEqual(metrics['format_noncompliant_responses'], 1)
            self.assertIsNone(metrics['mean_belief_accuracy'])
            self.assertTrue((directory / 'results.csv').exists())
            self.assertEqual(len((directory / 'results.jsonl').read_text().splitlines()), 100)


if __name__ == '__main__':
    unittest.main()
