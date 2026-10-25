import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import finalize_regrade as finalize


class FinalizeTests(unittest.TestCase):
    def test_recover_explicit_leading_label_from_format_error_only(self):
        row = {'status': 'error', 'error': 'ValueError; HTTP None', 'finish_reason': 'stop',
               'returned_model': 'test', 'raw_judge_response': 'INCORRECT\n\nThe response misattributes the quotation.'}
        self.assertEqual(finalize.recover_label(row), 'INCORRECT')
        for change in [
            {'raw_judge_response': 'INCORRECT\nActually CORRECT'},
            {'raw_judge_response': 'INCORRECT\nThe answer is uncertain.'},
            {'raw_judge_response': 'GRADE: INCORRECT'},
            {'raw_judge_response': 'Not a grade'}, {'raw_judge_response': None},
            {'refusal': 'Refused'}, {'finish_reason': 'length'},
            {'error': 'APIConnectionError; HTTP None'}, {'returned_model': None},
        ]:
            with self.subTest(change=change):
                self.assertIsNone(finalize.recover_label({**row, **change}))


if __name__ == '__main__':
    unittest.main()
