from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_inference as chat


class ChatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = chat.AutoTokenizer.from_pretrained(
            chat.DEFAULT_MODEL, trust_remote_code=True, local_files_only=True,
        )
        cls.model = SimpleNamespace(config=SimpleNamespace(
            eos_token_id=cls.tokenizer.eos_token_id,
            mask_token_id=126336, max_sequence_length=4096,
        ))

    def test_actual_chat_template_carries_previous_user_and_assistant(self):
        tokenizer = self.tokenizer
        output = tokenizer.encode('Your name is Sam.', add_special_tokens=False)
        output += [tokenizer.eos_token_id]
        output += tokenizer.encode('Trailing text must not enter history.', add_special_tokens=False)
        with patch.object(chat, 'generate', return_value=torch.tensor([output])) as generation:
            first = chat.reply(self.model, tokenizer, [], 'My name is Sam.', 'cpu', 16, 16)
            second = chat.reply(self.model, tokenizer, first['messages'], 'What is my name?', 'cpu', 16, 16)
        self.assertEqual(first['response'], 'Your name is Sam.')
        self.assertFalse(first['possibly_truncated'])
        self.assertEqual([m['role'] for m in second['messages']], ['user', 'assistant', 'user', 'assistant'])
        expected = tokenizer.apply_chat_template([
            {'role': 'user', 'content': 'My name is Sam.'},
            {'role': 'assistant', 'content': 'Your name is Sam.'},
            {'role': 'user', 'content': 'What is my name?'},
        ], tokenize=True, add_generation_prompt=True)
        self.assertEqual(generation.call_args.args[1][0].tolist(), expected)
        self.assertNotIn('Trailing text', str(second['messages']))

    def test_context_overflow_and_generation_error_do_not_mutate_history(self):
        history = [{'role': 'user', 'content': 'First question'},
                   {'role': 'assistant', 'content': 'First answer'}]
        original = [dict(m) for m in history]
        limited = SimpleNamespace(config=SimpleNamespace(mask_token_id=126336, max_sequence_length=1))
        with patch.object(chat, 'generate') as generation:
            with self.assertRaisesRegex(ValueError, '/reset'):
                chat.reply(limited, self.tokenizer, history, 'Follow-up', 'cpu', 16, 16)
            generation.assert_not_called()
        with patch.object(chat, 'generate', side_effect=RuntimeError('Inference failure')):
            with self.assertRaisesRegex(RuntimeError, 'Inference failure'):
                chat.reply(self.model, self.tokenizer, history, 'Follow-up', 'cpu', 16, 16)
        self.assertEqual(history, original)

    def test_end_of_turn_stops_history_and_missing_end_is_flagged(self):
        tokens = self.tokenizer.encode('Hello', add_special_tokens=False)
        response, truncated = chat.decode_response(self.tokenizer, self.model, tokens)
        self.assertEqual(response, 'Hello')
        self.assertTrue(truncated)
        eot = self.tokenizer.get_vocab()['<|eot_id|>']
        response, truncated = chat.decode_response(self.tokenizer, self.model, [*tokens, eot, *tokens])
        self.assertEqual(response, 'Hello')
        self.assertFalse(truncated)

    def fake_reply(self, model, tokenizer, history, user_text, device, length, steps):
        return {'prompt': user_text, 'response': 'Reply to ' + user_text,
                'messages': [*history, {'role': 'user', 'content': user_text},
                             {'role': 'assistant', 'content': 'Reply to ' + user_text}],
                'possibly_truncated': False, 'generated_token_ids': [1],
                'generation_seconds': 0, 'input_tokens': 10}

    def test_interactive_history_reset_and_per_turn_transcript(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'nested/chat.json'
            fake_model = SimpleNamespace(eval=lambda: self.model)
            with patch.object(chat.AutoTokenizer, 'from_pretrained', return_value=self.tokenizer), \
                 patch.object(chat.AutoModel, 'from_pretrained', return_value=fake_model), \
                 patch.object(chat, 'reply', side_effect=self.fake_reply) as reply, \
                 patch('builtins.input', side_effect=['My name is Sam.', 'What is my name?', '/reset', 'Hello', '/exit']), \
                 patch.object(sys, 'argv', ['test_inference.py', '--device', 'cpu', '--output', str(output)]), \
                 redirect_stdout(io.StringIO()):
                chat.main()
            result = json.loads(output.read_text())
            self.assertEqual(result['status'], 'completed')
            self.assertEqual([turn['conversation_index'] for turn in result['turns']], [0, 0, 1])
            self.assertEqual([len(call.args[2]) for call in reply.call_args_list], [0, 2, 0])
            self.assertEqual(len(result['messages']), 2)
            self.assertEqual(result['messages'][0]['content'], 'Hello')
            self.assertEqual(list(output.parent.glob('*.tmp')), [])

    def test_scripted_turns_need_no_interactive_input(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'chat.json'
            fake_model = SimpleNamespace(eval=lambda: self.model)
            with patch.object(chat.AutoTokenizer, 'from_pretrained', return_value=self.tokenizer), \
                 patch.object(chat.AutoModel, 'from_pretrained', return_value=fake_model), \
                 patch.object(chat, 'reply', side_effect=self.fake_reply), \
                 patch('builtins.input', side_effect=AssertionError('Unexpected interactive input')), \
                 patch.object(sys, 'argv', ['test_inference.py', '--device', 'cpu',
                     '--prompt', 'First', '--prompt', 'Second', '--no-interactive', '--output', str(output)]), \
                 redirect_stdout(io.StringIO()):
                chat.main()
            result = json.loads(output.read_text())
            self.assertEqual(len(result['messages']), 4)
            self.assertEqual([t['prompt'] for t in result['turns']], ['First', 'Second'])

    def test_diffusion_keeps_the_entire_multi_turn_prompt_unchanged(self):
        class Model:
            def __call__(self, tokens):
                logits = torch.zeros((*tokens.shape, 8))
                logits[..., 3] = 10
                return SimpleNamespace(logits=logits)
        prompt = torch.tensor([[1, 2, 4, 5]])
        with redirect_stdout(io.StringIO()):
            output = chat.generate(Model(), prompt, 4, 4, 7)
        self.assertEqual(output.tolist(), [[3, 3, 3, 3]])
        self.assertEqual(prompt.tolist(), [[1, 2, 4, 5]])


if __name__ == '__main__':
    unittest.main()
