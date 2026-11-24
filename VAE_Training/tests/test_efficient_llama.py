import unittest

import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from vae.adaptive_batches import physical_batches, padded_batch
from vae.efficient_decoder import checkpoint_frozen_decoder, reconstruction_loss, decode_prefix
from vae.attention import build_mask
from vae.llama_offload import frozen_decoder_loss


class EfficientDecoderTests(unittest.TestCase):
    def test_unequal_batches_preserve_sentence_objective_and_gradients(self):
        torch.manual_seed(10)
        config = LlamaConfig(vocab_size=43, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, attention_dropout=0.)
        decoder = AutoModelForCausalLM.from_config(config, attn_implementation='sdpa')
        decoder.requires_grad_(False); decoder.eval()
        rows = [{'index': index, 'input_ids': [3 + index] * count + [2]}
                for index, count in enumerate([6, 2, 4, 5, 3])]
        prefix = torch.randn(5, 5, 32, requires_grad=True)
        individual = []
        for row in rows:
            ids, mask = padded_batch([row], 2, 'cpu')
            individual.append(frozen_decoder_loss(decoder, prefix[row['index']:row['index'] + 1],
                                                  ids, mask, num_latents=4))
        reference = torch.stack(individual).mean()
        reference.backward(); expected_gradient = prefix.grad.clone()
        other = prefix.detach().clone().requires_grad_()
        checkpoint_frozen_decoder(decoder)
        actual = 0
        for batch in physical_batches(rows, max_examples=2, max_tokens=16):
            ids, mask = padded_batch(batch, 2, 'cpu')
            selected = other[[row['index'] for row in batch]]
            ce, metrics = reconstruction_loss(decoder, selected, ids, mask, chunk_tokens=3)
            self.assertEqual(metrics['tokens'], sum(len(row['input_ids']) for row in batch))
            actual = actual + ce * len(batch) / len(rows)
        actual.backward()
        self.assertTrue(torch.allclose(actual, reference, atol=1e-6))
        self.assertTrue(torch.allclose(other.grad, expected_gradient, atol=1e-6))
        self.assertFalse(decoder.training)
        self.assertTrue(all(not layer.training for layer in decoder.model.layers))
        self.assertTrue(all(parameter.grad is None for parameter in decoder.parameters()))

    def test_cached_latent_only_generation_matches_full_prefix_decoding(self):
        torch.manual_seed(15)
        config = LlamaConfig(vocab_size=43, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, attention_dropout=0.)
        decoder = AutoModelForCausalLM.from_config(config, attn_implementation='sdpa').eval()
        prefix = torch.randn(2, 7, 32)
        class Tokenizer:
            eos_token_id = -1
            def decode(self, tokens, **kwargs):
                return str(tokens)
        generated = decode_prefix(decoder, prefix, Tokenizer(), maximum=5)
        tokens = [[] for _ in range(len(prefix))]
        with torch.no_grad():
            for step in range(5):
                embeds = prefix
                if step:
                    ids = torch.tensor(tokens)
                    embeds = torch.cat((prefix, decoder.get_input_embeddings()(ids)), dim=1)
                keys = torch.ones(embeds.shape[:2], dtype=torch.long)
                hidden = decoder.model(inputs_embeds=embeds,
                    attention_mask=build_mask('hybrid', keys, 0, 6, embeds.dtype),
                    use_cache=False).last_hidden_state
                predicted = decoder.lm_head(hidden[:, -1]).argmax(-1).tolist()
                for row, token in zip(tokens, predicted):
                    row.append(token)
        self.assertEqual([row['generated_token_ids'] for row in generated], tokens)
        self.assertTrue(all(row['token_limit_reached'] for row in generated))

    def test_long_targets_are_retained_without_truncation(self):
        rows = [{'index': index, 'input_ids': [3] * length}
                for index, length in enumerate([1608, 1024, 512, 128, 64, 16, 4] * 19)][:128]
        batches = list(physical_batches(rows))
        self.assertEqual(sorted(row['index'] for batch in batches for row in batch), list(range(128)))
        for batch in batches:
            self.assertLessEqual(len(batch), 32)
            self.assertLessEqual(len(batch) * max(len(row['input_ids']) for row in batch), 4096)


if __name__ == '__main__': unittest.main()
