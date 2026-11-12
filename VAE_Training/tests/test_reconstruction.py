import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from vae.model import ReasoningVAE, VAEConfig
from vae.reconstruction import decode_latents


class ReconstructionTests(unittest.TestCase):
    def test_free_prefix_matches_teacher_forced_first_prediction(self):
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                            head_dim=8, max_position_embeddings=64, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        model = ReasoningVAE(VAEConfig(num_latents=3, latent_dim=8),
                             Qwen3ForCausalLM(config), Qwen3ForCausalLM(config)).eval()
        ids = torch.tensor([[3, 4, 5], [6, 7, 8]])
        mask = torch.ones_like(ids)
        with torch.no_grad():
            mean, _ = model.encode(ids, mask)
            hidden = model.decoder_hidden(mean, ids, mask)
            expected = model.decoder_model.lm_head(hidden[:, 0]).argmax(-1).tolist()
        class Tokenizer:
            eos_token_id = 2
            def decode(self, tokens, **kwargs):
                return str(tokens)
        records = decode_latents(model, mean, Tokenizer(), max_new_tokens=1)
        self.assertEqual([r["generated_token_ids"][0] for r in records], expected)
        self.assertFalse(model.training)
        model.train()
        decode_latents(model, mean, Tokenizer(), max_new_tokens=1)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
