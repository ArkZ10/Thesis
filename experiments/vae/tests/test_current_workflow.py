"""CPU regression checks for retained data, encoder, and portable weights."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModel, LlamaConfig

from scripts.export_preserved_vae_weights import export, file_hash, llama_parameters, tensor_hash
from vae.indexed_math_data import IndexedMathData
from vae.llama_offload import LlamaLatentEncoder
from vae.pilot_metrics import normalized, reconstruction_values
from vae.runtime import decoder_hash


class CurrentWorkflowTests(unittest.TestCase):
    def test_indexed_data_keeps_token_and_source_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.jsonl'
            rows = [{'split': 'train', 'reasoning_block': 'x=3.'},
                    {'split': 'test', 'reasoning_block': 'HELD OUT'},
                    {'split': 'train', 'reasoning_block': 'y=4.'}]
            encoded = [(json.dumps(row) + '\n').encode() for row in rows]
            source.write_bytes(b''.join(encoded))
            np.array([11, 12, 2, 21, 2], dtype='<u4').tofile(root / 'train.tokens.u32')
            np.array([0, 3, 5], dtype='<u8').tofile(root / 'train.offsets.u64')
            np.array([0, len(encoded[0]) + len(encoded[1])], dtype='<u8').tofile(root / 'train.source-offsets.u64')
            data = IndexedMathData(root, source)
            try:
                self.assertEqual(len(data), 2)
                self.assertEqual(data[0]['input_ids'], [11, 12, 2])
                self.assertEqual(data[1]['reasoning_block'], 'y=4.')
                self.assertTrue(all(data[i]['split'] == 'train' for i in range(len(data))))
                with self.assertRaises(IndexError):
                    data[2]
            finally:
                data.close()

    def test_encoder_gradient_path_and_state_layout(self):
        torch.manual_seed(1)
        config = LlamaConfig(vocab_size=43, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)
        model = LlamaLatentEncoder(AutoModel.from_config(config, attn_implementation='sdpa'),
                                  num_latents=4, latent_dim=8)
        model.eval()
        ids = torch.tensor([[3, 4, 2], [5, 2, 2]])
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        prefix, kl, _, _ = model(ids, mask, eos_id=2, bos_id=1)
        self.assertEqual(prefix.shape, (2, 5, 32))
        (prefix.square().mean() + 1e-5 * kl).backward()
        for name, value in model.named_parameters():
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(torch.isfinite(value.grad).all(), name)
        keys = set(model.state_dict())
        self.assertTrue({'queries', 'ae', 'fc_mu.weight', 'fc_var.weight', 'to_decoder.weight'} <= keys)
        before = decoder_hash(model)
        with torch.no_grad():
            model.queries.add_(1)
        self.assertNotEqual(decoder_hash(model), before)

    def test_small_encoder_projects_to_larger_decoder_and_controls_noise(self):
        torch.manual_seed(2)
        config = LlamaConfig(vocab_size=43, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)
        model = LlamaLatentEncoder(AutoModel.from_config(config, attn_implementation='sdpa'),
                                  num_latents=6, latent_dim=8, decoder_hidden_size=48,
                                  input_substitution_probability=0., extra_latent_noise_std=0.,
                                  sample_posterior=False)
        ids, mask = torch.tensor([[3, 4, 2]]), torch.ones(1, 3, dtype=torch.long)
        model.train()
        first, _, _, _ = model(ids, mask, 2, 1)
        second, _, _, _ = model(ids, mask, 2, 1)
        self.assertEqual(first.shape, (1, 7, 48))
        self.assertTrue(torch.equal(first, second))
        model.sample_posterior = True
        noisy, _, _, _ = model(ids, mask, 2, 1)
        self.assertFalse(torch.equal(first[:, :-1], noisy[:, :-1]))
        self.assertTrue(torch.equal(first[:, -1:], noisy[:, -1:]))
        model.eval()
        mean, _, _, _ = model(ids, mask, 2, 1)
        self.assertTrue(torch.equal(first, mean))

    def test_exact_fp32_export_and_incomplete_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = [('head.weight', torch.nn.Parameter(torch.arange(6., dtype=torch.float32).reshape(2, 3))),
                      ('head.bias', torch.tensor([0.125, -0.25]))]
            dest = root / 'export'
            report = export(iter(values), dest, source=root / 'native', kind='llama', shard_bytes=24)
            self.assertTrue(report['all_exported_tensors_hash_verified'])
            self.assertEqual(report['tensor_count'], 2)
            self.assertFalse(report['exact_training_resume_from_export'])
            shard = dest / 'model-0000.safetensors'
            stamp = shard.stat().st_mtime_ns
            with safe_open(str(shard), framework='pt') as saved:
                self.assertEqual(tensor_hash(saved.get_tensor('head.weight')), tensor_hash(values[0][1]))
            with self.assertRaises(RuntimeError):
                export(iter(values), dest, source=root / 'native', kind='llama', shard_bytes=24, resume_incomplete=True)
            # Model an interrupted export with only complete shards left on disk.
            (dest / 'export-manifest.json').unlink()
            (dest / 'model.safetensors.index.json').unlink()
            export(iter(values), dest, source=root / 'native', kind='llama', shard_bytes=24, resume_incomplete=True)
            self.assertEqual(shard.stat().st_mtime_ns, stamp)

    def test_native_fp32_reader_handles_resident_and_swapped_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resident = torch.nn.Parameter(torch.tensor([1.25, -2.5]))
            swapped = torch.empty(0, dtype=torch.float32)
            swapped.ds_id = 7
            raw = root / 'offloaded_tensors/rank00/7.tensor.swp'
            raw.parent.mkdir(parents=True)
            np.array([3.125, -4.5, 5.25], dtype='<f4').tofile(raw)
            model_path = root / 'zero_pp_rank_0_mp_rank_00_model_states.pt'
            optimizer_path = root / 'bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt'
            torch.save({'dp_world_size': 1, 'shared_params': [], 'buffer_names': [],
                        'param_shapes': [{'resident': torch.Size([2])}, {'swapped': torch.Size([3])}]}, model_path)
            torch.save({'optimizer_state_dict': {'partition_count': 1,
                        'fp32_flat_groups': [resident, swapped]}}, optimizer_path)
            files = {str(p.relative_to(root)): {'sha256': file_hash(p), 'bytes': p.stat().st_size}
                     for p in [model_path, optimizer_path, raw]}
            (root / 'COMPLETE.json').write_text(json.dumps({'files': files}))
            result = dict(llama_parameters(root))
            self.assertTrue(torch.equal(result['resident'], resident))
            self.assertTrue(torch.equal(result['swapped'], torch.tensor([3.125, -4.5, 5.25])))

    def test_reconstruction_metrics_do_not_claim_math_equivalence(self):
        self.assertEqual(normalized(' x = 3. \n'), 'x = 3.')
        rows = [{'target': 'x=3.', 'text': 'x=4.', 'exact_match': False,
                 'eos_reached': True, 'token_limit_reached': False},
                {'target': 'Done.', 'text': 'Done.', 'exact_match': True,
                 'eos_reached': True, 'token_limit_reached': False}]
        metrics = reconstruction_values(rows)
        self.assertEqual(metrics['exact_count'], 1)
        self.assertEqual(metrics['numeric_total'], 1)
        self.assertEqual(metrics['numeric_count'], 0)


if __name__ == '__main__':
    unittest.main()
