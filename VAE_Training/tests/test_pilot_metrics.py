from pathlib import Path
import tempfile
import unittest

import torch

from vae.data import TokenizedDataset
from vae.pilot_metrics import TokenMetrics, reconstruction_values
from vae.train import load_config, train


class PilotMetricTests(unittest.TestCase):
    def test_token_weighting_and_exact_numbers(self):
        metric = TokenMetrics()
        metric.add({"ce_loss": torch.tensor(2.0), "kl_loss": torch.tensor(3.0), "n_tokens": 1, "n_correct": 1}, 1)
        metric.add({"ce_loss": torch.tensor(4.0), "kl_loss": torch.tensor(5.0), "n_tokens": 3, "n_correct": 1}, 1)
        values = metric.values(0.1)
        self.assertAlmostEqual(values["ce_loss"], 3.5)
        self.assertAlmostEqual(values["token_accuracy"], 0.5)
        self.assertAlmostEqual(values["loss"], 3.9)
        records = [{"target": "$2500 / 5 = 500", "text": "$250 / 5 = 50", "exact_match": False,
                    "eos_reached": True, "token_limit_reached": False},
                   {"target": "This holds.", "text": "This holds.", "exact_match": True,
                    "eos_reached": True, "token_limit_reached": False}]
        result = reconstruction_values(records)
        self.assertEqual(result["numeric_total"], 1)
        self.assertEqual(result["numeric_count"], 0)
        self.assertEqual(result["exact_count"], 1)

    def test_epoch_callbacks_include_last_partial_batch(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))
            def trainable_summary(self):
                return {"weight": {"params": 1, "trainable": 1}}
            def forward(self, input_ids, attention_mask, labels, kl_weight):
                loss = (self.weight - 1).square()
                return {"loss": loss, "ce_loss": loss, "kl_loss": loss * 0,
                        "n_tokens": int(attention_mask.sum()), "n_correct": 0}
        with tempfile.TemporaryDirectory() as directory:
            cfg = load_config(overrides=[f"train.output_dir={directory}", "train.cpu=true", "train.mixed_precision=no",
                 "train.epochs=3", "train.micro_batch_size=2", "train.grad_accum=1", "train.num_workers=0",
                 "train.eval_every=0", "train.save_every=0", "train.save_at_end=false", "train.drop_last=false",
                 "train.export_weights=false", "train.resume_from=null", "wandb.enabled=false"])
            dataset = TokenizedDataset([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
            examples, epochs = [], []
            train(cfg, model=Model(), train_ds=dataset, val_ds=dataset, pad_token_id=0,
                  batch_callback=lambda m, o, a, s, out, batch: examples.append(int(batch["input_ids"].shape[0])),
                  epoch_callback=lambda m, o, a, s: epochs.append((s.epoch + 1, s.global_step, s.batches_in_epoch)))
            self.assertEqual(epochs, [(1, 3, 3), (2, 6, 3), (3, 9, 3)])
            self.assertEqual(sum(examples), 15)
            self.assertFalse((Path(directory) / "checkpoints").exists())


if __name__ == "__main__":
    unittest.main()
