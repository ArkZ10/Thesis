from pathlib import Path
import tempfile
import unittest

import torch

from vae.data import TokenizedDataset
from vae.train import load_config, override_resume_lr, train


class TrainingControlTests(unittest.TestCase):
    def test_resume_lr_survives_scheduler_step(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        optimizer.step()
        scheduler.step()
        state = scheduler.state_dict()
        optimizer.param_groups[0]["lr"] = 1e-4
        scheduler.load_state_dict(state)
        override_resume_lr(optimizer, scheduler, 2e-5)
        self.assertEqual(optimizer.param_groups[0]["lr"], 2e-5)
        optimizer.step()
        scheduler.step()
        self.assertEqual(scheduler.get_last_lr(), [2e-5])
        self.assertEqual(optimizer.param_groups[0]["lr"], 2e-5)
        with self.assertRaises(ValueError):
            override_resume_lr(optimizer, scheduler, -1)

    def test_early_stop_saves_actual_step(self):
        class ToyModel(torch.nn.Module):
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
            cfg = load_config(overrides=[f"train.output_dir={directory}", "train.cpu=true",
                 "train.mixed_precision=no", "train.max_steps=8", "train.epochs=10",
                 "train.micro_batch_size=2", "train.grad_accum=1", "train.num_workers=0",
                 "train.eval_every=0", "train.save_every=0", "train.export_weights=false",
                 "train.resume_from=null", "wandb.enabled=false"])
            dataset = TokenizedDataset([[1, 2], [3, 4], [5, 6], [7, 8]])
            result = train(cfg, model=ToyModel(), train_ds=dataset, val_ds=dataset, pad_token_id=0,
                           step_callback=lambda model, opt, acc, state, out: state.global_step == 2)
            self.assertEqual(result["state"].global_step, 2)
            self.assertTrue((Path(directory) / "checkpoints/step_2").is_dir())
            self.assertFalse((Path(directory) / "checkpoints/step_8").exists())


if __name__ == "__main__":
    unittest.main()
