"""Three-epoch fresh-checkpoint VAE pilot with live progress and epoch metrics."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from scripts.run_small_diagnostic import load_rows, parameters_fp32, state_hash, write
from vae.data import VAECollator, TokenizedDataset
from vae.model import build_vae
from vae.pilot_metrics import TokenMetrics, reconstruction_values
from vae.reconstruction import autocast_for, reconstruct_rows
from vae.train import load_config, make_accelerator, save_checkpoint, train, vae_config

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/vae-pilot-1000q-v1"
FIELDS = ["epoch", "global_step", "train_examples", "train_tokens", "train_ce_loss", "train_kl_loss",
          "train_loss", "train_token_accuracy", "val_examples", "val_tokens", "val_ce_loss", "val_kl_loss",
          "val_loss", "val_token_accuracy", "val_exact_count", "val_exact_total", "val_exact_match_rate",
          "val_numeric_count", "val_numeric_total", "val_numeric_match_rate", "val_eos_count", "val_token_limit_count",
          "control_examples", "control_normal_exact_rate", "control_zero_exact_rate", "control_shuffled_exact_rate",
          "train_seconds", "validation_seconds", "checkpoint_seconds", "epoch_seconds", "learning_rate", "peak_allocated_gib"]


def configure():
    cfg = load_config(str(ROOT / "configs/vae_small_pilot.yaml"))
    cfg.wandb.enabled = False
    return cfg


def get_data():
    metadata = json.loads((DATA / "summary.json").read_text())
    assert metadata["train"]["questions"] == 1000 and metadata["dev"]["questions"] == 100
    rows = {split: load_rows(DATA / f"{split}.jsonl") for split in ("train", "dev")}
    assert not ({r["problem_id"] for r in rows["train"]} & {r["problem_id"] for r in rows["dev"]})
    assert all(r["input_ids"] and len(r["input_ids"]) <= 128 for split in rows.values() for r in split)
    return metadata, rows, load_rows(DATA / "controls.jsonl")


def tokenizer_for(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def fresh_model(cfg, device):
    model = build_vae(vae_config(cfg), pretrained=True, dtype=torch.bfloat16)
    parameters_fp32(model)
    return model.to(device)


def memory_check():
    cfg = configure()
    _, rows, _ = get_data()
    set_seed(42)
    torch.set_num_threads(4)
    tokenizer = tokenizer_for(cfg)
    model = fresh_model(cfg, "cuda:0")
    torch.cuda.reset_peak_memory_stats()
    before = state_hash(model.decoder_model)
    worst = sorted(rows["train"], key=lambda r: len(r["input_ids"]), reverse=True)[:cfg.train.micro_batch_size]
    batch = VAECollator(tokenizer.pad_token_id)(worst)
    batch = {k: v.to("cuda:0") for k, v in batch.items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.train.lr)
    model.train()
    started = time.time()
    with autocast_for(model):
        result = model(**batch, kl_weight=cfg.train.kl_weight)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert state_hash(model.decoder_model) == before
    longest_dev = sorted(rows["dev"], key=lambda r: len(r["input_ids"]), reverse=True)[:16]
    predictions = reconstruct_rows(model, tokenizer, longest_dev, max_new_tokens=160)
    report = {"passed": True, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "training_batch": len(worst), "training_sequence_length_including_eos": max(len(r["input_ids"]) for r in worst),
              "reconstruction_batch": len(predictions), "generation_limit": 160,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "device_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
              "seconds": time.time() - started, "decoder_unchanged": True,
              "probe_updates_discarded": True, "real_pilot_initialization": "fresh pretrained weights in a new process"}
    write(ROOT / "manifests/pilot_memory.json", report)
    print(json.dumps(report, indent=2), flush=True)


class Pilot:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "source").mkdir()
        self.start = time.time()
        self.state = {"pid": os.getpid(), "stage": "loading", "epoch": 0, "epochs": 3,
                      "batch": 0, "batches": 0, "percent": 0.0}
        self.update()
        self.cfg = configure()
        self.cfg.train.output_dir = str(directory / "training")
        self.metadata, self.rows, self.controls = get_data()
        self.batches = math.ceil(len(self.rows["train"]) / self.cfg.train.micro_batch_size)
        self.state.update({"batches": self.batches, "train_questions": 1000, "dev_questions": 100,
                           "train_sentences": len(self.rows["train"]), "dev_sentences": len(self.rows["dev"])})
        self.total_steps = 3 * self.batches
        OmegaConf.save(self.cfg, directory / "config.yaml")
        for path in [Path(__file__), ROOT / "vae/model.py", ROOT / "vae/train.py", ROOT / "vae/reconstruction.py",
                     ROOT / "vae/pilot_metrics.py", ROOT / "requirements-venv.lock.txt"]:
            shutil.copy2(path, directory / "source" / path.name)
        shutil.copy2(DATA / "summary.json", directory / "data-summary.json")
        self.metric_rows = []
        self.training = TokenMetrics()
        self.epoch_started = None
        write(directory / "acceptance_criteria.json", {
              "mechanical": ["all three epochs complete with finite losses", "decoder weights unchanged",
                             "every selected example included in each epoch", "saved checkpoint and export available"],
              "generalization_signal": "Best full-development exact reconstruction improves over the fresh baseline; fixed-subset normal codes outperform zero and shuffled codes",
              "numeric_metric": "Exact ordered numeric-string sequence on targets that contain numbers; no algebraic equivalence or problem-solving correctness implied",
              "model_selection": "highest full-dev exact-match rate, then lowest full-dev CE; test set never used",
              "automatic_extension": False, "initialization": "fresh pretrained small encoder/decoder, not the 16-example overfit model"})

    def update(self, **kwargs):
        self.state.update(kwargs)
        self.state["elapsed_seconds"] = round(time.time() - self.start, 2)
        self.state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(self.directory / "status.json", self.state)

    @torch.no_grad()
    def teacher_validation(self, model, epoch):
        started = time.time()
        model.eval()
        data = TokenizedDataset([r["input_ids"] for r in self.rows["dev"]])
        loader = DataLoader(data, batch_size=4, collate_fn=VAECollator(self.tokenizer.pad_token_id), shuffle=False)
        metric = TokenMetrics()
        for index, batch in enumerate(loader, 1):
            batch = {k: v.to(model.latents_query.device) for k, v in batch.items()}
            with autocast_for(model):
                output = model(**batch, kl_weight=self.cfg.train.kl_weight)
            assert torch.isfinite(output["loss"])
            metric.add(output, int(batch["input_ids"].shape[0]))
            if index % 10 == 0 or index == len(loader):
                current = metric.values(self.cfg.train.kl_weight)
                self.update(stage="validation_tokens", epoch=epoch, evaluation_completed=current["examples"],
                            evaluation_total=len(data), val_token_accuracy=current["token_accuracy"], val_loss=current["loss"])
        return metric.values(self.cfg.train.kl_weight)

    @torch.no_grad()
    def reconstruct(self, model, rows, epoch, label, control="normal"):
        model.eval()
        epoch_dir = self.directory / f"epoch_{epoch:02d}"
        epoch_dir.mkdir(exist_ok=True)
        ordered = sorted(rows, key=lambda r: len(r["input_ids"]), reverse=True)
        all_records = []
        started = time.time()
        with (epoch_dir / f"{label}-{control}.jsonl").open("w") as stream:
            for index in range(0, len(ordered), 16):
                chunk = ordered[index:index + 16]
                records = reconstruct_rows(model, self.tokenizer, chunk, control=control, max_new_tokens=160)
                all_records.extend(records)
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                current = reconstruction_values(all_records)
                self.update(stage=f"reconstruct_{label}_{control}", epoch=epoch,
                            evaluation_completed=len(all_records), evaluation_total=len(rows),
                            evaluation_exact=current["exact_count"],
                            evaluation_eta_seconds=(time.time() - started) / len(all_records) * (len(rows) - len(all_records)))
                if index % 80 == 0 or len(all_records) == len(rows):
                    print(f"EVAL epoch={epoch} {label}/{control}: {len(all_records)}/{len(rows)}, exact={current['exact_count']}", flush=True)
        return reconstruction_values(all_records), all_records

    def append_metrics(self, row):
        self.metric_rows.append(row)
        temporary = self.directory / "metrics.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({field: r.get(field) for field in FIELDS} for r in self.metric_rows)
        temporary.replace(self.directory / "metrics.csv")

    def evaluate_epoch(self, model, epoch, step, training_values=None, train_seconds=0):
        began = time.time()
        self.update(epoch=epoch, stage="validation_tokens", evaluation_completed=0,
                    evaluation_total=len(self.rows["dev"]))
        token_values = self.teacher_validation(model, epoch)
        normal, records = self.reconstruct(model, self.rows["dev"], epoch, "dev")
        control_values = {}
        for control in ("normal", "zero", "shuffled"):
            control_values[control], _ = self.reconstruct(model, self.controls, epoch, "controls", control)
        epoch_dir = self.directory / f"epoch_{epoch:02d}"
        write(epoch_dir / "reconstruction_examples.json", records[:20])
        row = {"epoch": epoch, "global_step": step, "val_examples": token_values["examples"],
               "val_tokens": token_values["tokens"], "val_ce_loss": token_values["ce_loss"],
               "val_kl_loss": token_values["kl_loss"], "val_loss": token_values["loss"],
               "val_token_accuracy": token_values["token_accuracy"],
               "val_exact_count": normal["exact_count"], "val_exact_total": normal["total"],
               "val_exact_match_rate": normal["exact_match_rate"], "val_numeric_count": normal["numeric_count"],
               "val_numeric_total": normal["numeric_total"], "val_numeric_match_rate": normal["numeric_match_rate"],
               "val_eos_count": normal["eos_count"], "val_token_limit_count": normal["token_limit_count"],
               "control_examples": len(self.controls),
               "control_normal_exact_rate": control_values["normal"]["exact_match_rate"],
               "control_zero_exact_rate": control_values["zero"]["exact_match_rate"],
               "control_shuffled_exact_rate": control_values["shuffled"]["exact_match_rate"],
               "train_seconds": train_seconds, "validation_seconds": time.time() - began,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
        if training_values:
            row.update({"train_examples": training_values["examples"], "train_tokens": training_values["tokens"],
                        "train_ce_loss": training_values["ce_loss"], "train_kl_loss": training_values["kl_loss"],
                        "train_loss": training_values["loss"], "train_token_accuracy": training_values["token_accuracy"]})
        write(epoch_dir / "evaluation.json", row)
        print(f"EPOCH {epoch} VALIDATION: loss={row['val_loss']:.5f}, token_acc={row['val_token_accuracy']:.2%}, exact={normal['exact_count']}/{normal['total']}, numeric={normal['numeric_count']}/{normal['numeric_total']}", flush=True)
        return row

    def batch_callback(self, model, optimizer, accelerator, state, output, batch):
        if self.epoch_started is None:
            self.epoch_started = time.time()
        self.training.add(output, int(batch["input_ids"].shape[0]))
        assert torch.isfinite(output["loss"])
        index = state.batches_in_epoch
        if index % 5 == 0 or index == 1 or index == self.batches:
            current = self.training.values(self.cfg.train.kl_weight)
            elapsed = time.time() - self.epoch_started
            eta = elapsed / max(index, 1) * (self.batches - index)
            self.update(stage="training", epoch=state.epoch + 1, batch=index, batches=self.batches,
                        percent=100 * index / self.batches, global_step=state.global_step + int(accelerator.sync_gradients),
                        train_loss=current["loss"], train_token_accuracy=current["token_accuracy"],
                        learning_rate=optimizer.param_groups[0]["lr"], epoch_training_eta_seconds=eta,
                        evaluation_completed=0, evaluation_total=0)
            if index % 20 == 0 or index == 1 or index == self.batches:
                print(f"TRAIN epoch={state.epoch + 1}/3 batch={index}/{self.batches} ({100 * index / self.batches:.1f}%) loss={current['loss']:.5f} token_acc={current['token_accuracy']:.2%} epoch_training_eta={eta:.0f}s", flush=True)

    def epoch_callback(self, wrapped, optimizer, accelerator, state):
        epoch = state.epoch + 1
        assert self.training.examples == len(self.rows["train"])
        train_seconds = time.time() - self.epoch_started
        model = accelerator.unwrap_model(wrapped)
        assert state_hash(model.decoder_model) == self.decoder_hash
        row = self.evaluate_epoch(model, epoch, state.global_step, self.training.values(self.cfg.train.kl_weight), train_seconds)
        self.update(stage="saving_checkpoint", epoch=epoch, percent=100,
                    val_loss=row["val_loss"], val_token_accuracy=row["val_token_accuracy"],
                    val_exact_count=row["val_exact_count"], val_exact_total=row["val_exact_total"])
        began = time.time()
        checkpoint = save_checkpoint(accelerator, wrapped, self.cfg, state)
        row.update({"checkpoint_seconds": time.time() - began,
                    "epoch_seconds": train_seconds + row["validation_seconds"] + time.time() - began,
                    "learning_rate": optimizer.param_groups[0]["lr"]})
        write(self.directory / f"epoch_{epoch:02d}" / "metrics.json", row)
        self.append_metrics(row)
        completed = [r for r in self.metric_rows if r["epoch"] > 0]
        best = max(completed, key=lambda r: (r["val_exact_match_rate"], -r["val_ce_loss"]))
        write(self.directory / "best_checkpoint.json", {"epoch": best["epoch"], "global_step": best["global_step"],
              "val_exact_match_rate": best["val_exact_match_rate"], "val_ce_loss": best["val_ce_loss"],
              "checkpoint": str(self.directory / "training/checkpoints" / f"step_{best['global_step']}"),
              "exported_weights": str(self.directory / "training/weights" / f"step_{best['global_step']}.safetensors")})
        print(f"EPOCH {epoch}/3 COMPLETE: train_token_acc={row['train_token_accuracy']:.2%} val_token_acc={row['val_token_accuracy']:.2%} val_exact={row['val_exact_count']}/{row['val_exact_total']} checkpoint={checkpoint}", flush=True)
        model.train()
        self.training = TokenMetrics()
        self.epoch_started = time.time()

    def run(self):
        assert json.loads((ROOT / "manifests/pilot_memory.json").read_text())["passed"]
        set_seed(42)
        torch.set_num_threads(4)
        self.accelerator, _ = make_accelerator(self.cfg)
        assert self.accelerator.num_processes == 1
        self.tokenizer = tokenizer_for(self.cfg)
        model = fresh_model(self.cfg, self.accelerator.device)
        torch.cuda.reset_peak_memory_stats()
        self.decoder_hash = state_hash(model.decoder_model)
        write(self.directory / "model_setup.json", {"decoder_sha256": self.decoder_hash,
              "parameter_counts": model.trainable_summary(), "trainable_dtype": "float32", "compute_dtype": "bfloat16",
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": torch.cuda.get_device_name(),
              "initialization": "fresh revision-pinned Qwen3-0.6B-Base encoder and decoder"})
        baseline = self.evaluate_epoch(model, 0, 0)
        self.append_metrics(baseline)
        self.update(stage="training", epoch=1, batch=0, percent=0.0, evaluation_completed=0, evaluation_total=0)
        train_ds = TokenizedDataset([r["input_ids"] for r in self.rows["train"]])
        val_ds = TokenizedDataset([r["input_ids"] for r in self.rows["dev"]])
        self.epoch_started = time.time()
        result = train(self.cfg, model=model, train_ds=train_ds, val_ds=val_ds,
                       pad_token_id=self.tokenizer.pad_token_id, accelerator=self.accelerator,
                       batch_callback=self.batch_callback, epoch_callback=self.epoch_callback)
        completed = [r for r in self.metric_rows if r["epoch"] > 0]
        assert len(completed) == 3
        assert all(r["train_examples"] == len(train_ds) and r["val_examples"] == len(val_ds) for r in completed)
        best = max(completed, key=lambda r: (r["val_exact_match_rate"], -r["val_ce_loss"]))
        signal = (best["val_exact_match_rate"] > baseline["val_exact_match_rate"] and
                  best["control_normal_exact_rate"] > max(best["control_zero_exact_rate"], best["control_shuffled_exact_rate"]))
        summary = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "epochs": 3,
                   "global_steps": result["state"].global_step, "baseline": baseline, "epoch_metrics": completed,
                   "best_epoch": best["epoch"], "generalization_signal": signal, "seconds": time.time() - self.start,
                   "test_used": False, "full_size_training_started": False, "decoder_unchanged": True}
        write(self.directory / "summary.json", summary)
        self.update(stage="completed", epoch=3, percent=100, best_epoch=best["epoch"], generalization_signal=signal)
        print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--memory-check", action="store_true")
    args = parser.parse_args()
    if args.memory_check:
        memory_check()
        return
    if not args.directory:
        parser.error("--directory is required")
    directory = args.directory.resolve()
    try:
        Pilot(directory).run()
    except Exception as error:
        directory.mkdir(parents=True, exist_ok=True)
        write(directory / "status.json", {"stage": "failed", "pid": os.getpid(), "error": str(error)})
        raise


if __name__ == "__main__":
    main()
