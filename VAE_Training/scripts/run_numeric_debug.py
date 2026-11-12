"""Bounded lower-LR continuation of the fixed 16-sentence codec diagnostic.

Writes live status.json for monitoring. All original artifacts are read-only.
No test-set examples or output correction/postprocessing are used.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
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
from transformers import AutoTokenizer

from scripts.run_small_diagnostic import (
    load_rows, mean_codes, parameters_fp32, reconstruction_report, state_hash, write,
)
from vae.data import VAECollator, build_datasets
from vae.model import build_vae, load_state_file
from vae.reconstruction import autocast_for, reconstruct_rows
from vae.train import load_config, make_accelerator, train, vae_config

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@torch.no_grad()
def inspect_numeric_choice(model, tokenizer, selections, source_records):
    """Teacher-forced probabilities at the old free-decoding divergence.

    This is an analysis only. Ground-truth tokens never enter the free decoder.
    """
    batch = VAECollator(tokenizer.pad_token_id)([{"input_ids": row["input_ids"]} for row in selections])
    batch = {k: v.to(model.latents_query.device) for k, v in batch.items()}
    was_training = model.training
    model.eval()
    try:
        with autocast_for(model):
            mean, _ = model.encode(batch["input_ids"], batch["attention_mask"])
            hidden = model.decoder_hidden(mean, batch["input_ids"], batch["attention_mask"])
            records = []
            for i, (row, original) in enumerate(zip(selections, source_records)):
                if original["exact_match"]:
                    continue
                target = row["input_ids"]
                generated = original["generated_token_ids"]
                divergence = next((j for j, (a, b) in enumerate(zip(target, generated)) if a != b), None)
                if divergence is None:
                    continue
                scores = model.decoder_model.lm_head(hidden[i:i + 1, divergence]).float()[0]
                probs = scores.softmax(-1)
                top_values, top_ids = probs.topk(5)
                records.append({"parent_id": row["parent_id"], "position": divergence,
                     "target": row["reasoning_block"], "old_reconstruction": original["text"],
                     "expected_token": tokenizer.decode([target[divergence]]),
                     "old_generated_token": tokenizer.decode([generated[divergence]]),
                     "expected_token_probability": float(probs[target[divergence]]),
                     "old_generated_token_probability": float(probs[generated[divergence]]),
                     "top5": [{"token": tokenizer.decode([int(t)]), "probability": float(p)} for p, t in zip(top_values, top_ids)],
                     "scope": "Teacher-forced probability at the saved divergence; not output rewriting"})
        return records
    finally:
        model.train(was_training)


def run(source, directory, additional_steps):
    previous = json.loads((source / "summary.json").read_text())
    assert json.loads((source / "status.json").read_text())["stage"] == "completed"
    assert previous["optimizer_steps"] == 500
    assert 0 < additional_steps <= 300
    directory.mkdir(parents=True, exist_ok=False)
    start = time.time()
    original_summary_hash = checksum(source / "summary.json")
    status = {"pid": os.getpid(), "stage": "loading", "source_step": 500,
              "additional_step": 0, "max_additional_steps": additional_steps,
              "learning_rate": 2e-5, "train_exact_last_probe": 15,
              "source_directory": str(source), "directory": str(directory)}
    def update(**values):
        status.update(values)
        status["elapsed_seconds"] = round(time.time() - start, 2)
        write(directory / "status.json", status)
    update()
    (directory / "source").mkdir()
    for path in [Path(__file__), ROOT / "vae/train.py", ROOT / "vae/model.py", ROOT / "vae/reconstruction.py"]:
        shutil.copy2(path, directory / "source" / path.name)
    for name in ["train.jsonl", "dev.jsonl", "diagnostic.vae.json", "selection.json", "requirements-venv.lock.txt"]:
        shutil.copy2(source / name, directory / name)
    checkpoint = source / "training/checkpoints/step_500"
    cfg = load_config(str(source / "config.yaml"))
    cfg.data.path = str(directory / "diagnostic.vae.json")
    cfg.train.output_dir = str(directory / "training")
    cfg.train.resume_from = str(checkpoint)
    cfg.train.max_steps = 500 + additional_steps
    cfg.train.epochs = 1000
    cfg.train.lr = 2e-5
    cfg.train.resume_lr_override = 2e-5
    cfg.train.save_every = 0
    cfg.train.keep_last = 1
    OmegaConf.save(cfg, directory / "config.yaml")
    write(directory / "run_manifest.json", {"source_checkpoint": str(checkpoint),
          "source_summary_sha256": original_summary_hash, "initial_global_step": 500,
          "max_additional_steps": additional_steps, "new_learning_rate": 2e-5,
          "old_learning_rate": 1e-4, "probe_interval": 25,
          "early_stop": "16/16 train exact with EOS", "official_test_used": False,
          "data_fingerprints": {name: checksum(directory / name) for name in ["train.jsonl", "dev.jsonl"]}})
    torch.set_num_threads(4)
    set_seed(42)
    accelerator, _ = make_accelerator(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    selections = {split: load_rows(directory / f"{split}.jsonl") for split in ("train", "dev")}
    train_ds, dev_ds = build_datasets(cfg.data.path, tokenizer, cfg.data.max_tokens, language="Math")
    # Inspect the exact exported step-500 weights before training. The trainer
    # subsequently restores the complete optimizer/scheduler/RNG checkpoint.
    model = build_vae(vae_config(cfg), pretrained=False, dtype=torch.bfloat16)
    parameters_fp32(model)
    model.load_vae_state(load_state_file(str(source / "training/weights/step_500.safetensors")))
    model.to(accelerator.device)
    torch.cuda.reset_peak_memory_stats()
    original_decoder_hash = json.loads((source / "model_setup.json").read_text())["decoder_initial_sha256"]
    assert state_hash(model.decoder_model) == original_decoder_hash
    old_reconstructions = load_rows(source / "after-500-train-normal.jsonl")
    before = inspect_numeric_choice(model, tokenizer, selections["train"], old_reconstructions)
    write(directory / "numeric-choice-before.json", before)
    print("Numerical divergence before:", json.dumps(before), flush=True)
    probes = []
    def callback(wrapped, optimizer, acc, state, output):
        assert torch.isfinite(output["loss"])
        actual_lr = optimizer.param_groups[0]["lr"]
        assert abs(actual_lr - 2e-5) < 1e-12, actual_lr
        additional = state.global_step - 500
        update(stage="training", global_step=state.global_step, additional_step=additional,
               learning_rate=actual_lr, loss=float(output["loss"].detach()))
        if additional % 25 != 0 and state.global_step != cfg.train.max_steps:
            return False
        update(stage="checking_reconstruction")
        unwrapped = acc.unwrap_model(wrapped)
        records = reconstruct_rows(unwrapped, tokenizer, selections["train"])
        exact = sum(r["exact_match"] for r in records)
        ended = sum(r["eos_reached"] for r in records)
        with (directory / f"probe-step-{state.global_step}.jsonl").open("w") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        probes.append({"global_step": state.global_step, "additional_step": additional,
                       "train_exact": exact, "eos_reached": ended, "learning_rate": actual_lr})
        write(directory / "probes.json", probes)
        update(stage="training", train_exact_last_probe=exact, last_probe_step=state.global_step)
        print(f"FREE RECONSTRUCTION step={state.global_step}: train exact={exact}/16, EOS={ended}/16, lr={actual_lr}", flush=True)
        passed = exact == 16 and ended == 16
        if passed:
            update(stage="saving_checkpoint", early_stop_reason="16/16 exact reconstruction")
        return passed
    result = train(cfg, model=model, train_ds=train_ds, val_ds=dev_ds,
                   pad_token_id=tokenizer.pad_token_id, accelerator=accelerator, step_callback=callback)
    update(stage="evaluating", global_step=result["state"].global_step,
           additional_step=result["state"].global_step - 500)
    underlying = accelerator.unwrap_model(result["model"])
    assert state_hash(underlying.decoder_model) == original_decoder_hash
    final = reconstruction_report(underlying, tokenizer, selections, directory, "final", ["normal", "zero", "shuffled"])
    after = inspect_numeric_choice(underlying, tokenizer, selections["train"], old_reconstructions)
    write(directory / "numeric-choice-after.json", after)
    write(directory / "history.json", result["history"])
    update(stage="verifying_export")
    step = result["state"].global_step
    exported = directory / "training/weights" / f"step_{step}.safetensors"
    mean_before = mean_codes(underlying, tokenizer, selections["train"][:2])
    restored = build_vae(vae_config(cfg), pretrained=False, dtype=torch.bfloat16)
    parameters_fp32(restored)
    restored.load_vae_state(load_state_file(str(exported)))
    restored.to(accelerator.device)
    assert torch.equal(mean_before, mean_codes(restored, tokenizer, selections["train"][:2]))
    assert state_hash(restored.decoder_model) == original_decoder_hash
    matches = {}
    for split, rows in selections.items():
        reloaded = reconstruct_rows(restored, tokenizer, rows)
        saved = load_rows(directory / f"final-{split}-normal.jsonl")
        matches[split] = sum(a["generated_token_ids"] == b["generated_token_ids"] for a, b in zip(reloaded, saved))
        assert matches[split] == 16
    assert checksum(source / "summary.json") == original_summary_hash
    summary = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
          "source_global_step": 500, "final_global_step": step, "additional_steps": step - 500,
          "learning_rate": 2e-5, "original_learning_rate": 1e-4,
          "after": final, "probes": probes,
          "overfit_gate_passed": final["train"]["normal"]["exact"] == 16,
          "decoder_backbone_unchanged": True, "fresh_reload_token_sequences_identical": matches,
          "original_run_preserved": True, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
          "seconds": time.time() - start, "larger_pilot_started": False,
          "scope": "Lower-learning-rate numerical preservation debug on the same 16 training sentences"}
    write(directory / "summary.json", summary)
    update(stage="completed", summary=str(directory / "summary.json"), **{
        "train_exact": final["train"]["normal"]["exact"],
        "dev_exact": final["dev"]["normal"]["exact"],
        "overfit_gate_passed": summary["overfit_gate_passed"]})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/vae-small-diagnostic-20261112-01")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--additional-steps", type=int, default=300)
    args = parser.parse_args()
    try:
        run(args.source.resolve(), args.directory.resolve(), args.additional_steps)
    except Exception as error:
        args.directory.mkdir(parents=True, exist_ok=True)
        write(args.directory / "status.json", {"stage": "failed", "pid": os.getpid(), "error": str(error)})
        raise


if __name__ == "__main__":
    main()
