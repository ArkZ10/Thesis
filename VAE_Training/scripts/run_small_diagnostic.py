"""Real-checkpoint, 16-sentence VAE overfit test with a frozen decoder.

Preparatory mode makes the exact train/dev selection reviewable. Run mode uses
the copied Accelerate trainer with FP32 trainable parameters and BF16 compute.
The initial 400-step budget is intentionally separate from any larger pilot.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from vae.data import VAECollator, build_datasets
from vae.model import build_vae, load_state_file
from vae.reconstruction import autocast_for, reconstruct_rows
from vae.train import evaluate, load_config, make_accelerator, save_checkpoint, train, vae_config

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def load_rows(path):
    return [json.loads(line) for line in path.open()]


def arithmetic(expression):
    node = ast.parse(expression, mode="eval").body
    def walk(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return Fraction(str(n.value))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(n.op, ast.USub) else 1) * walk(n.operand)
        if isinstance(n, ast.BinOp):
            left, right = walk(n.left), walk(n.right)
            if isinstance(n.op, ast.Add): return left + right
            if isinstance(n.op, ast.Sub): return left - right
            if isinstance(n.op, ast.Mult): return left * right
            if isinstance(n.op, ast.Div): return left / right
            if isinstance(n.op, ast.Pow) and right.denominator == 1 and abs(right) <= 10: return left ** int(right)
        raise ValueError("Unsupported arithmetic annotation")
    return walk(node)


def annotations_correct(text):
    annotations = re.findall(r"<<([^<>]+)>>", text)
    if not annotations:
        return False
    try:
        for annotation in annotations:
            parts = annotation.split("=")
            if len(parts) != 2 or arithmetic(parts[0]) != arithmetic(parts[1]):
                return False
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return False
    return True


def prepare(directory):
    checkpoint = json.loads((ROOT / "manifests/qwen3_0_6b_base.json").read_text())
    assert checkpoint["all_checksums_verified"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["directory"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepared = ROOT / "data/processed/math-sentences-v1"
    selections, provenance = {}, {}
    for split in ("train", "dev"):
        selected, used_questions, filters = [], set(), Counter()
        source = prepared / "sentences" / f"gsm8k.{split}.jsonl"
        for line in source.open():
            row = json.loads(line)
            text = row["reasoning_block"]
            filters["rows_scanned"] += 1
            ids = tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            if row["problem_id"] in used_questions:
                filters["repeated_question"] += 1
                continue
            if not 8 <= len(ids) <= 40 or "\n" in text:
                filters["length_or_multiline"] += 1
                continue
            if not annotations_correct(text):
                filters["no_verified_calculator_annotation"] += 1
                continue
            used_questions.add(row["problem_id"])
            selected.append({**row, "input_ids": ids, "token_count_including_eos": len(ids),
                             "calculator_annotations_checked": True})
            if len(selected) == 16:
                break
        assert len(selected) == 16
        selections[split] = selected
        provenance[split] = {"source": str(source), "scan_statistics": dict(filters), "selected": 16}
        with (directory / f"{split}.jsonl").open("w") as stream:
            for row in selected:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    assert not ({r["problem_id"] for r in selections["train"]} & {r["problem_id"] for r in selections["dev"]})
    write(directory / "diagnostic.vae.json", {"format": "vae_sentence_splits_v1", "language": "Math",
          "train": ["train.jsonl"], "validation": ["dev.jsonl"], "test": []})
    write(directory / "selection.json", {"model": checkpoint["repository"], "revision": checkpoint["revision"],
          "provenance": provenance, "selection": "First 16 qualifying distinct questions per original split",
          "official_test_data_used": False, "token_truncation": False,
          "annotation_policy": "Exact rational arithmetic for every calculator annotation; not full proof certification"})
    write(directory / "status.json", {"stage": "prepared", "training_started": False})
    return selections


def parameters_fp32(model):
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()


def state_hash(module):
    result, seen = hashlib.sha256(), set()
    for name, parameter in sorted(module.named_parameters()):
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        result.update(name.encode())
        result.update(parameter.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def mean_codes(model, tokenizer, rows):
    batch = VAECollator(tokenizer.pad_token_id)([{"input_ids": r["input_ids"]} for r in rows])
    batch = {k: v.to(model.latents_query.device) for k, v in batch.items()}
    was_training = model.training
    model.eval()
    with torch.no_grad(), autocast_for(model):
        means, _ = model.encode(batch["input_ids"], batch["attention_mask"])
    model.train(was_training)
    return means.detach().cpu()


def reconstruction_report(model, tokenizer, selections, directory, label, controls):
    report = {}
    for split, rows in selections.items():
        report[split] = {}
        for control in controls:
            records = reconstruct_rows(model, tokenizer, rows, control=control, max_new_tokens=64)
            with (directory / f"{label}-{split}-{control}.jsonl").open("w") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            report[split][control] = {"exact": sum(r["exact_match"] for r in records), "total": len(records),
                                     "eos_reached": sum(r["eos_reached"] for r in records),
                                     "token_limit_reached": sum(r["token_limit_reached"] for r in records)}
    write(directory / f"{label}-reconstruction.json", report)
    print(label, json.dumps(report), flush=True)
    return report


def verify_export(underlying, tokenizer, rows, cfg, exported, device):
    means = mean_codes(underlying, tokenizer, rows[:2])
    restored = build_vae(vae_config(cfg), pretrained=False, dtype=torch.bfloat16)
    parameters_fp32(restored)
    restored.load_vae_state(load_state_file(str(exported)))
    restored.to(device)
    restored_means = mean_codes(restored, tokenizer, rows[:2])
    assert torch.equal(means, restored_means), "Exported VAE reload changed posterior means"


def run(directory):
    began = time.time()
    assert json.loads((directory / "status.json").read_text())["stage"] == "prepared"
    write(directory / "status.json", {"stage": "loading", "training_started": False})
    cfg = load_config(str(ROOT / "configs/vae_small_diagnostic.yaml"))
    cfg.data.path = str(directory / "diagnostic.vae.json")
    cfg.train.output_dir = str(directory / "training")
    OmegaConf.save(cfg, directory / "config.yaml")
    (directory / "source").mkdir()
    for path in [ROOT / "vae/model.py", ROOT / "vae/train.py", ROOT / "vae/reconstruction.py", Path(__file__)]:
        shutil.copy2(path, directory / "source" / path.name)
    shutil.copy2(ROOT / "requirements-venv.lock.txt", directory / "requirements-venv.lock.txt")
    torch.set_num_threads(4)
    set_seed(42)
    accelerator, _ = make_accelerator(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    selections = {split: load_rows(directory / f"{split}.jsonl") for split in ("train", "dev")}
    train_ds, val_ds = build_datasets(cfg.data.path, tokenizer, cfg.data.max_tokens, language="Math")
    model = build_vae(vae_config(cfg), dtype=torch.bfloat16)
    parameters_fp32(model)
    model.to(accelerator.device)
    torch.cuda.reset_peak_memory_stats()
    before_decoder = state_hash(model.decoder_model)
    anchor = model.encoder_model.model.layers[0].self_attn.q_proj.weight
    before_anchor = anchor.detach().flatten()[:64].cpu().clone()
    write(directory / "model_setup.json", {"parameter_counts": model.trainable_summary(),
          "decoder_initial_sha256": before_decoder, "latent_shape_per_sentence": [6, cfg.model.latent_dim],
          "trainable_parameter_dtype": "float32", "frozen_decoder_dtype": "bfloat16",
          "compute_dtype": "bfloat16", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
          "gpu": torch.cuda.get_device_name(), "pytorch": torch.__version__})
    baseline = reconstruction_report(model, tokenizer, selections, directory, "before", ["normal"])
    hooks, gradients = [], {}
    watched = {"encoder": anchor, "queries": model.latents_query, "mean": model.fc_mu.weight,
               "variance": model.fc_var.weight, "decoder_interface": model.latent_to_decoder.weight}
    for name, parameter in watched.items():
        def observe(gradient, name=name):
            gradients[name] = {"finite": bool(torch.isfinite(gradient).all()), "nonzero": bool(gradient.abs().sum() > 0)}
        hooks.append(parameter.register_hook(observe))

    def callback(wrapped, optimizer, acc, state, out):
        write(directory / "status.json", {"stage": "training", "training_started": True,
              "step": state.global_step, "max_steps": cfg.train.max_steps,
              "loss": float(out["loss"].detach()), "elapsed_seconds": round(time.time() - began, 2)})
        if state.global_step != 1:
            return
        underlying = acc.unwrap_model(wrapped)
        assert all(v["finite"] and v["nonzero"] for v in gradients.values()) and set(gradients) == set(watched)
        assert not torch.equal(before_anchor, anchor.detach().flatten()[:64].cpu())
        assert all(not p.requires_grad and p.grad is None for p in underlying.decoder_model.parameters())
        assert state_hash(underlying.decoder_model) == before_decoder
        for handle in hooks:
            handle.remove()
        mean_before = mean_codes(underlying, tokenizer, selections["train"][:2])
        cfg.train.export_weights = False
        checkpoint = save_checkpoint(acc, wrapped, cfg, state)
        cfg.train.export_weights = True
        with torch.no_grad():
            underlying.fc_mu.bias.add_(0.5)
        acc.load_state(checkpoint)
        mean_after = mean_codes(underlying, tokenizer, selections["train"][:2])
        assert torch.equal(mean_before, mean_after), "Checkpoint reload changed posterior means"
        write(directory / "one_update.json", {"passed": True, "step": 1, "gradient_groups": gradients,
              "encoder_weights_changed": True, "decoder_unchanged": True, "checkpoint": checkpoint,
              "checkpoint_reload_means_bit_identical": True,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
        print("One-update gradient/freeze/checkpoint checks passed", flush=True)

    result = train(cfg, model=model, train_ds=train_ds, val_ds=val_ds,
                   pad_token_id=tokenizer.pad_token_id, accelerator=accelerator, step_callback=callback)
    underlying = accelerator.unwrap_model(result["model"])
    assert state_hash(underlying.decoder_model) == before_decoder
    final = reconstruction_report(underlying, tokenizer, selections, directory, "after", ["normal", "zero", "shuffled"])
    write(directory / "history.json", result["history"])
    exported = directory / "training/weights" / f"step_{result['state'].global_step}.safetensors"
    verify_export(underlying, tokenizer, selections["train"], cfg, exported, accelerator.device)
    summary = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "model": "Qwen/Qwen3-0.6B-Base",
               "train_sentences": 16, "dev_sentences": 16, "optimizer_steps": result["state"].global_step,
               "before": baseline, "after": final, "one_update_passed": True,
               "decoder_backbone_unchanged": True, "exported_weights_reload_means_bit_identical": True,
               "overfit_gate_passed": final["train"]["normal"]["exact"] == 16,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "seconds": time.time() - began,
               "scope": "16-sentence small-model codec diagnostic; not the full-size model or 1000-question pilot"}
    write(directory / "summary.json", summary)
    write(directory / "status.json", {"stage": "completed", **summary})
    print(json.dumps(summary, indent=2), flush=True)


def continue_run(directory, total_steps):
    """Bounded continuation of the same diagnostic, preserving its 400-step result."""
    previous = json.loads((directory / "summary.json").read_text())
    assert json.loads((directory / "status.json").read_text())["stage"] == "completed"
    assert previous["optimizer_steps"] < total_steps <= 500
    previous_step = previous["optimizer_steps"]
    write(directory / f"summary-{previous_step}.json", previous)
    began = time.time()
    cfg = load_config(str(directory / "config.yaml"))
    cfg.train.max_steps = total_steps
    cfg.train.keep_last = 3
    cfg.train.resume_from = str(directory / "training/checkpoints" / f"step_{previous_step}")
    OmegaConf.save(cfg, directory / "continuation-config.yaml")
    shutil.copy2(Path(__file__), directory / "source" / f"runner-to-{total_steps}.py")
    torch.set_num_threads(4)
    set_seed(42)
    accelerator, _ = make_accelerator(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    selections = {split: load_rows(directory / f"{split}.jsonl") for split in ("train", "dev")}
    train_ds, val_ds = build_datasets(cfg.data.path, tokenizer, cfg.data.max_tokens, language="Math")
    model = build_vae(vae_config(cfg), pretrained=False, dtype=torch.bfloat16)
    parameters_fp32(model)
    model.to(accelerator.device)
    torch.cuda.reset_peak_memory_stats()
    def callback(wrapped, optimizer, acc, state, out):
        assert torch.isfinite(out["loss"])
        write(directory / "status.json", {"stage": "training", "step": state.global_step,
              "max_steps": total_steps, "resumed_from": previous_step,
              "loss": float(out["loss"].detach()), "additional_seconds": time.time() - began})
    result = train(cfg, model=model, train_ds=train_ds, val_ds=val_ds,
                   pad_token_id=tokenizer.pad_token_id, accelerator=accelerator, step_callback=callback)
    underlying = accelerator.unwrap_model(result["model"])
    setup = json.loads((directory / "model_setup.json").read_text())
    assert state_hash(underlying.decoder_model) == setup["decoder_initial_sha256"]
    final = reconstruction_report(underlying, tokenizer, selections, directory,
                                  f"after-{total_steps}", ["normal", "zero", "shuffled"])
    exported = directory / "training/weights" / f"step_{total_steps}.safetensors"
    verify_export(underlying, tokenizer, selections["train"], cfg, exported, accelerator.device)
    write(directory / f"history-{total_steps}.json", result["history"])
    summary = {**previous, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "optimizer_steps": total_steps, "additional_steps": total_steps - previous_step,
               "resumed_from_step": previous_step, "after": final,
               "overfit_gate_passed": final["train"]["normal"]["exact"] == 16,
               "peak_allocated_gib": max(previous["peak_allocated_gib"], torch.cuda.max_memory_allocated() / 2**30),
               "seconds": previous["seconds"] + time.time() - began,
               "exported_weights_reload_means_bit_identical": True}
    write(directory / "summary.json", summary)
    write(directory / "status.json", {"stage": "completed", **summary})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--continue-to", type=int)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.prepare_only:
        directory.mkdir(parents=True, exist_ok=False)
        prepare(directory)
        print("Prepared", directory, flush=True)
        return
    try:
        if args.continue_to is not None:
            continue_run(directory, args.continue_to)
        else:
            run(directory)
    except Exception as error:
        write(directory / "status.json", {"stage": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
