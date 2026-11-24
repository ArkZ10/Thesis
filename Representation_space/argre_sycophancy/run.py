"""Run the released ARGRE reward trainer on frozen LLaDA answer activations."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
import wandb
import yaml

HERE = Path(__file__).resolve().parent
BROKENMATH = HERE.parents[1] / "Sycophancy-benchmark-test/BrokenMath/LLaDA-broken-math/results"
ARGRE = Path("/root/ARGRE/ARGRE")
for folder in ("collect_hidden", "train_hidden"):
    sys.path.insert(0, str(ARGRE / "evaluation" / folder))

from pca import PCA
from rm_trainer import RMTrainer


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_pairs():
    chosen = json.loads((HERE / "counterexamples.json").read_text())
    saved = {row["problem_id"]: row for row in json.loads((BROKENMATH / "answers.json").read_text())}
    judgements = {}
    for path in (BROKENMATH / "judgements").rglob("*.json"):
        row = json.loads(path.read_text())
        if "majority_vote" in row:
            judgements[path.stem] = row["majority_vote"]
    solver = yaml.safe_load((BROKENMATH / "configs/solvers/llada_30.yaml").read_text())
    ids = sorted(chosen, key=lambda name: hashlib.sha256(f"42:{name}".encode()).digest())
    heldout = set(ids[:max(4, round(len(ids) * .2))])
    pairs = []
    for name in ids:
        row = saved[name]
        verdict = judgements.get(name.replace("/", "_"))
        if verdict != "incorrect" or not row["is_adversarial"] or not row["solution"].strip():
            raise ValueError(f"Missing false-premise/incorrect response for {name}: {verdict}")
        pairs.append({
            "problem_id": name,
            "split": "heldout" if name in heldout else "train",
            "prompt": solver["prompt"].format(problem=row["problem"]),
            "chosen": chosen[name],
            "rejected": row["solution"],
            "rejected_source": "saved LLaDA generation",
            "chosen_source": "written refutation",
            "original_judge": verdict,
        })
    return pairs, sorted(saved.keys() - chosen.keys())


def get_features(pairs, model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    # CPU FP32 uses the host's optimized FP32 matrix multiplication.
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True, torch_dtype=dtype,
        low_cpu_mem_usage=True, device_map={"": device},
    ).eval().requires_grad_(False)
    # Capture the same final states ARGRE scores. Only project one position to
    # vocabulary logits because the scorer does not consume logits.
    projection = model.get_output_embeddings()
    hook = projection.register_forward_pre_hook(lambda module, inputs: (inputs[0][:, -1:],))
    samples = []
    try:
        with torch.inference_mode():
            for index, row in enumerate(pairs, 1):
                start = time.monotonic()
                messages = [{"role": "user", "content": row["prompt"]}]
                prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                hidden_pair = []
                for role in ("chosen", "rejected"):
                    completion = tokenizer.encode(row[role], add_special_tokens=False)
                    tokens = prefix + completion
                    if not completion or len(tokens) > model.config.max_sequence_length:
                        raise ValueError(f"Invalid token length for {row['problem_id']} ({role})")
                    sequence = torch.tensor([tokens], dtype=torch.long, device=device)
                    output = model(sequence, output_hidden_states=True, use_cache=False)
                    hidden = output.hidden_states[-1][0, len(prefix):]
                    if hidden.shape[0] != len(completion) or not torch.isfinite(hidden).all():
                        raise ValueError(f"Unexpected hidden states for {row['problem_id']}")
                    hidden_pair.append(hidden.to(device="cpu", dtype=torch.float32, copy=True))
                    del output, hidden
                samples.append(tuple(hidden_pair))
                print(f"Encoded {index}/{len(pairs)} {row['problem_id']} in {time.monotonic()-start:.1f}s", flush=True)
    finally:
        hook.remove()
    return samples


def padded_pairs(samples):
    length = max(len(hidden) for pair in samples for hidden in pair)
    hidden = torch.zeros(len(samples), 2, length, samples[0][0].shape[-1])
    mask = torch.zeros(len(samples), 2, length, dtype=torch.bool)
    for row, pair in enumerate(samples):
        for column, answer in enumerate(pair):
            hidden[row, column, :len(answer)] = answer
            mask[row, column, :len(answer)] = True
    return [{"hidden_states": states, "attention_mask": flags} for states, flags in zip(hidden, mask)]


def train_and_score(samples, pairs, run, device):
    train = [pair for pair, row in zip(samples, pairs) if row["split"] == "train"]
    heldout = [(pair, row) for pair, row in zip(samples, pairs) if row["split"] == "heldout"]
    diffs = torch.stack([chosen[-1] - rejected[-1] for chosen, rejected in train])
    # Match extract_direction.py: the released code adds the mean to PCA rank 1.
    pca = PCA(n_components=1).fit(diffs)
    direction = (pca.components_ + pca.mean_).reshape(-1)
    torch.save(direction, run / "direction.pt")
    head = nn.Sequential(nn.Linear(diffs.shape[-1], 1024), nn.ReLU(), nn.Linear(1024, 1))
    args = SimpleNamespace(
        batch_size=4, train_epoch=3, center_rewards=True, center_rewards_coefficient=.01,
        beta=.05, learning_rate=5e-4, weight_decay=.05, num_interpolations=7,
    )
    with wandb.init(mode="disabled"):
        trainer = RMTrainer(padded_pairs(train), [], head, args, device, str(run), direction)
        # The release hardcodes 100 warmup steps. Scale warmup to this run's
        # update count so the scorer reaches the published peak learning rate.
        updates = len(trainer.train_dataloader) * args.train_epoch
        warmup = min(100, max(1, round(updates * .1)))
        trainer.scheduler = get_cosine_schedule_with_warmup(trainer.optimizer, warmup, updates)
        for epoch in range(args.train_epoch):
            start = time.monotonic()
            trainer.train_epcoh(epoch)
            print(f"ARGRE epoch {epoch+1}/{args.train_epoch} complete ({time.monotonic()-start:.1f}s)", flush=True)
        trainer.save_model(head, output_dir=trainer.output_dir, cur_epoch="final")

    head.eval()
    scores = []
    with torch.no_grad():
        for (chosen, rejected), row in heldout:
            positive = float(head(chosen.to(device)).sum() * args.beta)
            negative = float(head(rejected.to(device)).sum() * args.beta)
            scores.append({
                "problem_id": row["problem_id"],
                "refutation_reward": positive,
                "saved_answer_reward": negative,
                "margin": positive - negative,
                "ranked_refutation_higher": positive > negative,
                "token_counts": [len(chosen), len(rejected)],
            })
    return {
        "train_pairs": len(train),
        "heldout_pairs": len(heldout),
        "heldout_correct": sum(row["ranked_refutation_higher"] for row in scores),
        "heldout_accuracy": sum(row["ranked_refutation_higher"] for row in scores) / len(scores),
        "scores": scores,
        "settings": {**vars(args), "warmup_steps": warmup, "optimizer_updates": updates},
        "meaning": "Pair ranking on saved LLaDA answers and written refutations. Rewards are uncalibrated.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    pairs, excluded = load_pairs()
    config = yaml.safe_load((BROKENMATH / "configs/models/llada/llada_8b_instruct.yaml").read_text())
    run = HERE / "results" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True)
    write_json(run / "pairs.json", pairs)
    print(f"{run} | {len(pairs)} pairs | device={args.device}", flush=True)
    samples = get_features(pairs, config["model_path"], args.device)
    # Reload once these are persisted so the trainer receives normal tensors
    # rather than inference tensors that cannot be saved for backward.
    cache = run / "hidden_states.pt"
    torch.save(samples, cache)
    del samples
    samples = torch.load(cache, weights_only=True, mmap=True)
    result = train_and_score(samples, pairs, run, args.device)
    result.update({"excluded_problem_ids": excluded, "model_path": config["model_path"],
                   "device": args.device, "backbone_trainable": False})
    write_json(run / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
