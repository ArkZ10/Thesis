"""Select a reproducible question-disjoint pilot from prepared DART-Math data."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from scripts.run_small_diagnostic import annotations_correct, write
from vae.prepare_sentences import canonical

ROOT = Path(__file__).resolve().parents[1]


def rank(value, seed=42):
    return hashlib.sha256(f"vae-pilot-v1:{seed}:{value}".encode()).hexdigest()


def fast_parent_id(line):
    prefix = '{"parent_id":"'
    if line.startswith(prefix):
        return line[len(prefix):line.index('"', len(prefix))]
    return json.loads(line)["parent_id"]


def select_split(prepared, split, questions, tokenizer, max_tokens, output, seed, excluded_parent_ids=()):
    chosen = {}
    parents_file = prepared / "parents" / f"dart-math-hard.{split}.jsonl"
    for line in parents_file.open():
        parent = json.loads(line)
        assert parent["split"] == split
        if parent["parent_id"] in excluded_parent_ids:
            continue
        pid = parent["problem_id"]
        if pid not in chosen or rank(parent["parent_id"], seed) < rank(chosen[pid]["parent_id"], seed):
            chosen[pid] = parent
    ordered = sorted(chosen, key=lambda pid: rank(pid, seed))
    pool = ordered[:min(len(ordered), questions * 3)]
    wanted = {chosen[pid]["parent_id"] for pid in pool}
    candidates = defaultdict(list)
    source = prepared / "sentences" / f"dart-math-hard.{split}.jsonl"
    scanned = 0
    for line in source.open():
        scanned += 1
        if fast_parent_id(line) in wanted:
            row = json.loads(line)
            candidates[row["parent_id"]].append(row)
        if scanned % 1000000 == 0:
            print(f"{split}: scanned {scanned:,} source sentence rows", flush=True)
    retained, selected_parents, exclusions = [], [], []
    counters, lengths = Counter(), []
    for pid in pool:
        parent = chosen[pid]
        accepted = []
        for row in candidates[parent["parent_id"]]:
            text = row["reasoning_block"]
            ids = tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            counters["candidate_sentences_examined"] += 1
            issue = None
            if len(ids) > max_tokens:
                issue = "over_token_limit_not_truncated"
            elif len(ids) < 4 and not any(c.isdigit() for c in text) and "\\" not in text:
                issue = "tiny_prose_fragment"
            elif "<<" in text and not annotations_correct(text):
                issue = "unverified_or_incorrect_calculator_annotation"
            if issue:
                counters[issue] += 1
                exclusions.append({"parent_id": row["parent_id"], "sentence_index": row["sentence_index"],
                                   "tokens_including_eos": len(ids), "reason": issue})
                continue
            assert row["problem_id"] == pid and row["split"] == split
            assert text == parent["reasoning"][row["start"]:row["end"]]
            accepted.append({**row, "input_ids": ids, "token_count_including_eos": len(ids)})
        if not accepted:
            counters["questions_with_no_eligible_targets"] += 1
            continue
        retained.extend(accepted)
        lengths.extend(r["token_count_including_eos"] for r in accepted)
        selected_parents.append(parent)
        if len(selected_parents) == questions:
            break
    if len(selected_parents) != questions:
        raise RuntimeError(f"Only {len(selected_parents)} eligible {split} questions; requested {questions}")
    for name, records in [(f"{split}.jsonl", retained), (f"{split}-parents.jsonl", selected_parents),
                          (f"{split}-exclusions.jsonl", exclusions)]:
        with (output / name).open("w") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    lengths.sort()
    stats = {"questions": len(selected_parents), "responses": len(selected_parents),
             "sentences": len(retained), "tokens_including_eos": sum(lengths),
             "min_tokens": min(lengths), "max_tokens": max(lengths),
             "median_tokens": lengths[len(lengths) // 2], "source_rows_scanned": scanned,
             "filter_counts": dict(counters), "question_ids": [p["problem_id"] for p in selected_parents],
             "parent_ids": [p["parent_id"] for p in selected_parents]}
    print(f"Selected {split}: {questions} questions, {len(retained):,} sentences", flush=True)
    return retained, selected_parents, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--train-questions", type=int, default=1000)
    parser.add_argument("--dev-questions", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manual-exclusions", type=Path)
    args = parser.parse_args()
    output = args.directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = json.loads((ROOT / "manifests/qwen3_0_6b_base.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["directory"], local_files_only=True)
    source = ROOT / "data/processed/math-sentences-v1"
    exclusions = json.loads(args.manual_exclusions.read_text()) if args.manual_exclusions else []
    excluded_parent_ids = {r["parent_id"] for r in exclusions}
    train_rows, train_parents, train_stats = select_split(source, "train", args.train_questions, tokenizer, args.max_tokens, output, args.seed, excluded_parent_ids)
    dev_rows, dev_parents, dev_stats = select_split(source, "dev", args.dev_questions, tokenizer, args.max_tokens, output, args.seed, excluded_parent_ids)
    assert not (set(train_stats["question_ids"]) & set(dev_stats["question_ids"]))
    train_hashes = {hashlib.sha256(canonical(r["reasoning_block"]).encode()).hexdigest() for r in train_rows}
    dev_hashes = {hashlib.sha256(canonical(r["reasoning_block"]).encode()).hexdigest() for r in dev_rows}
    assert not train_hashes & dev_hashes
    assert len(train_hashes) == len(train_rows) and len(dev_hashes) == len(dev_rows)
    sample = []
    for split, rows in [("train", train_rows), ("dev", dev_rows)]:
        by_question = defaultdict(list)
        for row in rows:
            by_question[row["problem_id"]].append(row)
        for pid in sorted(by_question, key=lambda x: rank("review:" + x, args.seed))[:50]:
            row = min(by_question[pid], key=lambda r: rank(r["parent_id"] + str(r["sentence_index"]), args.seed))
            sample.append({"review_index": len(sample) + 1, **row})
    with (output / "inspection_sample.jsonl").open("w") as stream:
        for row in sample:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    controls = sorted(dev_rows, key=lambda r: rank("controls:" + r["parent_id"] + str(r["sentence_index"]), args.seed))[:64]
    with (output / "controls.jsonl").open("w") as stream:
        for row in controls:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write(output / "pilot.vae.json", {"format": "vae_sentence_splits_v1", "language": "Math",
          "train": ["train.jsonl"], "validation": ["dev.jsonl"], "test": []})
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
          "dataset": "hkust-nlp/dart-math-hard", "source_directory": str(source),
          "model_tokenizer_revision": checkpoint["revision"], "max_tokens_including_eos": args.max_tokens,
          "one_response_per_question": True, "token_truncation": False, "test_used": False,
          "question_overlap": 0, "sentence_overlap": 0,
          "train": train_stats, "dev": dev_stats, "inspection_samples": len(sample),
          "control_sentences": len(controls), "mathematical_correctness_certified": False,
          "manual_response_exclusions": exclusions,
          "files": [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                    for p in sorted(output.iterdir()) if p.is_file()]}
    write(output / "summary.json", summary)
    print(json.dumps({k: {a: b for a, b in v.items() if not a.endswith("_ids")} for k, v in [("train", train_stats), ("dev", dev_stats)]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
