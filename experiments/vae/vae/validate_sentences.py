"""Independently check prepared files, source provenance and split separation."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
import re

import pyarrow.parquet as pq

from .prepare_sentences import CORPORA, SPLITS, ROOT, assign_split, canonical, problem_id, protected_ranges


def file_hash(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def validate(directory):
    summary = json.loads((directory / "summary.json").read_text())
    seed = summary["seed"]
    official_test_ids = {problem_id(row["question"]) for row in pq.read_table(ROOT / "data/raw/gsm8k/main/test-00000-of-00001.parquet", columns=["question"]).to_pylist()}
    for source in summary["source_files"]:
        assert file_hash(ROOT / source["path"]) == source["sha256"], source
    question_sets = {split: set() for split in SPLITS}
    parent_counts, sentence_counts = Counter(), Counter()
    source_parent_counts = Counter()
    for corpus in CORPORA:
        for split in SPLITS:
            for parent in rows(directory / "parents" / f"{corpus}.{split}.jsonl"):
                pid = parent["problem_id"]
                assert problem_id(parent["question"]) == pid
                assert parent["split"] == split == assign_split(pid, official_test_ids, seed)
                if parent["source"]["dataset"] == "gsm8k" and parent["source"]["source_split"] == "test":
                    assert split == "test"
                    source_parent_counts["official_gsm_test"] += 1
                question_sets[split].add(pid)
                parent_counts[corpus, split] += 1
                text = parent["reasoning"]
                _, protected = protected_ranges(text)
                compact = lambda s: "".join(s.split())
                assert "".join(compact(text[s["start"]:s["end"]]) for s in parent["sentences"]) == compact(text)
                previous_end = 0
                for index, span in enumerate(parent["sentences"]):
                    start, end = span["start"], span["end"]
                    assert previous_end <= start < end <= len(text)
                    assert index == span["index"]
                    assert not text[previous_end:start].strip()
                    # Boundaries cannot cut inside a protected equation.
                    assert start == 0 or not (protected[start] and protected[start - 1])
                    assert end == len(text) or not (protected[end - 1] and protected[end])
                    previous_end = end
                assert not text[previous_end:].strip()
            print(f"Validated parent coverage: {corpus} {split}", flush=True)
    overlaps = {f"{a}/{b}": len(question_sets[a] & question_sets[b]) for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]}
    assert not any(overlaps.values()), overlaps
    all_sentences = set()
    samples = defaultdict(list)
    sample_serial = 0
    max_chars = 0
    for corpus in CORPORA:
        for split in SPLITS:
            parent_iter = iter(rows(directory / "parents" / f"{corpus}.{split}.jsonl"))
            parent = next(parent_iter, None)
            for sentence in rows(directory / "sentences" / f"{corpus}.{split}.jsonl"):
                while parent is not None and parent["parent_id"] != sentence["parent_id"]:
                    parent = next(parent_iter, None)
                assert parent is not None, sentence
                start, end = sentence["start"], sentence["end"]
                text = sentence["reasoning_block"]
                assert parent["reasoning"][start:end] == text
                assert parent["sentences"][sentence["sentence_index"]] == {"index": sentence["sentence_index"], "start": start, "end": end}
                assert sentence["problem_id"] == parent["problem_id"]
                assert sentence["split"] == split
                assert sentence["language"] == "Math" and text.strip()
                key = hashlib.sha256(canonical(text).encode()).digest()
                assert key not in all_sentences, sentence
                all_sentences.add(key)
                sentence_counts[corpus, split] += 1
                max_chars = max(max_chars, len(text))
                if len(text) <= 800:
                    categories = [corpus]
                    if re.search(r"\d\.\d", text): categories.append("decimals")
                    if "\n" in text: categories.append("multiline")
                    if re.search(r"\\+[\[(]|\$\$", text): categories.append("display_math")
                    if "$" in text: categories.append("dollar_notation")
                    if len(text) <= 15: categories.append("short_steps")
                    if "<<" in text: categories.append("gsm_calculator")
                    if len(text) > 400: categories.append("long_steps")
                    if len(text) < 100 and "=" in text: categories.append("equations")
                    if not any(c in text for c in "$=\\<"): categories.append("plain_text")
                    rank = int(hashlib.sha256((sentence["parent_id"] + str(sentence["sentence_index"])).encode()).hexdigest(), 16)
                    candidate = {**sentence, "dataset": corpus, "final_answer": parent["final_answer"]}
                    sample_serial += 1
                    for category in categories:
                        heapq.heappush(samples[category], (-rank, sample_serial, candidate))
                        if len(samples[category]) > 20:
                            heapq.heappop(samples[category])
            assert sentence_counts[corpus, split] == summary["statistics"][corpus]["retained_sentences"].get(split, 0)
            assert parent_counts[corpus, split] == summary["statistics"][corpus]["parents"].get(split, 0)
            print(f"Validated unique VAE rows: {corpus} {split}", flush=True)
    excluded_responses = Counter(item["source"]["dataset"] for item in rows(directory / "audit/excluded_responses.jsonl"))
    excluded_sentences = Counter(item["parent_id"].split(":")[0] for item in rows(directory / "audit/excluded_sentences.jsonl"))
    for corpus in CORPORA:
        stats = summary["statistics"][corpus]
        assert stats["raw_rows"] == excluded_responses[corpus] + sum(parent_counts[corpus, s] for s in SPLITS)
        assert sum(stats["excluded"].values()) == excluded_responses[corpus]
        assert sum(stats["candidate_sentences"].values()) == excluded_sentences[corpus] + sum(sentence_counts[corpus, s] for s in SPLITS)
    selected, selected_ids = [], set()
    ordered = ["short_steps", "multiline", "display_math", "decimals", "gsm_calculator", "equations", "long_steps", "dollar_notation", "plain_text", "dart-math-hard", "gsm8k"]
    for category in ordered:
        added = 0
        for _, _, item in sorted(samples[category], reverse=True):
            key = item["parent_id"], item["sentence_index"]
            if key in selected_ids:
                continue
            selected_ids.add(key)
            selected.append({"review_index": len(selected) + 1, "category": category, **item})
            added += 1
            if added >= 10 or len(selected) >= 100:
                break
        if len(selected) >= 100:
            break
    # Fill gaps when the same example belongs to several strata.
    if len(selected) < 100:
        for category in ordered:
            for _, _, item in sorted(samples[category], reverse=True):
                key = item["parent_id"], item["sentence_index"]
                if key in selected_ids:
                    continue
                selected_ids.add(key)
                selected.append({"review_index": len(selected) + 1, "category": category, **item})
                if len(selected) >= 100:
                    break
            if len(selected) >= 100:
                break
    with (directory / "audit/inspection_sample.jsonl").open("w") as stream:
        for item in selected:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    artifacts = [{"path": str(path.relative_to(directory)), "bytes": path.stat().st_size, "sha256": file_hash(path)} for path in sorted(directory.rglob("*")) if path.is_file() and path.name not in {"validation.json", "manual_review.json"}]
    report = {
        "passed": True, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_overlap_counts": overlaps, "global_unique_sentence_targets": len(all_sentences),
        "unique_questions_by_split": {s: len(question_sets[s]) for s in SPLITS},
        "official_gsm_test_parents_reserved": source_parent_counts["official_gsm_test"],
        "source_checksums_verified": len(summary["source_files"]),
        "all_parent_text_coverage_checked": True, "all_formula_boundaries_checked": True,
        "all_sentence_targets_match_parent_spans": True, "exact_sentence_duplicates": 0,
        "all_row_accounting_checked": True, "max_sentence_characters": max_chars,
        "inspection_sample_count": len(selected), "artifacts": artifacts,
        "limitations": summary["limitations"],
    }
    (directory / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"artifacts", "limitations"}}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    validate(args.directory.resolve())


if __name__ == "__main__":
    main()
