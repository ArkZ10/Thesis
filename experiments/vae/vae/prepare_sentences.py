"""Prepare auditable, question-disjoint math sentence reconstruction datasets.

Run from the workspace: python -m vae.prepare_sentences
No tokenizer, model, network access, training, or paid judging is used.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
VERSION = "math-sentences-v1"
SPLITS = ("train", "dev", "test")
CORPORA = ("gsm8k", "dart-math-hard")
ANSWER_MARKER = re.compile(r"\bthe\s+(?:final\s+)?answer\s+is\s*:?\s*", re.I)
BOX = re.compile(r"\\+(?:boxed|fbox)\s*\{")
EXPLICIT_MATH = re.compile(
    r"\\+\[[\s\S]*?\\+\]"
    r"|\\+\([\s\S]*?\\+\)"
    r"|\\+begin\{(?P<env>[^{}]+)\}[\s\S]*?\\+end\{(?P=env)\}"
    r"|<<[^<>]*>>"
)
ABBREVIATIONS = {"e.g.", "i.e.", "etc.", "mr.", "mrs.", "ms.", "dr.", "vs.", "fig.", "eq.", "no.", "w.r.t.", "u.s.", "a.m.", "p.m."}


def canonical(text):
    """Conservative exact-duplicate normalization; variable case is preserved."""
    # NFC preserves meaningful superscripts/compatibility math characters;
    # NFKC would incorrectly identify "x²" and "x2" as the same target.
    return " ".join(unicodedata.normalize("NFC", text).split())


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def problem_id(question):
    return digest(canonical(question))


def assign_split(pid, official_test_ids, seed=42):
    if pid in official_test_ids:
        return "test"
    value = int(digest(f"{seed}:{pid}")[:16], 16) / 2**64
    return "train" if value < 0.90 else "dev" if value < 0.95 else "test"


def balanced_boxes(text):
    out = []
    for match in BOX.finditer(text):
        start = match.end()
        depth, i = 1, start
        while i < len(text) and depth:
            ch = text[i]
            # Escaped literal braces do not close the box; doubled backslashes
            # followed by braces are commonly present in this raw corpus.
            escaped = i > 0 and text[i - 1] == "\\" and (i < 2 or text[i - 2] != "\\")
            if ch == "{" and not escaped:
                depth += 1
            elif ch == "}" and not escaped:
                depth -= 1
            i += 1
        if depth:
            return [], "unbalanced_answer_box"
        out.append((match.start(), i, text[start:i - 1]))
    return out, None


def strip_answer_wrapper(text):
    value = text.strip().rstrip(".。 ").strip()
    for left, right in [("$$", "$$"), ("$", "$"), (r"\\[", r"\\]"), (r"\[", r"\]"), (r"\\(", r"\\)"), (r"\(", r"\)")]:
        if value.startswith(left) and value.endswith(right) and len(value) > len(left) + len(right):
            value = value[len(left):-len(right)].strip()
            break
    boxes, error = balanced_boxes(value)
    if error:
        return None, error
    if boxes:
        if len(boxes) != 1 or value[:boxes[0][0]].strip() or value[boxes[0][1]:].strip():
            return None, "multiple_or_embedded_final_answers"
        value = boxes[0][2].strip()
    if "$" in value:
        return None, "multiple_or_unbalanced_answer_math"
    if not value or len(value) > 256 or len(value.split()) > 24 or "\n" in value:
        return None, "ambiguous_or_long_answer"
    return value, None


def extract_answer(response, corpus):
    """Return clean reasoning prefix, source answer, canonical answer, method."""
    if corpus == "gsm8k":
        if response.count("####") != 1:
            return None, "missing_or_multiple_gsm_answer_delimiter"
        start = response.index("####")
        reasoning, raw_answer, method = response[:start].strip(), response[start + 4:].strip(), "gsm_hash_delimiter"
    else:
        markers = list(ANSWER_MARKER.finditer(response))
        if markers:
            last = markers[-1]
            reasoning, raw_answer, method = response[:last.start()].strip(), response[last.end():].strip(), "explicit_answer_marker"
        else:
            boxes, error = balanced_boxes(response)
            if error:
                return None, error
            if not boxes:
                return None, "no_explicit_final_answer"
            start, end, _ = boxes[-1]
            trailing = response[end:].strip()
            # A terminal box is safe to extract; a box embedded in an unfinished
            # derivation is deliberately left for future manual curation.
            if not re.fullmatch(r"[\s.$\\\]\)\u3002]*", trailing):
                return None, "nonterminal_answer_box"
            reasoning = response[:start].rstrip()
            # Drop the opening math delimiter immediately preceding the box.
            reasoning = re.sub(r"(?:\${1,2}|\\+[\[(])\s*$", "", reasoning).rstrip()
            spans = sentence_spans(reasoning)
            if spans:
                a, b = spans[-1]
                fragment = reasoning[a:b]
                if len(fragment) < 160 and re.search(r"\b(?:answer|solution|result)\b", fragment, re.I) and not re.search(r"[.!?]$", fragment):
                    reasoning = reasoning[:a].rstrip()
            raw_answer, method = response[start:end], "terminal_box"
    answer, error = strip_answer_wrapper(raw_answer)
    if error:
        return None, error
    if not reasoning:
        return None, "empty_reasoning"
    return {"reasoning": reasoning, "final_answer_text": raw_answer, "final_answer": answer, "answer_method": method}, None


def quality_issue(text):
    """Mechanical damage filters, not a proof-correctness assessment."""
    if "\ufffd" in text or any(ord(c) < 32 and c not in "\n\r\t" for c in text):
        return "replacement_or_control_characters"
    if "[asy]" in text.lower() or "```" in text:
        return "embedded_code_outside_sentence_scope"
    foreign = sum("\u4e00" <= c <= "\u9fff" or "\uac00" <= c <= "\ud7af" or "\u0400" <= c <= "\u04ff" for c in text)
    if foreign >= 10 and foreign / max(len(text), 1) > 0.01:
        return "mixed_script_generation_damage"
    words = text.split()
    if len(words) >= 100:
        grams = [tuple(words[i:i + 8]) for i in range(len(words) - 7)]
        if 1 - len(set(grams)) / len(grams) > 0.30:
            return "excessive_repeated_8grams"
    return None


def protected_ranges(text):
    """Protect complete TeX math, GSM calculator annotations, and dollar math."""
    intervals = [(m.start(), m.end()) for m in EXPLICIT_MATH.finditer(text)]
    covered = bytearray(len(text))
    for a, b in intervals:
        covered[a:b] = b"\1" * (b - a)
    i = 0
    while i < len(text):
        if text[i] != "$" or covered[i] or (i and text[i - 1] == "\\"):
            i += 1
            continue
        width = 2 if text.startswith("$$", i) else 1
        # GSM uses a literal dollar sign before its calculator annotations.
        # Currency such as "$12 an hour" and "$18/hour" is not inline math.
        currency = width == 1 and (
            text.startswith("$<<", i)
            or re.match(r"\$\d[\d,.]*(?:\s+[A-Za-z]|/[A-Za-z])", text[i:])
        )
        if currency:
            i += 1
            continue
        delimiter = "$" * width
        end = text.find(delimiter, i + width)
        while end >= 0 and (
            (end > 0 and text[end - 1] == "\\")
            # A new amount "$270" is not the closing delimiter of "$720".
            or (width == 1 and end + 1 < len(text) and (text[end + 1].isalnum() or text.startswith("<<", end + 1)))
        ):
            end = text.find(delimiter, end + width)
        if end < 0:
            i += width
            continue
        # Inline math extending across blank paragraphs is likely malformed.
        if width == 1 and "\n\n" in text[i + 1:end]:
            i += 1
            continue
        if (width == 1 and text[i + 1:i + 2].isdigit() and "\n" in text[i + 1:end]
                and re.search(r"\b(?:per|dollars?|hours?|minutes?|earns?|paid|costs?|makes?|works?)\b", text[i + 1:end], re.I)):
            i += 1
            continue
        end += width
        intervals.append((i, end))
        covered[i:end] = b"\1" * (end - i)
        i = end
    return sorted(intervals), covered


def sentence_spans(text):
    """Lossless non-whitespace coverage by sentences or intact line-level steps."""
    intervals, protected = protected_ranges(text)
    math_sentence_ends = set()
    closing = re.compile(r"(?:\\+end\{[^}]+\}|\\+[\]\)]|\${1,2})$")
    for a, b in intervals:
        if b < len(text) and protected[b]:
            continue  # a nested math region, not the end of the whole formula
        inner = text[a:b].rstrip()
        for _ in range(3):
            cleaned = closing.sub("", inner).rstrip()
            if cleaned == inner:
                break
            inner = cleaned
        following = text[b:].lstrip()
        if inner.endswith((".", "!", "?")) and following and following[0].isupper():
            math_sentence_ends.add(b)
    result, start = [], 0

    def emit(end):
        nonlocal start
        a, b = start, end
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            result.append((a, b))
        start = end

    for i, char in enumerate(text):
        if protected[i]:
            if i + 1 in math_sentence_ends:
                emit(i + 1)
            continue
        if char == "\n":
            emit(i + 1)
        elif char in ".!?" and (i + 1 == len(text) or text[i + 1].isspace()):
            if char == ".":
                # Also protect abbreviations and numbered-list prefixes.
                token = text[:i + 1].rsplit(None, 1)[-1].lower()
                if token in ABBREVIATIONS or token in {"e.", "i.", "w.", "r."}:
                    continue
                if re.fullmatch(r"\d+\.", text[start:i + 1].strip()):
                    continue
            emit(i + 1)
    emit(len(text))
    return result


def eligible_sentence(text):
    return any(c.isalnum() for c in text)


def json_line(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


def source_files(root):
    # Human-authored GSM examples have priority for exact duplicate responses.
    for split in ("train", "test"):
        yield "gsm8k", split, root / "data/raw/gsm8k/main" / f"{split}-00000-of-00001.parquet"
    for path in sorted((root / "data/raw/dart-math-hard/data").glob("*.parquet")):
        yield "dart-math-hard", "train", path


def prepare(root, destination, seed=42):
    if destination.exists():
        raise FileExistsError(f"Output exists; use a new version directory: {destination}")
    staging = destination.with_name(destination.name + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    for subdir in ("parents", "sentences", "audit"):
        (staging / subdir).mkdir()
    sources = list(source_files(root))
    for _, _, path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    official_test_ids = {
        problem_id(row["question"])
        for row in pq.read_table(root / "data/raw/gsm8k/main/test-00000-of-00001.parquet", columns=["question"]).to_pylist()
    }
    revisions = {name: json.loads((root / "manifests" / f"{name}.json").read_text())["revision"] for name in CORPORA}
    stats = {name: {"raw_rows": 0, "excluded": Counter(), "parents": Counter(), "questions": {}, "candidate_sentences": Counter(), "answer_methods": Counter(), "retained_sentences": Counter(), "sentence_exclusions": Counter()} for name in CORPORA}
    question_sources = defaultdict(set)
    all_question_splits = {}
    accepted_question_splits = {}
    response_seen = set()
    processed = 0
    with ExitStack() as stack:
        parents = {(corpus, split): stack.enter_context((staging / "parents" / f"{corpus}.{split}.jsonl").open("w")) for corpus in CORPORA for split in SPLITS}
        excluded = stack.enter_context((staging / "audit/excluded_responses.jsonl").open("w"))
        for corpus, source_split, path in sources:
            row_index = 0
            relative = str(path.relative_to(root))
            for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
                for row in batch.to_pylist():
                    question = row["question" if corpus == "gsm8k" else "query"]
                    response = row["answer" if corpus == "gsm8k" else "response"]
                    pid = problem_id(question)
                    split = assign_split(pid, official_test_ids, seed)
                    all_question_splits[pid] = split
                    question_sources[pid].add(f"{corpus}:{source_split}")
                    parent_id = f"{corpus}:{path.stem}:{row_index}"
                    source = {"dataset": corpus, "revision": revisions[corpus], "source_split": source_split, "file": relative, "row_index": row_index}
                    row_index += 1
                    stats[corpus]["raw_rows"] += 1
                    extracted, issue = extract_answer(response, corpus)
                    issue = issue or quality_issue(response)
                    response_key = digest(pid + "\0" + canonical(response))
                    if not issue and response_key in response_seen:
                        issue = "duplicate_question_response"
                    if not issue:
                        response_seen.add(response_key)
                        spans = sentence_spans(extracted["reasoning"])
                        if not spans:
                            issue = "no_reasoning_sentences"
                    if issue:
                        stats[corpus]["excluded"][issue] += 1
                        json_line(excluded, {"parent_id": parent_id, "problem_id": pid, "split": split, "reason": issue, "source": source})
                    else:
                        reasoning = extracted["reasoning"]
                        # This asserts no non-whitespace character was lost.
                        compact = lambda s: "".join(s.split())
                        assert "".join(compact(reasoning[a:b]) for a, b in spans) == compact(reasoning)
                        record = {"parent_id": parent_id, "problem_id": pid, "split": split, "question": question, **extracted, "sentences": [{"index": i, "start": a, "end": b} for i, (a, b) in enumerate(spans)], "source": source, "mathematical_correctness_verified": False}
                        json_line(parents[corpus, split], record)
                        accepted_question_splits[pid] = split
                        stats[corpus]["questions"][pid] = split
                        stats[corpus]["parents"][split] += 1
                        stats[corpus]["candidate_sentences"][split] += len(spans)
                        stats[corpus]["answer_methods"][extracted["answer_method"]] += 1
                    processed += 1
                    if processed % 50000 == 0:
                        print(f"Parsed {processed:,} raw responses", flush=True)

    # Reserve exact normalized sentence duplicates to test, then dev, then train.
    # Parent records retain the full original ordered reasoning, even when a VAE
    # row is excluded for duplication. GSM rows win ties within the same split.
    sentence_seen = {}
    length_histogram = Counter()
    with ExitStack() as stack:
        exclusions = stack.enter_context((staging / "audit/excluded_sentences.jsonl").open("w"))
        for split in ("test", "dev", "train"):
            for corpus in CORPORA:
                with (staging / "parents" / f"{corpus}.{split}.jsonl").open() as source, (staging / "sentences" / f"{corpus}.{split}.jsonl").open("w") as target:
                    for line in source:
                        parent = json.loads(line)
                        for span in parent["sentences"]:
                            text = parent["reasoning"][span["start"]:span["end"]]
                            key = hashlib.sha256(canonical(text).encode()).digest()
                            issue = None
                            if not eligible_sentence(text):
                                issue = "no_alphanumeric_content"
                            elif key in sentence_seen:
                                issue = "duplicate_sentence_same_split" if sentence_seen[key] == split else "duplicate_sentence_higher_priority_split"
                            if issue:
                                stats[corpus]["sentence_exclusions"][issue] += 1
                                json_line(exclusions, {"parent_id": parent["parent_id"], "sentence_index": span["index"], "split": split, "reason": issue, "sentence_hash": key.hex()})
                                continue
                            sentence_seen[key] = split
                            length_histogram[len(text)] += 1
                            stats[corpus]["retained_sentences"][split] += 1
                            json_line(target, {"parent_id": parent["parent_id"], "problem_id": parent["problem_id"], "split": split, "sentence_index": span["index"], "start": span["start"], "end": span["end"], "language": "Math", "reasoning_block": text})
                print(f"Wrote {corpus} {split}: {stats[corpus]['retained_sentences'][split]:,} unique sentences", flush=True)

    total_lengths = sum(length_histogram.values())
    quantiles = {}
    cumulative = 0
    pending = [0.50, 0.90, 0.95, 0.99]
    for length, count in sorted(length_histogram.items()):
        cumulative += count
        while pending and cumulative >= pending[0] * total_lengths:
            quantiles[str(pending.pop(0))] = length
    for corpus in CORPORA:
        questions = stats[corpus].pop("questions")
        stats[corpus]["unique_questions_by_split"] = Counter(questions.values())
    source_file_records = []
    for corpus in CORPORA:
        manifest = json.loads((root / "manifests" / f"{corpus}.json").read_text())
        for item in manifest["files"]:
            if item["path"].endswith(".parquet"):
                source_file_records.append({"path": item["path"], "sha256": item["sha256"], "dataset": corpus, "revision": revisions[corpus]})
    question_manifest = [{"problem_id": pid, "split": split, "sources": sorted(question_sources[pid]), "accepted": pid in accepted_question_splits} for pid, split in sorted(all_question_splits.items())]
    with (staging / "audit/question_splits.jsonl").open("w") as stream:
        for item in question_manifest:
            json_line(stream, item)
    shared_train = sum("gsm8k:train" in x and "dart-math-hard:train" in x for x in question_sources.values())
    shared_test = sum("gsm8k:test" in x and "dart-math-hard:train" in x for x in question_sources.values())
    summary = {
        "format": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "split_policy": "SHA256(seed:normalized-question-hash), 90/5/5; official GSM8K test always test; coordinated across corpora",
        "normalization": "Unicode NFC and whitespace collapse; mathematical variable case and compatibility characters preserved",
        "sentence_deduplication": "global normalized exact match; priority test > dev > train, GSM8K > DART-Math within split",
        "statistics": stats, "shared_dart_gsm_train_questions": shared_train, "shared_dart_gsm_test_questions": shared_test,
        "raw_unique_questions": len(all_question_splits), "accepted_unique_questions": len(accepted_question_splits),
        "official_gsm_test_questions": len(official_test_ids),
        "retained_sentence_char_length_quantiles": quantiles, "max_sentence_characters": max(length_histogram, default=0),
        "source_files": source_file_records,
        "preprocessing_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": ["Answer extraction and damage filtering do not certify correct mathematics.", "Exact/normalized overlaps are checked; semantic paraphrases may remain.", "No tokenization or model-specific length filtering has been performed.", "Parent reasoning retains earlier answer mentions; the terminal final answer is separated.", "Parent records preserve all steps; deduplicated VAE rows are a reconstruction subset."]
    }
    write_json(staging / "summary.json", summary)
    # Explicit manifests let the VAE loader preserve these splits instead of
    # applying the upstream sentence-level last-100 validation split.
    for corpus in CORPORA:
        write_json(staging / f"{corpus}.vae.json", {
            "format": "vae_sentence_splits_v1", "dataset": corpus, "language": "Math",
            "train": [f"sentences/{corpus}.train.jsonl"],
            "validation": [f"sentences/{corpus}.dev.jsonl"],
            "test": [f"sentences/{corpus}.test.jsonl"],
            "counts": dict(stats[corpus]["retained_sentences"]),
        })
    staging.rename(destination)
    print(json.dumps({"output": str(destination), "statistics": stats}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed" / VERSION)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(ROOT, args.output.resolve(), args.seed)


if __name__ == "__main__":
    main()
