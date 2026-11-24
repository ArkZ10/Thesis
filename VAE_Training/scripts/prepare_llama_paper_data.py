"""Tokenize all prepared DART-MATH training targets; no GSM8K mix or pilot cap.

Sentence parsing, filtering and question splits are explicitly local choices;
the paper does not release an exact mathematics data-preparation pipeline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from transformers import AutoTokenizer
from vae.prepare_sentences import problem_id

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'data/processed/llama-paper-dart-full-v1'


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    OUTPUT = parser.parse_args().output
    OUTPUT.mkdir(exist_ok=False)
    source = ROOT / 'data/processed/math-sentences-v1/sentences/dart-math-hard.train.jsonl'
    expected = 4918593
    progress = {'stage': 'tokenizing', 'source': str(source), 'expected_training_sentences': expected,
                'training_sentences': 0, 'training_started': False, 'test_used': False}
    write(OUTPUT / 'status.json', progress)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/Llama-3.1-8B', local_files_only=True)
    eos = tokenizer.eos_token_id
    count, total_tokens, max_tokens = 0, 0, 0
    bins = {str(limit): 0 for limit in [128, 256, 512, 1024, 2048, 4096]}
    try:
        with source.open('rb') as stream, (OUTPUT / 'train.tokens.u32').open('wb') as tokens_out, \
             (OUTPUT / 'train.offsets.u64').open('wb') as offsets_out, \
             (OUTPUT / 'train.source-offsets.u64').open('wb') as source_out:
            np.array([0], dtype='<u8').tofile(offsets_out)
            while True:
                lines, positions = [], []
                for _ in range(1024):
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    lines.append(json.loads(line)); positions.append(offset)
                if not lines:
                    break
                if shutil.disk_usage(ROOT).free < 32 * 1024**3:
                    raise RuntimeError('Data preparation stopped below 32 GiB free disk')
                assert all(x['split'] == 'train' and x['parent_id'].startswith('dart-math-hard:') for x in lines)
                encoded = tokenizer([x['reasoning_block'] for x in lines], add_special_tokens=False)['input_ids']
                lengths = []
                for ids in encoded:
                    ids.append(eos)
                    size = len(ids)
                    if size > tokenizer.model_max_length:
                        raise RuntimeError('Sentence exceeds pretrained model context; author length policy is unreleased')
                    lengths.append(size)
                    max_tokens = max(max_tokens, size)
                    for limit in bins:
                        bins[limit] += size > int(limit)
                flattened = np.fromiter((v for ids in encoded for v in ids), dtype='<u4')
                flattened.tofile(tokens_out)
                offsets = np.cumsum(np.array(lengths, dtype='<u8')) + total_tokens
                offsets.astype('<u8').tofile(offsets_out)
                np.array(positions, dtype='<u8').tofile(source_out)
                total_tokens += len(flattened); count += len(lines)
                if count % 16384 == 0:
                    progress.update(training_sentences=count, progress_percent=100 * count / expected,
                                    tokens=total_tokens, max_tokens=max_tokens,
                                    updated_at_utc=datetime.now(timezone.utc).isoformat())
                    write(OUTPUT / 'status.json', progress)
                    print(json.dumps(progress), flush=True)
        assert count == expected
        # Preserve a small existing DART-only dev subset for local monitoring.
        # All available training sentences remain eligible, rather than selecting
        # one response or a limited number of training questions.
        dev_source = ROOT / 'data/processed/math-sentences-v1/sentences/dart-math-hard.dev.jsonl'
        dev, per_question = [], {}
        with dev_source.open() as stream:
            for line in stream:
                row = json.loads(line)
                pid = row['problem_id']
                if pid not in per_question and len(per_question) >= 32:
                    continue
                per_question.setdefault(pid, 0)
                if per_question[pid] >= 4:
                    continue
                ids = tokenizer.encode(row['reasoning_block'], add_special_tokens=False) + [eos]
                dev.append({**row, 'dataset': 'dart-math-hard', 'input_ids': ids})
                per_question[pid] += 1
                if len(dev) >= 128:
                    break
        assert len(dev) >= 64
        with (OUTPUT / 'dev.jsonl').open('w') as stream:
            for row in dev:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        validation = json.loads((ROOT / 'data/processed/math-sentences-v1/validation.json').read_text())
        files = {f.name: sha(f) for f in OUTPUT.iterdir() if f.is_file() and f.name != 'status.json'}
        summary = {'format': 'llama_token_memmap_v1', 'source_directory': str(source.parent.parent),
            'training_source_jsonl': str(source), 'training_source_sha256': sha(source),
            'tokenizer_revision': 'd04e592bb4f6aa9cfee91e2e20afa771667e1d4b',
            'train': {'sentences': count, 'tokens_including_eos': total_tokens, 'max_tokens': max_tokens,
                      'lengths_above': bins, 'sampling': 'none; every prepared DART training target'},
            'dev': {'sentences': len(dev), 'questions': len(per_question),
                    'selection': 'first up to 32 existing dev questions, up to four sentences each'},
            'files': files, 'test_used': False, 'official_gsm8k_test_used': False, 'gsm8k_training_mixed': False,
            'token_truncation': False, 'max_tokens_including_eos': None, 'question_overlap': 0,
            'exact_authors_dataset_revision_verified': False,
            'raw_source_rows': 585392, 'raw_source_manifest': 'manifests/dart-math-hard.json',
            'original_preparation_manifest': 'manifests/preprocessing.json',
            'original_validation_available': bool(validation),
            'local_choices_not_author_confirmed': ['DART-Math-Hard subset/revision',
                'Existing reasoning extraction, protected sentence splitting, filtering and deduplication',
                'Existing question-disjoint train/dev/test holdout; official GSM8K test reserved',
                'Small fixed DART-only dev monitoring subset', 'EOS appended without BOS'],
            'completed_at_utc': datetime.now(timezone.utc).isoformat()}
        write(OUTPUT / 'summary.json', summary)
        progress.update(stage='completed', training_sentences=count, tokens=total_tokens,
                        progress_percent=100., max_tokens=max_tokens)
        write(OUTPUT / 'status.json', progress)
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as exc:
        progress.update(stage='failed', training_sentences=count, error=str(exc))
        write(OUTPUT / 'status.json', progress)
        raise


if __name__ == '__main__':
    main()
