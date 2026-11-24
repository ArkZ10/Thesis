# Data

Raw downloads remain under `data/raw/dart-math-hard` and `data/raw/gsm8k`. Their pinned revisions and checksums are in `manifests/dart-math-hard.json` and `manifests/gsm8k.json`.

`data/processed/math-sentences-v1/` contains the source reasoning sentences, train/dev/test question splits, provenance, filtering statistics and independent validation. Final answers are extracted separately. Sentence splitting protects equations, decimals and calculator annotations. Repeated questions remain in the same split; official GSM8K test questions are reserved. Shared sentence duplicates are excluded from training where required by the preparation policy. Details are in its `summary.json` and `validation.json`.

`data/processed/llama-paper-dart-full-v1/` is the current training input:

| Split | Contents |
| --- | --- |
| Train | 4,918,593 DART sentences; 168,031,583 tokens including EOS |
| Dev | 128 DART sentences from 32 held-out questions |
| Test | Reserved in the source corpus; never loaded by the trainer |

Train tokens and source offsets use memory-mapped arrays. The original JSONL must remain available because it supplies sentence metadata. No training target is truncated; the longest is 1608 tokens. GSM8K is available but is not mixed into the current paper-defined training configuration.

`data/processed/llama-mixed-pilot-v1/` is retained because it produced the saved LLaMA pilot: 1024 train sentences and 64 dev sentences, with the earlier 128-token selection cap. Obsolete Qwen pilot subsets have been deleted.

Preparation entrypoints (existing output directories are protected against overwrite):

```bash
python scripts/download_datasets.py --help
python -m vae.prepare_sentences --help
python -m vae.validate_sentences --help
python scripts/prepare_llama_paper_data.py --help
```

The authors did not release the mathematics sentence parser or confirm the exact DART subset/revision. These prepared data are an explicit local implementation, not a verified identical reproduction of their data.
