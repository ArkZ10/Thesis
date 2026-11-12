# Prepared sentence-level math data

Output directory: `data/processed/math-sentences-v1/`.

This stage prepares reconstruction text and provenance. It does not tokenize data, train a model, verify all mathematical reasoning, or evaluate safety/sycophancy.

## Files

- `parents/<dataset>.<split>.jsonl`: accepted question/response records, separated final answer, clean reasoning prefix, ordered sentence offsets, and exact source dataset/revision/file/row.
- `sentences/<dataset>.<split>.jsonl`: deduplicated VAE reconstruction targets with parent and problem IDs, sentence index, character offsets, and `language: Math`.
- `<dataset>.vae.json`: explicit train/validation/test file manifest, accepted by `vae.data.build_datasets`. Only train and validation are loaded for training; reserved test files are not read by this loader.
- `summary.json`: preparation settings, source fingerprints, detailed counts and exclusion reasons.
- `validation.json`: independent full-output validation and artifact hashes.
- `audit/question_splits.jsonl`: stable split assignment for every raw normalized question, including excluded responses.
- `audit/excluded_responses.jsonl`: response-level exclusions with source row references.
- `audit/excluded_sentences.jsonl`: exact duplicate or non-content sentence exclusions.
- `audit/inspection_sample.jsonl`: stratified sentence sample for inspection.
- `audit/manual_review.json`: review notes, created after inspecting the saved sample.

Final checks passed: 5,478,995 globally unique targets; zero normalized-question overlap across splits; all 1,319 official GSM8K test questions reserved; every saved target matches its parent span. All protected formula boundaries and parent non-whitespace text coverage were checked. There are 15 passing unit/integration tests and 100 inspected examples. Inspection is a formatting review, not mathematical certification; source reasoning errors and source fragments remain, and missing prose spacing can leave multiple sentences in an intact step.

## Counts

| Corpus | Train targets | Development targets | Test targets |
|---|---:|---:|---:|
| DART-Math-Hard | 4,918,593 | 283,246 | 245,303 |
| GSM8K main | 24,321 | 1,401 | 6,131 |

These are sentence/step counts, not question counts. Before sentence deduplication, accepted parent responses are:

| Corpus | Train responses | Development responses | Test responses |
|---|---:|---:|---:|
| DART-Math-Hard | 469,975 | 26,512 | 21,515 |
| GSM8K main | 6,742 | 380 | 1,670 |

67,390 DART-Math responses were excluded; all 8,792 GSM8K responses were accepted. Filtering is a mechanical parsing/damage policy, not a guarantee that accepted reasoning is correct.

## Split and segmentation policy

Question identity uses SHA256 over Unicode NFC plus whitespace normalization, preserving mathematical variable case and compatibility characters such as superscripts. A seeded hash assigns approximately 90/5/5 percent to train/dev/test, with the same assignment across both corpora. All original GSM8K test questions are forced into test. GSM8K's prepared test file includes both its official test set and a held-out portion of its training set; source provenance distinguishes them.

DART-Math and GSM8K training share 7,468 exact-normalized questions. Their shared questions remain together. The downloaded DART-Math has no exact-normalized question overlap with the official GSM8K test set. Semantic paraphrase overlap is not certified absent.

The splitter preserves complete TeX math regions, calculator annotations, decimals, and short equations. Sentence punctuation and unprotected newlines form boundaries, so standalone equations and line-level mathematical steps can be individual blocks. A sentence ending inside a closed formula can split after the delimiter when the next sentence starts with a capital letter, including missing-space formatting such as `$$x=2.$$Then...`. It never silently truncates a step. Parent offsets preserve the complete accepted reasoning text, including steps excluded from the deduplicated VAE targets. Line-level blocks can include source fragments or introductions to a following equation; this is a deterministic mathematical-step policy, not a grammatical sentence parser.

Exact normalized sentence duplicates are assigned to a single split, with priority test > development > training. Human-authored GSM8K wins ties within a split. Parent reasoning sequences remain intact for future blockwise training/export. Do not reconstruct an entire parent chain from the deduplicated VAE rows alone.

Terminal final answers are extracted from GSM8K's `####`, explicit DART-Math answer markers, or unambiguous terminal boxed answers. Ambiguous/multiple/unbalanced answers and nonterminal boxes are excluded. Earlier answer mentions may remain within the reasoning prefix. Parent records explicitly mark mathematical correctness as unverified.

Other exclusions include damaged mixed-script generations, replacement/control characters, excessive repeated 8-grams, embedded code, and duplicate question/response pairs. See the source and summary for exact thresholds.

## Loading and reproduction

For a future VAE config, use:

```yaml
data:
  path: /root/Thesis/VAE_Training/data/processed/math-sentences-v1/dart-math-hard.vae.json
  language: Math
  split_seed: null
  cache_path: null
```

Use `gsm8k.vae.json` for the diagnostic corpus. The manifest loader preserves the prepared splits, ignores the upstream tail-validation setting, rejects sentence reshuffling, and preserves mathematical targets shorter than ten characters.

Reproduce with the existing data tools, choosing a fresh output directory to avoid overwriting results:

```bash
cd /root/Thesis/VAE_Training
/opt/miniforge3/envs/diffuguard/bin/python -m vae.prepare_sentences --output data/processed/math-sentences-v2
/opt/miniforge3/envs/diffuguard/bin/python -m vae.validate_sentences data/processed/math-sentences-v2
```

Tests: `python -m unittest discover -s tests -v` (using an environment with PyTorch and PyArrow).

Token counts and 128/256-token length limits remain pending until model/tokenizer selection. The longest retained target is 8,240 characters, so a later explicit token-length audit is required before training.
