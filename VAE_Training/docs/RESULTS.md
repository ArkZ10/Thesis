# Retained results

## Latest completed LLaMA training

Directory: `outputs/vae-llama-mixed-pilot-20261114-181908/`.

This earlier pilot trained the full base encoder on 1024 mixed DART+GSM8K sentences for two epochs (16 updates), using six latents of width 512 and a frozen decoder. It is different from the current four-latent, full-DART configuration.

| Epoch | Train token accuracy | Dev token accuracy | Exact reconstruction | Numeric preservation |
| --- | --- | --- | --- | --- |
| 1 | 33.78% | 42.81% | 0/64 | 1/52 |
| 2 | 47.00% | 48.11% | 0/64 | 0/52 |

The decoder remained unchanged. Token accuracy improved, but exact sentence reconstruction did not succeed on the 64 held-out dev examples. This is a diagnostic result, not mathematical-answer utility or a successful full-scale encoder.

Read `epoch-2-dev-reconstructions.jsonl` to compare each `target` with generated `text`. `epoch-2-metrics.json` holds the metrics and prefix controls. `summary.json` contains the run summary. The historical source snapshot is retained within the result directory; active code is under `vae/` and `scripts/`.

`weights-fp32/` preserves 298 exact FP32 master tensors (30,044,999,680 tensor bytes), with every exported tensor hash verified at export. Encoder parameters, queries, posterior heads, projection and AE vector are present; frozen decoder weights load separately from the cached base model. The old optimizer state was deleted by request. These weights support evaluation or a fresh optimizer, not exact continuation of the old training trajectory. The old `checkpoints/` directory contains metadata, not a usable resumable checkpoint.

## Latest valid execution test

Directory: `outputs/llama-execution-profile-20261116-220708/`.

Two consecutive effective-batch-128 updates took **144.76 seconds** and **130.26 seconds**, with finite gradient norms, no skipped updates and an unchanged frozen decoder. Raw measurements and results remain in `optimized-measurements.json` and `optimized-results.json`; consolidated interpretation is in `manifests/llama_execution_optimization.json`.

The profile exercised short DART targets. The full-run estimate is 116–129 days before evaluation/checkpoint overhead; longest-target GPU fit and optimized-backend checkpoint/resume remain unverified. Profiling updates were disposable; this directory contains execution evidence rather than trained encoder weights.

All superseded experiment directories and their old logs were deleted directly. No archive was created.

## Paper comparison

[LaDiR Figure 5](https://arxiv.org/html/2510.04573v5) reports reconstruction accuracies of 89.3%, 99.3%, and 100.0% with 1, 6, and 16 latent tokens respectively; downstream GSM8K accuracies are 12.3%, 84.4%, and 66.3%. The figure labels the metric as reconstruction accuracy without a precise token-versus-sentence or generation-protocol definition. These values should not be equated directly with our normalized whole-sentence exact-match metric.

The new small-encoder experiment is indexed by `manifests/small_training.json`; its results are not assumed successful in advance.

## Stopped small-encoder trial

`outputs/vae-small-encoder-20261118-061012/` passed the clean 16-sentence overfit gate at update 50: 16/16 exact reconstructions and 16/16 numeric preservation. Its held-out pilot was stopped during epoch 3 when the user selected the official pretrained encoder instead. The last completed epoch metrics and atomic checkpoints remain in the run. The latest available dev evaluation had 0/128 exact reconstructions; overfit success did not establish generalization.

## Released encoder candidate

The official [README](https://github.com/mk322/LaDiR) explicitly allows reusing the trained VAE encoder inside [LaDiR-Code-8B](https://huggingface.co/mk322/LaDiR-Code-8B). Its 64 × 2560 representation differs from our mathematics configuration. Download/load status and actual local validation are recorded in `manifests/pretrained_vae.json` and `outputs/pretrained-vae-check/summary.json` when available.

The released encoder is now downloaded and strictly loaded. On four held-out GSM8K reasoning sentences it produced finite, repeatable `[4, 64, 2560]` posterior-mean latents. The FP32 inference reference agreed between individual and padded-batch encoding (minimum cosine 1.0; maximum absolute difference 8.76e-5), using about 15.21 GiB peak GPU memory. BF16 inference showed measurable batch rounding differences (minimum cosine 0.9983), so the saved example latents use the FP32 reference. Original weight files remain unchanged BF16. This is a technical encoding check; math reconstruction and LLaDA conditioning are not yet validated.

## Frozen released-encoder math quality probe

Run: `outputs/pretrained-vae-quality-20261118-070113/`. The encoder remained frozen. A linear ridge readout was fitted on 640 controlled arithmetic statements, with regularization selected on 160 validation statements. Final evaluation used 160 statements with disjoint unordered operand pairs, integers 1–20, and new wording. No official GSM8K/DART test split was opened.

| Evaluation | Ordered operands + operation recovered | Equivalent arithmetic expression recovered |
| --- | --- | --- |
| Unseen pairs, new wording (primary) | 13/160 (8.125%) | 15/160 (9.375%) |
| Same unseen pairs, familiar notation (follow-up diagnostic) | 160/160 (100%) | 160/160 (100%) |

The follow-up applied the exact same trained readout and validation-selected regularization, with no refitting. It was added after inspecting the primary result and is reported as a diagnostic rather than an independent confirmatory evaluation. Addition/multiplication operand swaps count as mathematically equivalent; subtraction/division swaps do not. The readout recovers expression components; it does not solve the expressions.

On new wording, the full-latent readout identified the operation in 94.375%, the first operand in 41.25%, and the second operand in 11.25%. A same-backbone text-pooling baseline recovered ordered expression components in 5/160 with new wording and 157/160 with familiar notation; shuffled training labels recovered 0/160 in both settings. Full-latent features have more dimensions than the pooling baseline. The primary paired ordered-recovery comparison with the backbone had exact paired-binomial p=0.0963, so this experiment does not establish a reliable advantage over the backbone.

Changing a numeric literal changed latents on all 64 tested real GSM8K/DART dev sentence pairs. This establishes numerical sensitivity, not recovery of the original number or mathematical correctness. Four-choice paraphrase matching reached 47.5% with centered full-latent features (25% chance); centered pooled latents reached 53.75%. These are additional representation diagnostics.

Conclusion: the frozen encoder retains recoverable arithmetic information in familiar formats. Broad mathematical preservation, original VAE sentence reconstruction, and LLaDA conditioning remain unvalidated. Failure of a linear readout on new wording does not show that the encoder discarded the numbers. The release lacks the original VAE decoder bridge, preventing a direct reconstruction score.

Detailed results: `summary.json`; example predictions: `readout-predictions.jsonl` and `familiar-wording-predictions.jsonl`; real edits: `real-number-sensitivity.json`. Protocol checks and the CPU suite passed (26 tests).
