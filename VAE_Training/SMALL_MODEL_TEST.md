# Small-model VAE diagnostic

Historical result at step 500: the real-checkpoint training/gradient/checkpoint path passed, but only 15/16 training examples reconstructed exactly. A later, separately saved lower-LR continuation reached 16/16 at step 525; see `NUMERIC_DEBUG.md`. This document retains the original step-500 evidence. Development reconstruction remained 0/16. No larger pilot or full-size training was started.

Run directory: `outputs/vae-small-diagnostic-20261112-01/`.

## Setup

- Encoder and decoder: two separate copies of `Qwen/Qwen3-0.6B-Base`, revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Every downloaded file passed its upstream checksum.
- Encoder, six learned queries, posterior heads and decoder interface trainable; decoder backbone frozen.
- Six latent vectors of dimension 1,024 per sentence.
- FP32 trainable parameters and optimizer state; BF16 compute and frozen decoder weights.
- Hybrid masks, gradient checkpointing, stochastic posterior training, posterior mean evaluation, KL weight 1e-5.
- Sixteen short sentences from 16 distinct prepared GSM8K training questions; 16 from distinct development questions. No official test examples used. All selected calculator annotations passed exact rational arithmetic checks. This does not certify the full source reasoning.
- All selected targets fit without truncation, including EOS. Selection and token IDs are saved in the run directory.
- Physical GPU 1, RTX 3090. Peak allocated CUDA memory approximately 12.29 GiB.
- Batch size 4, AdamW, learning rate 1e-4, 10 warmup steps. Initial budget 400 steps, followed by a separately recorded 100-step checkpoint continuation, stopping at 500 total.

This is a smaller-model adaptation for debugging the codec, not the full-size LaDiR pipeline, guided LLaDA, or MCTS.

## Results

| Free reconstruction condition | Training exact match | Development exact match |
|---|---:|---:|
| Before training, correct codes | 0/16 | 0/16 |
| After 400 steps, correct codes | 15/16 | 0/16 |
| After 500 steps, correct codes | 15/16 | 0/16 |
| After 500 steps, zero codes | 0/16 | 0/16 |
| After 500 steps, shuffled codes | 0/16 | 0/16 |

Free reconstruction supplies only projected latent codes plus the learned AE token to the decoder; subsequent inputs are generated tokens. The original text is supplied to the encoder as expected in a compression/reconstruction task. It is not supplied as a decoder prefix. Teacher-forced token accuracy was approximately 99.92% in the final training logging window and 57.91% on development data; these are distinct from free sentence exact match.

All 16 normal-code training and development reconstructions reached EOS. Zero-code outputs reached the 64-token cap. All 16 shuffled training outputs matched the corresponding rotated normal outputs, confirming that the codes distinguish these memorized examples.

The run used approximately 257 seconds of active diagnostic time, including model loading, training, checkpoint work and reconstruction checks across both phases. Package installation, download and script preparation are outside that timing.

One failure after 500 steps:

```text
Target:
Since Johnson got $2500, each part is therefore $2500/5 = $<<2500/5=500>>500

Reconstruction:
Since Johnson got $250, each part is therefore $250/5 = $<<250/5=50>>50
```

The changed quantities are a real reconstruction error. Training loss alone would conceal this failure. The 400-step run had a different numerical mismatch; both results and all examples are preserved.

## Verification

- Finite, nonzero first-update gradients in encoder, queries, posterior mean/variance heads, and decoder interface.
- Encoder weights changed; every frozen decoder parameter remained unchanged, verified by parameter hashing.
- Actual training checkpoint save/load reproduced posterior means bit-for-bit.
- Final exported VAE weights reloaded into a fresh model and reproduced posterior means bit-for-bit.
- A separate fresh-process reload regenerated all 32 final normal-code token sequences identically (16 training and 16 development), and its decoder hash matched the original frozen checkpoint. See `export-reload.json`.
- Sixteen unit/integration tests passed. The new reconstruction check verifies that the free decoder's first-token prediction matches the teacher-forced decoder's aligned first-token prediction.

This establishes a functioning small-model VAE training path and use of the latent codes on the training set. It does not establish generalization, mathematical reasoning quality, numerical reliability, sycophancy reduction, or full-size memory feasibility. The strict overfit gate remains open.

## Artifacts and commands

- `summary.json`: final 500-step summary; `summary-400.json` preserves the initial result.
- `verification.json`, `one_update.json`, `export-reload.json`: checks and the remaining mismatch.
- `before-*`, `after-*`, `after-500-*`: every reconstruction and control result.
- `config.yaml`, `continuation-config.yaml`, `selection.json`, `train.jsonl`, `dev.jsonl`: effective configurations, provenance and exact examples.
- `training/checkpoints/step_1`, `step_400`, `step_500`: complete Accelerate checkpoints.
- `training/weights/step_400.safetensors`, `step_500.safetensors`: exported VAE weights.
- `source/`: run-specific source snapshots; `history*.json`: optimization metrics.
- Workspace logs: `logs/vae-small-diagnostic-20261112-01*.log`.

Reproduction from a fresh directory:

```bash
cd /root/Thesis/VAE_Training
source .venv/bin/activate
python scripts/run_small_diagnostic.py --directory outputs/NEW_DIAGNOSTIC --prepare-only
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python -u scripts/run_small_diagnostic.py --directory outputs/NEW_DIAGNOSTIC
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python -u scripts/run_small_diagnostic.py --directory outputs/NEW_DIAGNOSTIC --continue-to 500
```

Use the runner: it promotes trainable parameters to FP32 and records the extra integrity checks. Running the upstream CLI directly with the same YAML does not perform that promotion or those diagnostic checks.

Next work is to resolve the remaining numerical reconstruction failure and define pilot acceptance criteria before committing to the larger pilot or full-size model.
