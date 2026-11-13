# Three-epoch small-model VAE pilot

Run directory: `outputs/vae-small-pilot-1000q-20261112-01/`.

Dataset: `data/processed/vae-pilot-1000q-v1/`.

## Data and scope

- 1,000 unique DART-Math training questions, one accepted response per question: 7,564 retained sentence/step targets.
- 100 development questions from the existing development split: 822 retained targets.
- A deterministic seed-42 question/response selection; no train/dev question or exact-normalized sentence overlap.
- All retained token sequences decode exactly to their source strings; EOS included; maximum 128 tokens; overlong steps excluded and counted, never truncated.
- One hundred sampled targets and two replacements were inspected. Two unsuitable source responses were excluded, with reasons preserved in `manifests/pilot_manual_exclusions.json`.
- The source corpus is not certified mathematically correct. This pilot measures reconstruction of supplied text, not theorem correctness, question answering, sycophancy, or safety.
- No test-set examples are loaded. The two earlier 16-sentence debugging runs remain unchanged.

The reviewed subset, token counts, source IDs, exclusions and file hashes are recorded in the dataset directory's `summary.json` and `review.json`.

## Training

Fresh revision-pinned `Qwen/Qwen3-0.6B-Base` encoder and separate decoder, not the earlier memorized VAE. The encoder, six queries, posterior heads and decoder interface are trainable; the decoder backbone is frozen.

- Six latent vectors of dimension 1,024 per sentence.
- FP32 trainable parameters and optimizer state, BF16 compute and frozen decoder weights.
- Learning rate 2e-5 with 100 warmup updates, AdamW, batch size 4, one optimizer update per batch.
- Three epochs; 1,891 batches/updates per epoch; 5,673 planned optimizer updates total.
- Every selected training example is included each epoch; no last-batch dropping.
- Clean encoder inputs, stochastic posterior training, KL weight 1e-5, hybrid attention masks and gradient checkpointing.
- Physical GPU 1, RTX 3090. Preflight at the maximum 128-token input length passed, peaking at approximately 12.30 GiB allocated memory. The preflight update was discarded; the actual pilot starts from fresh pretrained weights in a new process.
- No automatic extension past three epochs. No full-size model training is launched by this job.

The effective config and source snapshots are saved with the run. Configuration template: `configs/vae_small_pilot.yaml`.

## Live monitoring

```bash
cd /root/Thesis/VAE_Training
watch -n 2 .venv/bin/python scripts/pilot_status.py
```

The monitor uses only local JSON/CSV files and does not load a model or consume GPU memory. It shows epoch/batch progress, running training loss and token accuracy, an epoch-training ETA, evaluation progress, and the per-epoch metric table. The training ETA excludes validation and checkpoint time.

Detailed execution log:

```bash
tail -f /root/Thesis/VAE_Training/logs/vae-small-pilot-1000q-20261112-01.log
```

Raw status and completed-epoch metrics:

```bash
cat /root/Thesis/VAE_Training/outputs/vae-small-pilot-1000q-20261112-01/status.json
cat /root/Thesis/VAE_Training/outputs/vae-small-pilot-1000q-20261112-01/metrics.csv
```

GPU activity: `watch -n 2 nvidia-smi`.

Ctrl+C exits these monitors without stopping the detached pilot process. The actual PID and launch record are in `manifests/pilot_launch.json`.

## Epoch metrics

An epoch-0 `base` row measures the fresh, untrained VAE interface. Epochs 1, 2 and 3 add rows after their full evaluation/checkpoint phases. During validation, normal progress can continue while the training epoch percentage stays at 100%.

- **Train token accuracy:** correct teacher-forced token predictions divided by all real training tokens, including EOS and excluding padding, accumulated across that epoch's training batches. Weights change during these batches; this is an online epoch metric.
- **Validation token accuracy:** teacher-forced token accuracy over every selected development target, in evaluation mode using posterior mean codes.
- **Validation exact reconstruction:** greedy full-sentence reconstruction from the latent code plus AE token, followed only by generated tokens. The decoder is never supplied the source sentence as a prompt. Whitespace-normalized text must match the source completely. EOS and token-cap counts are reported separately.
- **Numeric preservation:** exact ordered numeric-string sequence on targets containing numbers. This is a strict textual metric, not algebraic equivalence or correctness of the original calculation.
- **Loss:** token-weighted reconstruction CE plus beta times example-weighted KL. Training uses sampled posterior codes; validation uses means. Raw CE/KL are saved separately.
- **Controls:** the same fixed 64 development targets are reconstructed using normal, zero and shuffled codes each epoch. Codes are shuffled within the fixed 16-example decode batches. Full-development exact match and control-subset exact match have different denominators.

Full-development and control reconstructions are stored under `epoch_00/` through `epoch_03/`, with one record per target. Decoding uses a 160-token cap; all source targets fit within 128 including EOS. There is no output correction or numeric replacement.

## Checkpoints and interpretation

Each completed epoch checks the frozen decoder hash, verifies all train/dev examples were counted, and saves an Accelerate checkpoint plus an exported VAE. All three epoch checkpoints are retained. `best_checkpoint.json` selects the highest full-dev exact match, breaking ties by lower development CE; test data is not used.

`acceptance_criteria.json` is written before training. `summary.json` is written only after all three epochs finish. A positive pilot signal requires improved full-development exact match over the fresh baseline and stronger normal-code than zero/shuffled-code reconstruction on the fixed control subset. This is a diagnostic signal, not a claim of statistical significance or benchmark reasoning performance.

Unit/integration checks passed before launch, including token-weighted accuracy, numeric mismatch detection, and one epoch callback per epoch including the final partial batch. Existing gradient, checkpoint, decoder-freeze and free-decoder alignment checks were retained.

Reproduce preparation with the same source data and a fresh directory:

```bash
python -m scripts.prepare_pilot --directory data/processed/NEW_PILOT --manual-exclusions manifests/pilot_manual_exclusions.json
```

The current runner is pinned to the reviewed `vae-pilot-1000q-v1` dataset. Start a separate reproduction run only when intended:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python -u -m scripts.run_pilot --directory outputs/NEW_PILOT_RUN
```
