# Larger-model VAE pilot

Run directory: `/root/Thesis/VAE_Training/outputs/vae-large-pilot-1000q-20261112-140652`. Log: `/root/Thesis/VAE_Training/logs/vae-large-pilot-1000q-20261112-140652.log`. This is the authorized three-epoch pilot, initialized from fresh pinned Qwen3-4B-Base encoder and Qwen3-8B-Base decoder checkpoints. Smoke and resume-test updates are discarded. The encoder is on GPU 1, frozen decoder on GPU 0; full FP32 CPU AdamW uses streamed gradient/weight transfers.

## Monitoring

```bash
cd /root/Thesis/VAE_Training
watch -n 2 .venv/bin/python scripts/large_status.py
```

```bash
tail -f /root/Thesis/VAE_Training/logs/vae-large-pilot-1000q-20261112-140652.log
```

The monitor is read-only and uses no GPU. Ctrl+C stops only the monitor. The run is detached and does not depend on an open terminal or chat. An automatic guard stops this job only if available host RAM falls below 2 GiB; failures and exceptions appear in status/log files.

## Data and schedule

- 1,000 training questions / 7,564 sentence targets; 100 dev questions / 822 targets. Existing question splits preserved; no test set is loaded.
- Three epochs, batch size 16, 473 batches/updates per epoch and 1,419 updates total. Every selected training sentence is included each epoch; the final batch has four sentences. No automatic extension.
- Six 2560-dimensional latents, encoder fully trained, decoder backbone frozen, posterior sampling enabled. KL weight 1e-5, LR 2e-5 after 100 warmup updates, clipping norm 1. Extra latent noise and encoder-input substitution remain disabled for this baseline pilot.
- The checkpoint-resume prerequisite passed on the real large checkpoint, and all 24 regression tests passed before launch.

## Progress and metrics

First the process loads the models and evaluates the fresh model (epoch `base`/0). Epoch 1 starts after this baseline evaluation. The monitor shows the stage, epoch, completed batches, percentage, running training loss/token accuracy and estimated remaining epoch training time. The training ETA excludes evaluation/checkpoint time and becomes useful after several updates.

`metrics.csv` gains a row for the baseline and after each completed epoch. Training token metrics cover all training batches; dev token metrics and free reconstruction cover all 822 dev targets. A fixed 128-training-sentence sample is freely reconstructed each epoch. Normal/zero/shuffled codes are compared on the same fixed 64 dev targets, using both free reconstruction and teacher-forced CE/token accuracy. Greedy reconstruction uses a 160-token cap and posterior means.

Token accuracy uses correct preceding text. Exact reconstruction compares the whole generated sentence after whitespace normalization. Numeric preservation compares ordered literal numeric strings in targets containing numbers; it is not mathematical correctness or algebraic-equivalence judging. Raw outputs and full metrics are saved under `epoch_00/` through `epoch_03/`.

## Checkpoints and space

Each completed epoch retains sharded trainable weights, RNG and metadata under `checkpoints/epoch_XX/`. The final epoch additionally saves the complete FP32 masters and Adam states. Intermediate epoch exports do not support exact optimizer resumption. This final-only optimizer policy keeps new artifacts around 71 GiB, preserving all earlier smoke/pilot results within the available disk space. Checkpoints are committed from incomplete directories only after writing and tensor verification finish.

`best_checkpoint.json` selects highest full-dev exact reconstruction, then lower dev CE. `summary.json` is written after all three epochs complete. A complete final checkpoint contains the state needed for future continuation; the current pilot command has no resume CLI. Full-corpus training and robustness experiments remain separate later stages.
