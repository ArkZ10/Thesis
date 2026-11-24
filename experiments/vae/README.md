# LaDiR VAE training workspace

This directory contains VAE encoder-training experiments. The published LaDiR VAE encoder is the current candidate; the full LLaMA DART-MATH run has **not started**.

- [Environment and dependencies](docs/SETUP.md)
- [Datasets and preparation](docs/DATA.md)
- [Training configuration and remaining checks](docs/TRAINING.md)
- [Retained experiment results](docs/RESULTS.md)

## Current commands

Run these from `/root/Thesis/experiments/vae`:

```bash
source .venv/bin/activate
python scripts/pretrained_vae.py --status
python scripts/launch_llama_pilot.py --check
python scripts/llama_pilot_status.py --paper
python scripts/llama_pilot_status.py
```

The pretrained-VAE status command shows the current download. The small-model experiment was stopped after preserving its latest checkpoint. The LLaMA `--check` command reports data/disk readiness without starting training. `--paper` monitors the intended full-data run; the final command shows the retained completed pilot, including per-epoch accuracy.

## Retained results

`outputs/vae-llama-mixed-pilot-20261114-181908/` holds the latest completed training, reconstructions, metrics and exact FP32 encoder weights. It used six latents and mixed DART+GSM8K data.

`outputs/llama-execution-profile-20261116-220708/` holds the latest valid execution test of the four-latent paper configuration. It validates two optimizer updates, not a completed training run.

Old experiments, obsolete code and superseded documents were deleted directly. No archive was created. Raw datasets, the current prepared data, cached pretrained models and `.venv` are preserved. The original `/root/LaDiR` checkout is separate and untouched.

## Small-encoder fallback experiment

`configs/vae_small.json` selects the cached Qwen3-0.6B encoder and frozen Qwen3-4B decoder. `scripts/train_small.py` owns training, launch and monitoring. Encoder weights and Adam state are FP32 with BF16 computation; no optimizer NVMe offload is used.

```bash
watch -n 10 '.venv/bin/python scripts/train_small.py --status'
```

The first phase requires 16/16 training-sentence reconstructions before a 1024-train/128-dev pilot starts. The dev set is question-disjoint; test data are reserved. Engineering targets are at least 95% dev teacher-forced token accuracy, 90% literal numeric preservation and 50% normalized exact reconstruction. Passing these targets still requires equation review, latent-use controls, noise robustness and a separate LLaDA adapter evaluation. These are local debugging criteria, not the authors' published thresholds.

## Current pretrained encoder

The official [LaDiR release](https://huggingface.co/mk322/LaDiR-Code-8B) bundles a trained Qwen3-4B VAE encoder. We extract only its `vae.*` tensor range from revision `aab72a886105d0e53b0c1c5c3a02d55a9228a7c0`: about 8.85 GB, without downloading the 25.5 GB reasoning model. Original BF16 values are preserved.

```bash
python scripts/pretrained_vae.py --status
# Resume a partial transfer and run the encoding check:
python scripts/pretrained_vae.py --download --test
```

Weights live under `models/LaDiR-Code-VAE/`; provenance is in `manifests/pretrained_vae.json` after completion. The encoding check writes `outputs/pretrained-vae-check/summary.json` and `math-latents.pt`.

This encoder returns 64 latent vectors of width 2560. It is the code-domain release, not the mathematics paper's 4/6 × 512 encoder. Strict loading and finite latent outputs establish technical usability; mathematical information preservation and LLaDA compatibility require separate evaluation. The released `vae.*` component excludes the original VAE decoder bridge, so this encoding check is not a reconstruction-accuracy result.

## Encoder quality evaluation

The frozen arithmetic-information evaluation is complete; see [results and limits](docs/RESULTS.md#frozen-released-encoder-math-quality-probe). It recovered the ordered numbers and operation for 160/160 unseen combinations in familiar notation, versus 13/160 with new wording using the same fitted readout. This establishes some recoverable information, not general math reconstruction quality.

```bash
python scripts/pretrained_vae.py --quality-status
# Repeat the complete frozen evaluation:
python scripts/pretrained_vae.py --quality
```

The encoder is never updated by this evaluation. Only a small readout is fitted, with separate train/validation/held-out groups and shuffled-label/backbone controls.
