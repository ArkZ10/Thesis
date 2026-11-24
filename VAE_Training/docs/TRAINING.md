# Training

The user selected the mathematics paper as the authority: <https://arxiv.org/html/2510.04573v5>. Scientific settings and all unreleased/conflicting details are recorded in `configs/vae_llama_paper_math.json`. `configs/llama_zero3_resident_optimized.json` defines the execution backend.

The full LLaMA-3.1-8B base encoder is trainable. Learned query vectors produce Gaussian posterior parameters; samples are projected into a frozen LLaMA decoder prefix. Reconstruction loss plus KL trains the encoder and interface, while gradients pass through the frozen decoder.

| Setting | Current value |
| --- | --- |
| Latents | 4 × 512 |
| Epochs | 2 |
| Effective batch | 128 sentences |
| Physical batches | At most 32 sentences / 4096 padded tokens |
| Learning rate | 2e-5 |
| KL weight | 1e-5 |
| Input substitution probability | 0.3 |
| Extra latent noise standard deviation | 3 |
| Precision | BF16 live model/accumulation; FP32 Adam masters and moments |

Table 13 specifies four latents, while paragraph D.1 specifies six. We selected Table 13 and recorded the conflict. Constant scheduling, Adam hyperparameters, exact checkpoint revision, split/parser choices and evaluation settings remain local assumptions where unreleased. The implementation is not claimed to reproduce every author detail exactly.

## Remaining work before the full run

1. Verify the longest 1608-token target on the actual GPUs.
2. Verify full checkpoint/save/resume using the optimized backend. Earlier-backend evidence does not establish this.
3. Choose a feasible training scope or hardware plan: measured timing estimates 116–129 days for both full epochs, excluding evaluation and checkpoints.

Two valid batch-128 updates were already measured. The full run has not launched. Cleanup does not launch it.

Readiness and monitoring:

```bash
python scripts/launch_llama_pilot.py --check
watch -n 10 '.venv/bin/python scripts/llama_pilot_status.py --paper'
```

`--check` reports prepared-summary presence and disk readiness only; it does not certify longest-target GPU fit or optimized checkpoint/resume. The launcher checks available RAM, disk and GPU occupancy before execution and supervises remaining resources. Running `python scripts/launch_llama_pilot.py` **starts training** with the current paper configuration; run it only after completing the remaining work and choosing scope.

`run_llama_pilot.py` is the worker used by the launcher. Source snapshots, status, logs, per-epoch token accuracy, exact reconstruction and numeric preservation are written into the new run. Teacher-forced accuracy is not complete-sentence correctness; numeric preservation compares literal numeric strings, not mathematical equivalence.

`export_preserved_vae_weights.py` exports exact FP32 weights from a complete native LLaMA checkpoint. It cannot recover removed optimizer state. The retained pilot already has a verified export and does not need re-exporting.

## Smaller encoder debugging

The stopped fallback experiment uses Qwen3-0.6B as the fully trained encoder and frozen Qwen3-4B as the reconstruction decoder. A learned projection handles their different hidden widths (1024 and 2560). The latent interface remains six vectors of dimension 512; it does not become an input to LLaDA until an explicit adapter is trained and evaluated.

`configs/vae_small.json` records all settings. The bounded clean overfit phase disables substitution, posterior sampling and extra latent noise, checking 16 training sentences every 25 updates, for at most 600 updates. If it fails to reconstruct all 16 exactly after whitespace normalization, the broader pilot does not start. This isolates reconstruction behavior before imposing robustness requirements.

If the gate passes, a 12-epoch pilot trains on 1024 sentences with 128 held-out dev sentences from the existing DART/GSM8K splits. Posterior noise ramps to a reduced scale of 0.1 over four epochs. This is a staged local experiment, not exact paper reproduction; full-strength robustness is not demonstrated by this run.

Every pilot epoch saves training-sample and dev token accuracy, free reconstructions, literal numeric preservation, normal/zero/shuffled latent-prefix controls, a latest encoder/optimizer state, and the best encoder selected by dev numeric preservation then exact matching then token accuracy. Only one latest checkpoint and one best encoder are retained. No archives are made. The frozen decoder is hash-checked.

```bash
python scripts/train_small.py --check
python scripts/train_small.py --status
# Starts a new bounded experiment when GPUs are free:
python scripts/train_small.py --launch
```

Run paths and logs are indexed by `manifests/small_training.json`. A failed or completed bounded run is not automatically proof that the encoder is good; inspect its summary and reconstructions before advancing.

## Reusing the released encoder

The current candidate reuses the author's trained encoder from `mk322/LaDiR-Code-8B`. `scripts/pretrained_vae.py` downloads only the contiguous encoder tensor range with an exact HTTP Content-Range check, removes the `vae.` namespace prefix, and preserves the original BF16 values. Partial downloads can resume; no original checkpoint archive is kept.

`vae/released_encoder.py` contains only the required official encoder architecture and its padding-position helper, attributed to the upstream revision. The original `/root/LaDiR` checkout is untouched. The encoder uses the published loader's causal mask and returns posterior means of shape `[batch, 64, 2560]`.

The smoke check strictly validates every state-dict key/shape, tests finite and repeatable math encoding and compares single-example versus padded-batch representations. These checks do not prove reconstruction quality or useful math reasoning. The code-domain checkpoint's original full VAE decoder bridge is not present in the extracted encoder; only the encoder is reused.

Before LLaDA training, evaluate mathematical information preservation and define the learned mapping from these latent vectors into LLaDA's input space. An encoder that loads successfully does not establish that LLaDA uses its latents. Training remains a separate stage.

The actual loading/encoding test passed. For canonical cached targets, use the FP32 reference described in `outputs/pretrained-vae-check/summary.json`: BF16 inference had small batch-rounding differences, while FP32 batch/single outputs agreed closely. Source checkpoint precision is still BF16; upcasting for inference is exact and does not recover precision absent from the source.
