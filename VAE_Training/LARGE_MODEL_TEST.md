# Larger VAE setup and smoke tests

The current run is `outputs/vae-large-split-smoke-20261112-02`. Its `summary.json` and the run-specific launch manifest are authoritative. This stage prepares a Qwen3-4B encoder and frozen Qwen3-8B decoder as a LaDiR adaptation; it is not an exact reproduction of the paper. No multi-epoch pilot or full-data training has started.

## Models and data

- Encoder: `models/Qwen3-4B-Base`, revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- Frozen decoder: `models/Qwen3-8B-Base`, revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.
- Every downloaded file's upstream checksum was verified; see `manifests/qwen3_4b_base.json` and `manifests/qwen3_8b_base.json`.
- `data/processed/vae-large-comparable-v1/` retains the previous pilot's 1,000 training questions / 7,564 sentences and 100 dev questions / 822 sentences. Both model tokenizers reproduce all prior token IDs and round-trip the sentences. No question/sentence overlap or truncation; no test data is used.
- The same targets support comparison; this is not a newly cleaned or mathematically certified dataset. Two long training targets were separately inspected. The final memory test repeats them to form a batch of 16; these are not 16 distinct questions.

## Working execution path

`configs/vae_large_split_smoke.yaml` and `scripts/run_large_split_smoke.py` use explicit placement:

- GPU 1: full trainable Qwen3-4B encoder, latent queries and posterior heads.
- GPU 0: frozen Qwen3-8B decoder, trainable latent projection and AE embedding.
- CPU: FP32 master weights and full AdamW moment states for every trainable parameter. Gradients and updated parameters are streamed one parameter at a time, avoiding a second complete CPU gradient copy.

`vae/split_training.py` transfers latent representations between GPUs without detaching their gradients. Training uses BF16 backbone weights/computation, FP32 new trainable heads/queries, FP32 CPU masters/moments and FP32 cross-entropy. The vocabulary projection uses BF16 multiplication to avoid a large FP32 weight-copy temporary. The decoder remains frozen; there is no backbone quantization or LoRA.

The latent shape is 6 x 2560. Posterior sampling is enabled, extra latent noise/input corruption are off, KL weight is 1e-5 and learning rate is 2e-5. The bounded test performs two updates at batch size 16, checking the second update after Adam states exist. It verifies finite nonzero gradients through six representative paths, FP32 master parameter changes, complete frozen-decoder hashes, free reconstruction and checkpoint saving. It is not an overfit or accuracy test.

## Checkpoints

The final format stores sharded trainable model tensors, all unrounded CPU FP32 master weights, CPU Adam states, parameter names, step count, RNG states and configuration. Frozen decoder weights are excluded and must be restored from the pinned model source. Every saved trainable tensor is checked against its live model value. The optimizer regression tests compare updates with PyTorch AdamW and verify exact next-update equivalence after restoring unrounded masters and moments. A later full checkpoint-resume verification restored all 406 trainable tensors, FP32 masters, Adam states and RNG, then continued from step 2 to step 3 successfully; that probe update was discarded. See `manifests/large_resume_verification.json`.

The initial split smoke (`...split-smoke-20261112-01`) passed two updates at batch size 2, frozen-weight checks, reconstruction and trainable tensor saving. Its optimizer file omitted unrounded master weights, so it cannot provide exact optimizer resumption; its summary explicitly records this limitation. The corrected final run starts fresh.

## Earlier memory attempts

ZeRO-3 trials are preserved separately and must not be used as the pilot launcher:

1. `vae-large-smoke-20261112-01`: forward/backward passed; 75% CPU optimizer-subgroup offload hit the host memory guard during the first Adam step.
2. `...-02`: 65% CPU offload exceeded GPU memory during Adam state initialization.
3. `...-03`: 70% CPU offload passed one update and unchanged decoder-shard checks, but free reconstruction exceeded GPU memory.
4. `...-04`: 72% CPU offload and BF16 output projection passed the first update but exceeded GPU memory on the second forward/backward pass.

The final explicit-placement path avoids these repeated backbone all-gathers. DeepSpeed 0.19.9 is retained for its compiled CPU Adam kernel, not a ZeRO engine. `vae/hybrid_checkpoint.py` and its test preserve GPU Adam sidecars for the archived hybrid-optimizer route; they are unnecessary for the final all-CPU optimizer state.

## Environment and monitoring

DeepSpeed 0.19.9, Ninja and their dependencies were installed only in `.venv`; original Torch/Transformers versions remain unchanged. See `requirements-large-venv.lock.txt` and `manifests/deepspeed-install-report.json`. CPU Adam was compiled and verified. `DS_SKIP_CUDA_CHECK=1` bypasses the toolkit gate for host CUDA 12.8 versus PyTorch CUDA 12.4; no CUDA optimizer extension is used by the working path.

```bash
cd /root/Thesis/VAE_Training
watch -n 2 .venv/bin/python scripts/large_status.py
```

```bash
tail -f /root/Thesis/VAE_Training/logs/vae-large-split-smoke-20261112-02.log
```

The monitor follows `manifests/large_smoke_latest.json`. Ctrl+C stops the monitor, not the job. The wrapper records exit status and minimum available host RAM, and stops only its own process group if available host memory falls below 2 GiB. Runs and logs are never overwritten.

Next: implement the multi-epoch pilot loop for this explicit placement, verify full checkpoint resume and plan disk retention. The existing Accelerate single-device pilot runner must not be used because its model preparation can move both backbones onto one GPU. The new pilot monitor must show epoch/batch counts, progress percentage, ETA, train/dev loss and token accuracy, free dev exact reconstruction and numeric preservation each epoch. Full-data training and robustness augmentations remain later experiments.

## Final measured result

The final batch-16 test completed successfully in 157.93 seconds, including model loading and checkpoint saving. The two updates took 20.69 and 17.73 seconds. The batch contained repeated copies of the two inspected long targets; no dev/test accuracy conclusion follows. GPU peak allocations were 19.42 GiB (decoder GPU 0) and 16.91 GiB (encoder GPU 1). Minimum available host RAM was 7.11 GiB. Full decoder hashes matched, all 406 saved trainable tensors matched the live values, and the 48.55 GB optimizer file was read back successfully with all 406 FP32 masters and Adam moment entries present. All 23 regression tests passed.

Free reconstruction executed but did not reproduce the fixture after only two updates; the generated text was Java code and hit the 32-token cap. This confirms the generation path runs, not that reconstruction is useful yet. No development or test evaluation was performed here. See `summary.json`, `checkpoint_audit.json`, `source/` and `model_sources.json` in the final run directory.

The multi-epoch pilot loop and monitor have now been implemented and launched after full checkpoint-resume verification. Current experiment instructions and retention policy are in `LARGE_PILOT.md`; the earlier smoke-test sections describe the completed setup history.
