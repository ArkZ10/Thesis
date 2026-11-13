# VAE training progress

Completed:

- [x] Copy reusable VAE, model and latent-export code into this workspace.
- [x] Download revision-pinned DART-Math-Hard and GSM8K raw datasets and verify their checksums.

- [x] Prepare sentence-level math data, coordinated parent-question splits, provenance, exclusions, and an independent validation report. Final output: `data/processed/math-sentences-v1/`; 5,478,995 unique reconstruction targets, 100 inspected examples, 15 passing tests.
- [x] Create and validate `.venv`: installed project dependencies; both GPUs passed BF16 checks; 15 tests and a tiny offline VAE gradient probe passed. See `ENVIRONMENT.md` and `manifests/environment.json`.
- [x] Download and checksum-verify the small Qwen3-0.6B-Base checkpoint and tokenizer.
- [x] Select, inspect and tokenize the 16-training/16-development diagnostic subset without changing its original splits.
- [x] Create the small-model encoder-training config and verify real-data gradients, frozen decoder weights, checkpoint save/load, and exported latent means.
- [x] Implement free reconstruction and zero/shuffled latent controls; 16 unit/integration tests passed.
- [x] Execute the 500-step small-model overfit diagnostic: 15/16 training and 0/16 development reconstructions exact. The strict 16/16 gate remains unmet. See `SMALL_MODEL_TEST.md`.
- [x] Resolve the small diagnostic's numerical mismatch: continue at LR 2e-5 for 25 steps, reaching 16/16 exact training reconstruction at step 525. Development remains 0/16. Frozen decoder, zero/shuffled controls, and all 32 reloaded outputs verified. All 18 tests passed. See `NUMERIC_DEBUG.md`.
- [x] Prepare and inspect the 1,000-training-question/100-development-question pilot: 7,564 and 822 tokenized targets, no overlap/truncation, two manual response exclusions, explicit acceptance criteria, and a 128-token memory preflight. All 20 tests passed.
- [x] Complete and review the three-epoch pilot: best epoch 3, development token accuracy 70.87%, exact reconstruction 5/822, numeric preservation 20/681. Frozen decoder unchanged; test unused. The generalization gate remains unmet because the 64-example normal/zero/shuffled controls all scored zero exact matches. See `outputs/vae-small-pilot-1000q-20261112-01/summary.json`.

- [x] Download and checksum-verify pinned Qwen3-4B-Base and Qwen3-8B-Base checkpoints; verify both tokenizers on the unchanged question-disjoint pilot subset.
- [x] Measure large-model memory routes. ZeRO-3 trials exceeded host/GPU memory; explicit encoder GPU 1 / frozen decoder GPU 0 with streamed FP32 CPU AdamW passed two batch-16 updates. Full decoder hash unchanged; every saved trainable tensor checked; full FP32 masters and Adam moments saved. All 23 tests passed. See `LARGE_MODEL_TEST.md`.

- [x] Restore the real full large checkpoint: all 406 live tensors, FP32 master weights, Adam states and RNG verified; one continuation update passed with frozen decoder unchanged and was discarded.
- [x] Implement and launch the three-epoch larger-model pilot with epoch/batch progress, ETA, per-epoch train/dev metrics and a disk-aware checkpoint policy. All 24 tests passed. See `LARGE_PILOT.md`.

Remaining:

1. Review reconstruction errors from the completed larger-model pilot, including length effects, incorrect numbers/equations, and semantically equivalent paraphrases.
2. Compare the larger-model pilot with the small-model pilot, accounting for changed model size, latent width and batch size.
3. Add and compare encoder-input substitution and extra latent noise while preserving clean reconstruction targets.
4. Estimate full-corpus runtime/storage and train only after pilot quality and resources are satisfactory; select/freeze/export the encoder and ordered sentence latent blocks for later guided LLaDA work.

The three-epoch larger-model pilot completed: best epoch 3, exact dev reconstruction 154/822, numeric preservation 304/681, and normal/zero/shuffled exact controls 13/0/0 out of 64. Full-corpus training and test evaluation have not started. Reconstruction metrics do not measure mathematical problem-solving, sycophancy or safety.
