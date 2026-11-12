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

Remaining:

1. Define pilot acceptance criteria, select/review approximately 1,000 training questions, and tokenize/audit lengths without changing existing split assignments.
2. Run the larger small-model pilot and evaluate held-out reconstruction, quantities, equations, completion and latent use; the 16-example development result is still 0/16.
3. Add and compare encoder-input corruption/extra latent noise with clean reconstruction targets.
4. Assess memory and optional DeepSpeed/offloading for the intended full-size LaDiR configuration before downloading and training the larger encoder/decoder pair.
5. Scale only after the pilot passes; freeze/export the selected encoder and ordered sentence-level latent blocks for later guided LLaDA work.

Only the small-model diagnostic has run. The larger pilot and full-size training have not started. Reconstruction tests do not measure mathematical problem-solving, sycophancy, or safety.
