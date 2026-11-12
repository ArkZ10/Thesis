# VAE training progress

Completed:

- [x] Copy reusable VAE, model and latent-export code into this workspace.
- [x] Download revision-pinned DART-Math-Hard and GSM8K raw datasets and verify their checksums.

- [x] Prepare sentence-level math data, coordinated parent-question splits, provenance, exclusions, and an independent validation report. Final output: `data/processed/math-sentences-v1/`; 5,478,995 unique reconstruction targets, 100 inspected examples, 15 passing tests.
- [x] Create and validate `.venv`: installed project dependencies; both GPUs passed BF16 checks; 15 tests and a tiny offline VAE gradient probe passed. See `ENVIRONMENT.md` and `manifests/environment.json`.

Remaining:

1. Select and download the encoder/decoder model checkpoints; start with the planned small-model diagnostic before assessing larger-model memory requirements.
2. Select and review the pilot subset, then tokenize with the selected tokenizer; record length exclusions and preserve all current split assignments. Accepted source reasoning is not certified mathematically correct.
3. Create the encoder-training configuration: six latent vectors per sentence, train encoder/queries/heads/interface, freeze decoder backbone, KL weight 1e-5. Install/validate optional DeepSpeed only if the selected memory strategy requires it.
4. Add encoder-input corruption with clean decoder targets, and standalone reconstruction evaluation with zero/shuffled latent controls.
5. Run a one-update real-data gradient/memory/checkpoint test.
6. Run the 16-sentence overfit diagnostic.
7. Run the approximately 1,000-question math pilot and evaluate held-out reconstruction, numbers, equations, completion and latent use.
8. Scale only after the pilot passes; freeze and export the selected encoder and ordered sentence-level latent blocks for later guided LLaDA work.

Training has not started. Preparing reconstruction targets does not certify mathematical correctness or measure sycophancy/safety.
