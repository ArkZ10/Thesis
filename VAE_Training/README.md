# VAE encoder training workspace

Main workspace: `/root/Thesis/VAE_Training`.

Status: the authorized three-epoch larger-model pilot completed in `vae-large-pilot-1000q-20261112-140652`, with a trainable Qwen3-4B encoder on GPU 1, frozen Qwen3-8B decoder on GPU 0 and streamed FP32 CPU AdamW. It uses 7,564 training sentences and 822 dev sentences, batch size 16 and 473 batches per epoch. Full checkpoint resumption passed and all 24 tests passed. Monitor: `python scripts/large_status.py`. See `LARGE_PILOT.md` for stage/metric definitions and checkpoint retention. Full-dataset training and test evaluation have not started. Best epoch: 3, with 154/822 exact dev reconstructions (18.73%), 87.15% dev token accuracy and 304/681 numeric matches (44.64%). Correct latent codes achieved 13/64 exact controls versus 0/64 for zero/shuffled codes. See the run summary.

The prior small diagnostic reached 16/16 exact training reconstruction at step 525 and 0/16 development reconstruction. Those artifacts are preserved in `NUMERIC_DEBUG.md` and `SMALL_MODEL_TEST.md`. The larger pilot starts from fresh pretrained small-model weights to assess generalization.

Sentence preprocessing produced 5,478,995 unique reconstruction targets. DART-Math-Hard has 585,392 raw training rows; GSM8K main has 7,473 raw training rows and 1,319 raw test rows. Downloaded dataset/model files passed upstream checksums. See `DATASET.md`, `ENVIRONMENT.md`, and the manifests.

Activate the installed environment with `source /root/Thesis/VAE_Training/.venv/bin/activate`. Exact package versions and checks are recorded in `manifests/environment.json` and `requirements-venv.lock.txt`.

## Layout

- `vae/`: original VAE model, data pipeline, trainer, and upstream configuration.
- `scripts/train_vae.sh`: upstream Accelerate launcher.
- `ladir/`: model components, including the frozen VAE encoder used downstream.
- `sft/`: latent-export, weight-loading, and token-data utilities, with their package dependencies preserved.
- `scripts/download_datasets.py`: downloads pinned datasets, checks upstream hashes, and reads every parquet row.
- `data/raw/dart-math-hard/`: main mathematical reasoning dataset, original files.
- `data/raw/gsm8k/main/`: original GSM8K main train/test files. Test is reserved for evaluation.
- `data/processed/math-sentences-v1/`: prepared sentence-level datasets and question-level train/dev/test splits; see `DATASET.md` and `validation.json`.
- `manifests/`: source provenance, dataset revisions, file checksums, schemas, row counts, and status.
- `configs/`: reserved for experiment configurations.
- `outputs/`, `logs/`: experiment artifacts and execution logs, including the completed small-model diagnostic.
- `PLAN.md`: the implementation plan created before this preparation step. Its original suggested workspace path is superseded by this directory.

The initial copied source files came from `/root/LaDiR` at commit `5eaa5968393e5e93262bb53b867683704dd1da3a`; `manifests/source.json` retains their original hashes. Local changes include explicit-split/short-math support in `vae/data.py`, an optional diagnostic step callback in `vae/train.py`, preparation/validation/reconstruction modules, test runners and tests. The original checkout is preserved. `UPSTREAM_README.md` and `manifests/upstream_pyproject.toml` retain upstream documentation and dependency declarations.

## Dataset download

Using the existing Python environment, without installing or modifying packages:

```bash
cd /root/Thesis/VAE_Training
/opt/miniforge3/envs/diffuguard/bin/python scripts/download_datasets.py
```

This command resumes pinned downloads and verifies their checksums. It requires network access. It does not start training, load a model, call a paid judge, or segment examples.

Datasets:

- [hkust-nlp/dart-math-hard](https://huggingface.co/datasets/hkust-nlp/dart-math-hard), revision `0c2603fba274c2e83ec9121bee3067e286ced194`.
- [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), revision `740312add88f781978c0658806c59bc2815b9866`, `main` subset only.

Dataset cards are kept alongside the raw files. Multiple DART-Math responses to the same question must stay in the same split. GSM8K and DART-Math share source questions; they must not be treated as independent unseen datasets without checking overlaps.

## Before training

The upstream `vae/configs/vae_4b_8b_64.yaml` is a reference, not the proposed experiment config: it freezes the encoder, trains the decoder, and filters for Python. The upstream launcher also defaults to eight GPUs. A sentence-math config must reverse the encoder/decoder backbone training flags, configure the latent count, and use the measured GPU allocation.

Sentence-level math data, the reviewed/tokenized pilot subset, and explicit parent-question splits are prepared; see `DATASET.md`, `PILOT.md`, and `TODO.md`. The isolated training environment, gradient/checkpoint checks and 16/16 overfit check are complete. The three-epoch pilot completed with a fresh baseline, full-development evaluation each epoch, and normal/zero/shuffled-code controls. Epoch 3 achieved 5/822 exact development reconstructions and 20/681 numeric matches; all three 64-example controls had zero exact matches, so the predefined generalization gate remains unmet. See the run summary before further experiments. Encoder-input corruption and the full-size memory strategy remain later work; see `ENVIRONMENT.md` for optional DeepSpeed/multi-process setup.

The copied VAE itself needs only `vae/`; `ladir/` and `sft/` retain the code needed for the planned downstream encoder loading and latent export. Code-generation scorers and RL training packages were intentionally left out. DeepSpeed is an optional additional installation if the later memory strategy needs it.

The larger-model pilot now has its own loop, live epoch monitor and disk-retention policy; full checkpoint resumption was verified before launch. The earlier single-GPU launcher remains separate.
