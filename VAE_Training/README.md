# VAE encoder training workspace

Main workspace: `/root/Thesis/VAE_Training`.

Status: code, datasets and isolated `.venv` are prepared. The Qwen3-0.6B-Base VAE reached 16/16 exact training reconstruction after a lower-LR continuation to step 525, with 0/16 development reconstruction and 0/16 zero/shuffled-code controls. Gradient, frozen-decoder and checkpoint reload checks passed. The small overfit gate is now met; generalization remains unestablished. See `NUMERIC_DEBUG.md` for the latest run and monitoring commands, and `SMALL_MODEL_TEST.md` for the preserved step-500 result. The larger pilot and full-size run have not started.

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

Sentence-level math data and explicit parent-question splits are prepared; see `DATASET.md`, the saved validation/review reports, and `TODO.md`. The isolated training environment is verified. The small-model checkpoint, tokenized diagnostic subset, encoder-training configuration, reconstruction controls, one-update checks and 16/16 overfit check are complete. Next work is to prepare and evaluate the larger pilot, then assess the full-size memory strategy. Encoder-input corruption remains a later experiment. See `ENVIRONMENT.md` for optional DeepSpeed/multi-process setup.

The copied VAE itself needs only `vae/`; `ladir/` and `sft/` retain the code needed for the planned downstream encoder loading and latent export. Code-generation scorers and RL training packages were intentionally left out. DeepSpeed is an optional additional installation if the later memory strategy needs it.
