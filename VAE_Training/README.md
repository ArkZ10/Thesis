# VAE encoder training workspace

Main workspace: `/root/Thesis/VAE_Training`.

Preparation status: LaDiR VAE code copied; raw datasets downloaded and verified. DART-Math-Hard has 585,392 training rows; GSM8K main has 7,473 training rows and 1,319 test rows. Every downloaded file passed its upstream checksum, and every parquet row was read successfully. See `manifests/preparation.json` and `manifests/download_status.json`. Training and model downloads have not been started.

## Layout

- `vae/`: original VAE model, data pipeline, trainer, and upstream configuration.
- `scripts/train_vae.sh`: upstream Accelerate launcher.
- `ladir/`: model components, including the frozen VAE encoder used downstream.
- `sft/`: latent-export, weight-loading, and token-data utilities, with their package dependencies preserved.
- `scripts/download_datasets.py`: downloads pinned datasets, checks upstream hashes, and reads every parquet row.
- `data/raw/dart-math-hard/`: main mathematical reasoning dataset, original files.
- `data/raw/gsm8k/main/`: original GSM8K main train/test files. Test is reserved for evaluation.
- `data/processed/`: reserved for sentence-level datasets; currently unprepared.
- `manifests/`: source provenance, dataset revisions, file checksums, schemas, row counts, and status.
- `configs/`: reserved for experiment configurations.
- `outputs/`, `logs/`: reserved for later experiments.
- `PLAN.md`: the implementation plan created before this preparation step. Its original suggested workspace path is superseded by this directory.

The copied source files are byte-identical to `/root/LaDiR` at commit `5eaa5968393e5e93262bb53b867683704dd1da3a`; see `manifests/source.json`. The original checkout is preserved. `UPSTREAM_README.md` and `manifests/upstream_pyproject.toml` retain upstream documentation and dependency declarations.

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

Next work: build sentence-level math data and explicit parent-question splits; implement encoder-input corruption and standalone reconstruction checks; create experiment configs; validate an isolated training environment and one forward/backward update. The current `diffuguard` environment has older training dependencies than the copied project declares. No environment installation was performed here.

The copied VAE itself needs only `vae/`; `ladir/` and `sft/` retain the code needed for the planned downstream encoder loading and latent export. Code-generation scorers and RL training packages were intentionally left out. DeepSpeed is an optional additional installation if the later memory strategy needs it.
