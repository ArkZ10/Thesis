# Verified VAE virtual environment

Location: `/root/Thesis/VAE_Training/.venv`.

Activate it:

```bash
cd /root/Thesis/VAE_Training
source .venv/bin/activate
```

Python is 3.11.0. The venv uses that interpreter's standard library but has its own site-packages, with system and user site-packages disabled. The existing `diffuguard` environment's installed packages were preserved.

Verified package versions:

| Package | Version |
|---|---|
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.5.4 |
| Accelerate | 1.15.0 |
| Hugging Face Hub | 1.33.0 |
| Datasets | 5.2.0 |
| PyArrow | 26.0.1 |
| OmegaConf | 2.4.0 |
| Safetensors | 0.8.0 |

`requirements-venv.lock.txt` records the complete installed package set. The project is installed in editable mode, so local VAE code changes are available to this environment.

## Verification performed

- `pip check`: no broken requirements.
- All 15 data preparation unit/integration tests passed in this venv.
- VAE model/data/trainer and latent-export imports passed.
- `python -m vae.train --help` passed.
- Both RTX 3090 GPUs were visible and passed BF16 matrix multiplication and backward checks.
- A tiny randomly initialized Qwen3 VAE passed a forward/backward check with hybrid masks, padding, six latent slots, gradient checkpointing, posterior sampling, and FP32 reconstruction CE.
- Gradients reached the encoder, queries, posterior heads and decoder interface; the decoder backbone remained frozen.

The initial synthetic environment check used no pretrained model weights, dataset examples, or optimizer updates. Subsequently, the authorized pretrained small-model diagnostic completed; see `SMALL_MODEL_TEST.md`. Full-size model memory usage and multi-process training are not yet validated.

Detailed results: `manifests/environment.json`. Installation metadata: `manifests/torch-install.json` and `manifests/environment-install.json`. Logs are under `logs/environment-*.log`.

## Optional components

DeepSpeed, vLLM, the RL stack, and W&B logging were not installed. They are not required for the initial standard Accelerate VAE path. The upstream `scripts/train_vae.sh` defaults to DeepSpeed and eight GPUs; its defaults must be overridden or replaced by the planned experiment launcher before use. A CPU-offload/DeepSpeed configuration can be installed and checked when the larger-model memory strategy is chosen.

## Recreating the environment

From this workspace, with a Python 3.11 venv activated, run:

```bash
python -m pip install -r requirements-venv.lock.txt
```

The lock file supplies the official CUDA 12.4 wheel index for the pinned PyTorch build. See [PyTorch's previous-version installation instructions](https://pytorch.org/get-started/previous-versions/).

To repeat the existing checks:

```bash
python -m pip check
python -m unittest discover -s tests -v
python scripts/check_environment.py
```

GPU checks require an execution context with access to the GPU devices. The later small-model checkpoint, subset tokenization, config and training diagnostic are documented separately in `SMALL_MODEL_TEST.md`; full-size memory and training remain unvalidated.
