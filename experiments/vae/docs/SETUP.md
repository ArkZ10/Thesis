# Environment

Use the existing isolated environment:

```bash
cd /root/Thesis/VAE_Training
source .venv/bin/activate
python -m pip check
```

`pyproject.toml` declares the current VAE package and its direct dependencies. `requirements.lock.txt` records the installed environment; cleanup does not upgrade or uninstall packages. DeepSpeed is pinned to 0.19.9 because the bounded-storage checkpoint implementation depends on its internals.

The cached base checkpoint is `models/Llama-3.1-8B`, revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`. Tokenizer/model checksums are recorded in `manifests/llama_3.1_8b.json`. Decoder weights are frozen. Active training loads local files only.

The launcher supplies the locally extracted libaio paths through `vae/runtime.py`. They live under `.venv/ladir-offload/libaio/prefix/usr`; preserve this dependency when recreating the environment.

Both RTX 3090 GPUs are used: GPU 0 for the trainable encoder and GPU 1 for the frozen decoder. GPU 1's observed active PCIe link width is x1; GPU 0 is x16. FP32 Adam state is offloaded to NVMe. This is a single-rank process spanning two GPUs, not two data-parallel workers.

CPU verification:

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```
