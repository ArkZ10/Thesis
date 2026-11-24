# MetaState pilot: part 1

No training, model inference, or training-data preparation has been performed.
The original MetaState checkout, diffuguard environment, and earlier experiments remain unchanged.

## Activate

```bash
cd /root/Thesis/Sycophancy-benchmark-test/MetaState-Latent-Reasoning/My-Metastate
source scripts/activate.sh
python scripts/verify_setup.py
```

Verification checks package versions/imports, source hashes, preservation of the original environment's package list, offline tokenization, and CUDA visibility. It does not load model weights or establish training memory fit.

## Frozen inputs

- Source commit: `307549ef9b62a8a371c5c8e59cfeacfe537630c1`.
- Pristine source snapshot: `vendor/MetaState/`.
- Backbone: `GSAI-ML/LLaDA-8B-Instruct`.
- Revision: `08b83a6feb34df1a6011b80c3c00c7563e963b07`.
- Use the exact local snapshot paths in `setup/backbone.json` for both model and backbone arguments. These paths avoid downloading a moving branch. The JSON also contains metadata keys; do not pass the entire file directly as CLI arguments.
- Shard existence and configuration hashes were checked; large weight files were not rehashed or loaded.

## Environment

`.env/` is an independent Conda copy of the original diffuguard environment with MetaState dependencies installed. The original environment is not changed.

The upstream requirements conflict: `lm_eval==0.4.9.1` requires `datasets<4.0`, while MetaState requires `datasets==4.2.0`. Our packaging overlay at `setup/package/` changes only the lm_eval requirement to `0.4.9.2`, which allows datasets 4.2.0. Python source remains unchanged. See `setup/dependency-adjustment.json` for all adjustments.

Exact installed packages: `setup/environment.freeze.txt`. Conda packages: `setup/conda-explicit.txt`. Pip dependency validation: `setup/pip-check.txt`.

To repeat environment creation from the preserved original environment and install the recorded project requirements:

```bash
bash scripts/create_environment.sh
```

This installer uses `setup/constraints.txt`; unpinned transitive dependencies may resolve differently on a later date. Use the full environment freeze for exact package-version restoration, retaining the local editable package entry. Conda explicit URLs capture Conda packages only, not the pip additions.

## Layout

- `scripts/`: setup, activation, verification, environment capture.
- `setup/`: manifest, pinned model reference, dependency overlay, package records, verification report.
- `vendor/MetaState/`: preserved upstream source snapshot.
- `logs/`: setup logs.
- `.env/`: isolated Python environment.

Data preparation is part 2 and has not started. Compatibility with an actual LLaDA training forward/backward pass remains to be checked in the training pilot.
