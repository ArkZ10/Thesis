#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$ROOT/.env/bin/python" ]]; then
    CONDA_OFFLINE=true /opt/miniforge3/bin/conda create --offline --prefix "$ROOT/.env" --clone /opt/miniforge3/envs/diffuguard --copy -y
fi
bash "$ROOT/scripts/install_environment.sh"
# Unrelated inherited image/captioning packages conflict with the pilot's NumPy.
"$ROOT/.env/bin/python" -m pip uninstall -y capture-metric FactualSceneGraph spacy thinc decord
"$ROOT/.env/bin/python" -m pip check
