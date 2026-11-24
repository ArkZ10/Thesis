#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/.env/bin/python" -m pip check > "$ROOT/setup/pip-check.txt"
"$ROOT/.env/bin/python" -m pip freeze --all > "$ROOT/setup/environment.freeze.txt"
/opt/miniforge3/bin/conda list --prefix "$ROOT/.env" --explicit > "$ROOT/setup/conda-explicit.txt"
