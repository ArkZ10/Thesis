#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DS_BUILD_OPS=0
"$ROOT/.env/bin/python" -m pip install --constraint "$ROOT/setup/constraints.txt" -e "$ROOT/setup/package"
