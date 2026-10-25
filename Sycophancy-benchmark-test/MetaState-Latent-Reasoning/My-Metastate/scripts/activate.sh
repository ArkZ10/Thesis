# Source this file: source scripts/activate.sh
_METASTATE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate "$_METASTATE_ROOT/.env"
unset _METASTATE_ROOT
