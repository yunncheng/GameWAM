#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
MODEL_ROOT="${MODEL_ROOT:-/path/to/vizdoom_appo_models}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec conda run --no-capture-output -n vizdoom_appo \
  python "${SCRIPT_DIR}/appo/prepare_appo_models.py" \
  --output-root "${MODEL_ROOT}" \
  "$@"
