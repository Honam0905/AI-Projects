#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$PROJECT_ROOT/.venv/bin/activate"
export AGENT_SWARMS_SANDBOX_DATA_DIR="${AGENT_SWARMS_SANDBOX_DATA_DIR:-$PROJECT_ROOT/../.agent_swarms_data}"
export WATCHFILES_FORCE_POLLING="${WATCHFILES_FORCE_POLLING:-true}"

uvicorn agent_swarms.api.app:app \
  --reload \
  --reload-dir agent_swarms \
  --reload-dir scripts \
  --reload-dir tests
