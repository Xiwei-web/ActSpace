#!/bin/bash
# OpenAI-compatible server for Cambrian-S-7B (paths come from environment variables).
# Usage: serve_cambrian.sh PORT [other arguments of cambrian_server.py]; listens on 127.0.0.1 only. open_queue (run / run-local) calls it via open_models.cambrian_serve.
#   CAMBRIAN_PYTHON  python of the Cambrian-S environment (default open_models.cambrian_python, set by run-local)
#   CAMBRIAN_REPO    directory of the official code github.com/cambrian-mllm/cambrian-s (commit 058b1b4)
#   CAMBRIAN_MODEL   weights directory (nyu-visionx/Cambrian-S-7B)
#   HF_HOME          cache holding google/siglip2-so400m-patch14-384 (offline)
set -euo pipefail
PORT="${1:?usage: serve_cambrian.sh PORT}"; shift || true
: "${CAMBRIAN_PYTHON:?}" "${CAMBRIAN_REPO:?}" "${CAMBRIAN_MODEL:?}" "${HF_HOME:?}"
export CAMBRIAN_REPO HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export PATH="$(dirname "$CAMBRIAN_PYTHON"):$PATH"
exec "$CAMBRIAN_PYTHON" "$(dirname "$0")/cambrian_server.py" --model-path "$CAMBRIAN_MODEL" --host 127.0.0.1 --port "$PORT" "$@"
