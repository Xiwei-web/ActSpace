#!/bin/bash
# One closed-model (API) evaluation job; submitted by scripts/run_api_eval.sh. No GPU needed. Adjust the #SBATCH lines to your cluster.
# Usage: sbatch scripts/run_api_job.sbatch <repo dir> <arguments of `python -m actspace_eval.providers.api_eval job`>
# API keys are read from environment variables (or an optional key file, see src/actspace_eval/providers/keys.py);
# this script never reads, writes or prints them.
#SBATCH -c 8
#SBATCH --mem=32G
set -uo pipefail
REPO=$1; shift
cd "$REPO"
export PYTHONPATH=src
exec "${ACTSPACE_PYTHON:-python}" -u -m actspace_eval.providers.api_eval job "$@"
