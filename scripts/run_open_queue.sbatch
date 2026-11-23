#!/bin/bash
# Example Slurm job for the open-model evaluation queue (actspace_eval.open_queue). Adjust the #SBATCH lines to your cluster.
# Usage: python -m actspace_eval.open_queue init ...   then, from the repository root:
#        mkdir -p results/_jobs && sbatch scripts/run_open_queue.sbatch
# The queue resubmits this script as a continuation job (afterany dependency) when the time limit approaches; extra sbatch
# arguments given to `open_queue init --sbatch-args` are reused. Paths (models, results, vLLM env) come from configs/paths.yaml.
# Python: set ACTSPACE_PYTHON to the interpreter of the `actspace` environment (default: python on PATH).
#SBATCH --gres=gpu:4
#SBATCH -c 64
#SBATCH --mem=230G
#SBATCH -t 48:00:00
#SBATCH -J open_queue
#SBATCH -o results/_jobs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT=/tmp/${USER}_vllm_cache_$SLURM_JOB_ID TORCHINDUCTOR_CACHE_DIR=/tmp/${USER}_inductor_$SLURM_JOB_ID
exec "${ACTSPACE_PYTHON:-python}" -m actspace_eval.open_queue run
