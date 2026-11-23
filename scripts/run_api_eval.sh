#!/bin/bash
# Closed-model (API) evaluation: key check, freeze verify, then one Slurm job per model and group set.
# Usage: scripts/run_api_eval.sh <model> <smoke|tiny|all> --cap <USD> [--settings a,b,...] [--prefix P] [--repeats N] [--time T] [--resume] [--local] [--key-file F] [--reset-key N]
#   <model> is a short name from configs/models_api.yaml; --settings defaults to all 8 settings; --prefix defaults to the group-set name
#   (run names <prefix>_r<seed>; the cost cap accumulates per prefix). --cap is mandatory: the total cap for this model and prefix,
#   including money already spent. --repeats: runs of active_v2 and random_walk_40 (seeds 0..N-1). --local runs inside the current
#   Slurm allocation instead of submitting a job. API keys come from environment variables (see README); never pass them as arguments.
set -uo pipefail
usage() { sed -n '3,3p' "$0" | sed 's/^# //'; exit 2; }
[ $# -ge 2 ] || usage
MODEL=$1; SET=$2; shift 2
CAP=""; SETTINGS=all; PREFIX=$SET; REPEATS=""; TIME=""; RESUME=0; LOCAL=0; KEY_FILE=""; RESET=()
while [ $# -gt 0 ]; do
  case $1 in
    --cap|--settings|--prefix|--repeats|--time|--key-file|--reset-key) [ $# -ge 2 ] || { echo "$1 is missing a value"; usage; } ;;&
    --cap) CAP=$2; shift 2 ;;
    --settings) SETTINGS=$2; shift 2 ;;
    --prefix) PREFIX=$2; shift 2 ;;
    --repeats) REPEATS=$2; shift 2 ;;
    --time) TIME=$2; shift 2 ;;
    --key-file) KEY_FILE=$2; shift 2 ;;
    --reset-key) RESET+=(--reset-key "$2"); shift 2 ;;
    --resume) RESUME=1; shift ;;
    --local) LOCAL=1; shift ;;
    *) echo "unknown argument $1"; usage ;;
  esac
done
case $SET in smoke|tiny|all) ;; *) echo "the group set must be smoke, tiny or all"; exit 2 ;; esac
if ! [[ "$CAP" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "a cost cap must be given with --cap <USD>; refusing to run without it"; exit 2; fi
if [ -n "$REPEATS" ] && ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then echo "--repeats must be a positive integer"; exit 2; fi
if [ "$LOCAL" = 1 ] && { [ -z "${SLURM_JOB_ID:-}" ] || [[ "$(hostname)" == *login* ]]; }; then echo "--local can only be used on a compute node inside a Slurm allocation, not on a login node"; exit 2; fi

REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${ACTSPACE_PYTHON:-python}
cd "$REPO"
export PYTHONPATH=src
EV=("$PY" -m actspace_eval.providers.api_eval)
ROOT=$("$PY" -c "import sys; sys.path.insert(0, 'src'); from actspace_eval.providers import registry as R; print(R.results_root(R.load_registry()))")
JOBS=$ROOT/_jobs
mkdir -p "$JOBS"
ARGS=(--model "$MODEL" --set "$SET" --prefix "$PREFIX" --settings "$SETTINGS" --cap "$CAP")
[ -n "$REPEATS" ] && ARGS+=(--repeats "$REPEATS")
[ "$RESUME" = 1 ] && ARGS+=(--resume)
KEYS=()
[ -n "$KEY_FILE" ] && KEYS=(--key-file "$KEY_FILE")
ARGS+=("${KEYS[@]}")

echo "== keys and switches"
"${EV[@]}" check --model "$MODEL" "${KEYS[@]}" || exit 2
[ ${#RESET[@]} -gt 0 ] && { "${EV[@]}" keys "${RESET[@]}" || exit 2; }
echo "== freeze verify"
"${EV[@]}" verify || exit 2

if [ "$LOCAL" = 1 ]; then
  exec "$PY" -u -m actspace_eval.providers.api_eval job "${ARGS[@]}"
fi
SB=(--parsable -J "api_${MODEL}_${PREFIX}" -o "$JOBS/%x_%j.out")
[ -n "$TIME" ] && SB+=(-t "$TIME")
if [ "$SET" = smoke ]; then
  echo "== submitting 1 job and waiting for it to finish (job log in $JOBS/, progress in $ROOT/$MODEL/_progress/$PREFIX/)"
  sbatch --wait "${SB[@]}" scripts/run_api_job.sbatch "$REPO" "${ARGS[@]}"
  RC=$?
  echo "== truncation rate"
  "${EV[@]}" truncation --model "$MODEL" --prefix "$PREFIX" --settings "$SETTINGS"
  [ $RC = 0 ] || echo "job exit code $RC (1 probe failed; 2 existing artifacts; 3 cost cap; 4 fatal error; 5 server error; 6 all keys exhausted); see $JOBS/api_${MODEL}_${PREFIX}_*.out"
  exit $RC
fi
J=$(sbatch "${SB[@]}" scripts/run_api_job.sbatch "$REPO" "${ARGS[@]}") || { echo "sbatch failed"; exit 1; }
echo "submitted job $J"
echo "  job log: $JOBS/api_${MODEL}_${PREFIX}_$J.out"
echo "  progress: tail -f $ROOT/$MODEL/_progress/$PREFIX/*.log"
echo "  score table: $ROOT/$MODEL/_reports/$PREFIX.md (generated once everything has finished)"
