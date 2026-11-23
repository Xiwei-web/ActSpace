#!/bin/bash
# Create the conda environments used for the paper runs, with the exact package versions that ran (exported 2026-11-15).
#   bash env/install_envs.sh                 # actspace (evaluation loop, scoring, reports) + actspace-vllm (open-model servers)
#   bash env/install_envs.sh --core          # only actspace (enough for tests, API models and reports)
#   bash env/install_envs.sh --cambrian      # additionally Cambrian-S (separate environment + official code), only to rerun Cambrian-S-7B
#   ENV_ROOT=/data/envs bash env/install_envs.sh   # put the environments elsewhere (default: <repo>/envs)
# Needs conda (Miniconda is enough) and internet access. actspace-vllm needs an NVIDIA driver >= 570 (torch 2.10.0+cu128, CUDA 12.8).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-$REPO/envs}"
mkdir -p "$ENV_ROOT"
mk() {   # mk <name> <python version> <requirements> [extra pip arguments...]
  local name=$1 py=$2 req=$3; shift 3
  [ -x "$ENV_ROOT/$name/bin/python" ] || conda create -y -q -p "$ENV_ROOT/$name" "python=$py" pip
  "$ENV_ROOT/$name/bin/python" -m pip install --no-input -r "$req" "$@"
  "$ENV_ROOT/$name/bin/python" -m pip check || echo "(pip check reported conflicts above; the same declared conflicts exist in the environment used for the paper and do not affect the runs)"
}
CORE=0; CAMBRIAN=0
for a in "$@"; do
  case $a in --core) CORE=1 ;; --cambrian) CAMBRIAN=1 ;; *) echo "unknown argument $a"; exit 2 ;; esac
done
mk actspace 3.11.18 "$REPO/env/actspace.requirements.txt"
"$ENV_ROOT/actspace/bin/python" -c "import cv2, numpy, yaml; print('actspace OK: opencv', cv2.__version__, 'numpy', numpy.__version__)"
if [ "$CORE" = 0 ]; then
  mk actspace-vllm 3.11.16 "$REPO/env/actspace-vllm.requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu128
  "$ENV_ROOT/actspace-vllm/bin/python" -c "import vllm, torch, transformers; print('actspace-vllm OK: vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
fi
if [ "$CAMBRIAN" = 1 ]; then
  CODE="$ENV_ROOT/cambrian-s-code"
  [ -d "$CODE/.git" ] || git clone https://github.com/cambrian-mllm/cambrian-s.git "$CODE"
  git -C "$CODE" checkout -q 058b1b4
  mk cambrian-s 3.10.22 "$REPO/env/cambrian-s.requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu121
  "$ENV_ROOT/cambrian-s/bin/python" -m pip install --no-input --no-deps -e "$CODE"
fi
echo "Environments are in $ENV_ROOT."
[ "$CORE" = 0 ] && { echo "Set in configs/paths.yaml:"; echo "  vllm_env: $ENV_ROOT/actspace-vllm"; }
[ "$CAMBRIAN" = 1 ] && { echo "  cambrian_python: $ENV_ROOT/cambrian-s/bin/python"; echo "  cambrian_repo: $ENV_ROOT/cambrian-s-code"; }
exit 0
