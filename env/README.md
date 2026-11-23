# Environments

Exact package versions of the environments that produced the paper results (exported 2026-11-15).

| Environment | Used for | Python | Key packages |
|---|---|---|---|
| `actspace` | evaluation loop, scoring, reports, job queue, tests, API models, dataset construction | 3.11.18 | numpy 2.4.6, opencv-python-headless 5.0.0.93, PyYAML 6.0.3, requests 2.34.2, open3d 0.20.0, pytest 9.2.0 |
| `actspace-vllm` | vLLM servers (OpenAI-compatible) for open models, model download | 3.11.16 | vllm 0.19.1, torch 2.10.0 (CUDA 12.8), transformers 5.5.4, flashinfer-python 0.6.6, xformers 0.0.34, triton 3.6.0, huggingface_hub 1.6.0 |
| `cambrian-s` (optional) | server for Cambrian-S-7B with its official code | 3.10.22 | torch 2.3.0+cu121, torchvision 0.18.0+cu121, transformers 4.37.0, tokenizers 0.15.0, accelerate 0.23.0; official code github.com/cambrian-mllm/cambrian-s at commit 058b1b4 (installed with `pip install -e`) |

- Hardware used for the paper: NVIDIA A100-SXM4-40GB, driver 570.195.03. torch 2.10.0+cu128 needs driver >= 570.
- Files: `<env>.requirements.txt` (what `install_envs.sh` installs), `<env>.pip-freeze.txt` (`pip list --format=freeze` as exported),
  `<env>.environment.yml` (`conda env export --no-builds`, for reference only).
- Install: `bash env/install_envs.sh [--core] [--cambrian]`, see the header of the script.
