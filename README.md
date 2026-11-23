# ActSpace-Bench

ActSpace-Bench evaluates **active spatial reasoning** of multimodal models in real indoor scenes. A model starts at one view of a
scanned room, receives 3–5 spatial questions, and may move through a graph of real camera views (translate or turn, 40 steps) before
it answers. The benchmark measures whether the model's own exploration helps compared with random and scripted exploration under
the same budget, and whether it abstains when the evidence was removed (unanswerable questions).

This repository contains the dataset construction pipeline (`actspace_build`), the evaluation environment, model adapters and
reports (`actspace_eval`), configurations, scripts and tests. **It contains no images, scans or annotations**: frames are rebuilt from
the official ARKitScenes and ScanNet++ downloads, and the ActSpace annotations are released separately (coming soon).

| Question type | Example | Answer | Scoring |
|---|---|---|---|
| T1 absolute distance | What is the distance between the center of the chair and the center of the table, in meters? | number (m) | MRA over 10 thresholds |
| T2 relative distance | Which of the following objects is closest to the lamp? | option letter | accuracy |
| T3 perspective taking | Imagine you are standing at the table and facing the chair. Where is the lamp relative to you? | 4 quadrants | accuracy |
| T4 start-relative location | Relative to your starting position and viewing direction, where is the sink? | 4 quadrants | accuracy |
| T5 multi-hop | Standing at the table facing the chair, among the objects on your left, which one is closest to you? | option letter | accuracy |
| Unanswerable (T1/T4) | evidence views masked out of the view graph | `cannot_determine` | 1 only for `cannot_determine` |

| Setting | Photos given to the model | Role |
|---|---|---|
| Blind | none (questions only, must guess) | language prior |
| Passive-8 / 20 / 40 | START + k random views of the group | passive coverage |
| random_walk_40 | 40 random moves, photos + actions + poses | baseline for Δ_RW |
| oracle_min | START + the scripted oracle's target views | evidence upper reference |
| oracle_route | the scripted oracle's shortest evidence route, photos + actions + poses | baseline for Δ_O |
| **Active v2** (main) | interactive exploration, 40 steps, odometry + cumulative pose, START + last 8 photos | main setting |

Data: `merged_v1.0`, ARKitScenes 41 scenes / 103 groups / 425 questions and ScanNet++ 31 scenes / 131 groups / 569 questions.
The full protocol is in [docs/evaluation_protocol.md](docs/evaluation_protocol.md).

## Contents

- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick check without data](#quick-check-without-data)
- [Data](#data)
- [Open models](#open-models)
- [API models](#api-models)
- [Ablations and probes](#ablations-and-probes)
- [Reproducing the paper tables](#reproducing-the-paper-tables)
- [Protocol summary](#protocol-summary)
- [Rebuilding the dataset (optional)](#rebuilding-the-dataset-optional)
- [FAQ](#faq)
- [Citation, license and data terms](#citation-license-and-data-terms)

## Repository structure

```
configs/
  eval.yaml              evaluation protocol and open-model settings (read by the code)
  models_api.yaml        API model registry (model IDs, endpoints, reasoning request, prices)
  build.yaml, merge.yaml dataset construction parameters
  paths.example.yaml     template for configs/paths.yaml (all local paths; git-ignored)
  chat_templates/        format-adapted chat templates for three open models
docs/
  evaluation_protocol.md the evaluation protocol, with all prompt templates verbatim
  data_construction.md   how the dataset was built (Steps 0-10, human checkpoints)
  models/                open models: repository, revision, serving settings, per-file SHA-256 manifests
env/                     exact package versions of the three conda environments + install_envs.sh
scripts/
  rebuild_frames.py      rebuild the evaluation frames from the official raw data
  verify_frames.py       check rebuilt frames against pixel hashes
  download_models.py     download open-model weights at pinned revisions and verify SHA-256
  run_api_eval.sh        API evaluation via Slurm (optional; the Python entry points run anywhere)
  run_open_queue.sbatch, run_api_job.sbatch   example Slurm jobs (optional)
  cambrian/              server wrapper for Cambrian-S (separate environment)
  check_chat_templates.py, vllm_image_tokens.py, server_info.py, compare_runs.py   diagnostics
src/actspace_build/      dataset construction (Steps 0-10)
src/actspace_eval/       environment, prompts, agents, model clients, scoring, job queue, reports
  providers/             API model adapters (OpenAI, Anthropic), key handling, cost control
tests/                   unit tests and end-to-end mock smoke tests on a synthetic dataset
vllm_plugins/            vLLM plugin for LLaVA-OneVision (base resolution) and MiniCPM-V (tokenizer attributes)
vocab/                   category vocabularies and label mappings
```

## Installation

Requirements: Linux, conda (Miniconda is enough), internet access for the installation. Open models need NVIDIA GPUs with driver
>= 570 (the paper used A100-SXM4-40GB, driver 570.195.03). API models and all tests run on CPU.

```bash
git clone <repo-url> ActSpace-Bench
cd ActSpace-Bench
bash env/install_envs.sh --core        # only the actspace environment (tests, API models, reports)
bash env/install_envs.sh               # actspace + actspace-vllm (open models)
bash env/install_envs.sh --cambrian    # additionally Cambrian-S (only to rerun Cambrian-S-7B)
```

| Environment | Python | Key packages |
|---|---|---|
| `envs/actspace` | 3.11.18 | numpy 2.4.6, opencv-python-headless 5.0.0.93, PyYAML 6.0.3, requests 2.34.2, pytest 9.2.0 |
| `envs/actspace-vllm` | 3.11.16 | vllm 0.19.1, torch 2.10.0 + CUDA 12.8, transformers 5.5.4, flashinfer-python 0.6.6, huggingface_hub 1.6.0 |
| `envs/cambrian-s` (optional) | 3.10.22 | torch 2.3.0+cu121, transformers 4.37.0; official code at commit 058b1b4 |

The environments are created under `envs/` (change with `ENV_ROOT=...`). All commands below are run from the repository root
with the `actspace` environment active and `src` on the Python path:

```bash
conda activate ./envs/actspace
export PYTHONPATH=src
```

Local paths (data, frames, models, results, vLLM environment) are set in one file:

```bash
cp configs/paths.example.yaml configs/paths.yaml     # then edit; every key is optional and documented in the file
```

Without `configs/paths.yaml`, everything lives inside the repository (`work/`, `models/`, `results/`, ...). The environment
variable `ACTSPACE_PATHS` can point to another paths file.

## Quick check without data

The tests use a small synthetic dataset (two scenes, generated on the fly; no images from any dataset) and the mock model, so they
need neither data nor GPUs nor API keys:

```bash
python -m pytest -q tests
```

Tests that need the annotation package (or the frozen build directory) are skipped automatically when it is not available. The same synthetic dataset can be used
to try the command-line tools:

```bash
python tests/synthetic.py synthetic_data
export ACTSPACE_PATHS=$PWD/synthetic_data/paths.yaml
python -m actspace_eval.llm_run --model mock --mode active_v2 --set all --name demo_active_v2
python -m actspace_eval.llm_run --model mock --mode random_walk --set all --run-seed 0 --name demo_rw40_r0
python -m actspace_eval.report_eval --out demo_report.md --set active_v2=demo_active_v2 --set random_walk_40=demo_rw40_r0
python -m actspace_eval.providers.api_eval job --model mock-openai --set all --prefix demo --cap 1 --settings blind,active_v2
unset ACTSPACE_PATHS
```

Runs are written to `synthetic_data/results/` (single runs in `eval_runs/`, API runs in `api/`).

## Data

The official benchmark is the **released frozen annotation package** (`merged_v1.0`). It contains no images: the evaluation frames
are rebuilt from your own copies of ARKitScenes and ScanNet++.

### 1. Official raw data

Download the scenes from the official sources and accept their terms; we do not redistribute any of their data.

| Dataset | Subset used | What the evaluation needs | Official source and terms |
|---|---|---|---|
| ARKitScenes | 3DOD, Validation split (41 videos in `merged_v1.0`) | raw `vga_wide` frames | https://github.com/apple/ARKitScenes (see its license and terms) |
| ScanNet++ (iPhone data) | scenes of `nvs_sem_val` (31 scenes in `merged_v1.0`) | `iphone/rgb.mkv` | https://kaldir.vc.in.tum.de/scannetpp/ (requires an application and accepting the ScanNet++ Terms of Use) |

The exact scene and video IDs are listed in the annotation package (`metadata/scene_list.txt`; ARKitScenes scene `arkit_<video_id>`,
ScanNet++ scene `scannetpp_<scene_id>`). ARKitScenes raw frames are downloaded with the official script, one video at a time, from a
clone of the official repository:

```bash
git clone https://github.com/apple/ARKitScenes.git third_party/ARKitScenes
cd third_party/ARKitScenes
python download_data.py raw --split Validation --video_id <video_id> --download_dir data --raw_dataset_assets vga_wide vga_wide_intrinsics
cd ../..
```

ScanNet++ is downloaded with the official download tool after your application is approved (it gives you a personal token; keep it
out of this repository).

### 2. ActSpace annotations (coming soon)

The annotation package will be released under terms compatible with the licenses of the source datasets (to be specified). Its layout:

```
<annotations>/
  data/questions/test.parquet     one row per question (Hugging Face `questions` configuration)
  data/groups/test.jsonl          one row per question group: the full records read by the evaluation (also as test.parquet)
  scenes/<scene_id>/              scene_meta.json, objects.json, frames.json (keyframes with poses, intrinsics, source frame),
                                  view_graph.json (views and action edges), visibility.npz (per object and view)
  metadata/                       freeze_manifest.json, build_info.json, scene_list.txt, tiny_groups.txt, eval_subsets.json,
                                  human_verification.csv, canary.txt, FRAMES_SHA256.txt (pixel hashes of all evaluation frames)
  tools/rebuild_frames.py         same as scripts/rebuild_frames.py
  SHA256SUMS                      hashes of all files above
```

Point `annotations:` in `configs/paths.yaml` to the unpacked directory. At start-up the evaluation checks every file against
`SHA256SUMS` and the tiny set and budget-infeasible groups against `configs/eval.yaml`, and refuses to run on any mismatch.
In the package, unanswerable questions have `answer = "cannot_determine"` with the geometric answer in `geometric_answer`, and T1
answers are the unrounded distances used for scoring; the loader (`actspace_eval.data.group_from_annotations`) maps the records back
to the frozen format. `tests/eval/test_annotations.py` checks, field by field, that the package and the frozen `merged_v1.0` give
the same groups, questions, answers, scene files, environment data and episodes.

### 3. Rebuild and verify the frames

```bash
python scripts/rebuild_frames.py --release <annotations> --out frames \
    --arkit-root third_party/ARKitScenes/data --scannetpp-root <ScanNet++ data root>
python scripts/verify_frames.py --frames frames --sums <annotations>/metadata/FRAMES_SHA256.txt
```

`rebuild_frames.py` writes `frames/<scene_id>/<frame_id>.png` (ARKitScenes) or `.jpg` (ScanNet++), upright and at full resolution;
the evaluation downscales them to 640 × 480. It needs only numpy, opencv and ffmpeg and decodes ScanNet++ frames from `rgb.mkv` with
the same ffmpeg command as the original build. `verify_frames.py` compares decoded pixels, not file bytes, so it is independent of the
PNG/JPEG encoder version. Then set in `configs/paths.yaml`:

```yaml
paths:
  annotations: /path/to/actspace_annotations/merged_v1.0
  frames: /path/to/ActSpace-Bench/frames
```

## Open models

17 open models were evaluated, each with a complete sealed run of all 8 settings (see [docs/models/models.yaml](docs/models/models.yaml)
for repositories, pinned revisions, the hardware of the paper runs and serving settings). One setting is N/A by protocol:
LLaVA-OneVision-7B's oracle_route requests exceed its 32,768-token context. All use the same protocol; servers are vLLM's OpenAI-compatible API (Cambrian-S uses its official code behind a
small wrapper, `scripts/cambrian/`).

### 1. Download and verify weights

```bash
envs/actspace-vllm/bin/python scripts/download_models.py --models Qwen3.5-9B
envs/actspace-vllm/bin/python scripts/download_models.py --models Qwen3.5-9B --verify-only
```

Weights go to `<models>/<local_dir>` (`models:` in `configs/paths.yaml`, default `models/`) and are checked file by file against
`docs/models/manifests/<model>.sha256`. Without `--models`, all models are downloaded. Gated repositories (RoboBrain2.0-7B) require
accepting the license on the Hugging Face website and a token in the environment (`read -s HF_TOKEN && export HF_TOKEN`, so that the
token appears neither on screen nor in the shell history); it is read only from the environment. Set `vllm_env:` (and `hf_home:`) in `configs/paths.yaml`.

### 2. Smoke test (3 groups × 8 settings)

```bash
python -m actspace_eval.open_queue run-local --gpus 0 --models Qwen3.5-9B --smoke-only
```

The smoke gate checks for program problems only (server errors, incomplete records, image token counts that differ between
landscape and portrait photos, interleaving of images and text).

### 3. Full run

On one machine (uses only the listed GPUs; models that need tensor parallelism get as many GPUs as `vllm.tp` in `configs/eval.yaml`):

```bash
python -m actspace_eval.open_queue run-local --gpus 0,1,2,3 --models Qwen3.5-9B
python -m actspace_eval.open_queue run-local --gpus 0,1,2,3 --models Qwen3.5-9B --settings active_v2 random_walk_40 --seeds 1,2
python -m actspace_eval.open_queue status
```

The first command runs all 8 settings once (run names `full_r0`, random walk `full_s0`); this is what the paper reports for every
open model (one run per setting, see `paper_runs` in `docs/models/models.yaml`). The second command adds the 2nd and 3rd runs of
Active v2 and random_walk_40 that the protocol specifies. Results are in `<results>/<model>/<setting>/<run>/`; a score table is
written per model.

With Slurm (example script; adjust the `#SBATCH` lines to your cluster):

```bash
python -m actspace_eval.open_queue init --prefix full --seeds 0 --report-dir reports/open_models --models Qwen3.5-9B
mkdir -p results/_jobs
sbatch scripts/run_open_queue.sbatch
```

The job runs the queue, stops cleanly before its time limit and resubmits itself as a continuation job.

### 4. Resume

Interrupted runs resume where they stopped: rerun the same `run-local` command (or let the continuation job start). Completed runs
are recognised by their `SHA256SUMS` and skipped; incomplete episodes are discarded and rerun. To retry a failed or paused task:

```bash
python -m actspace_eval.open_queue reset Qwen3.5-9B active_v2
```

A single setting can also be run against a vLLM server you started yourself:

```bash
python -m actspace_eval.llm_run --model Qwen3.5-9B --mode active_v2 --set all --run-seed 0 --base-url http://127.0.0.1:8000 --out-root results --name full_r0 --workers 8 --seal
```

## API models

API models are registered in `configs/models_api.yaml` (short name, model ID, endpoint, reasoning request, JSON mode, image detail,
prices). Real calls are refused unless `prices_confirmed` and `reasoning.approved` are true for the model.

**Keys come only from environment variables** and are never written to logs, results or command lines:

```bash
read -s OPENAI_API_KEY && export OPENAI_API_KEY          # OpenAI models (variable name = key_env in the registry)
read -s ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY    # Anthropic models
read -s ANTHROPIC_API_KEY_1 && export ANTHROPIC_API_KEY_1   # optional: several Anthropic keys, ANTHROPIC_API_KEY_1, _2, ... (rotated)
```

`read -s` reads the key without echoing it, so it does not end up in the shell history.

Every job needs a **cost cap** in USD (`--cap`): the total for this model and run prefix, including money already spent; the job stops
when the cap would be exceeded and can be resumed with a higher cap. Check the key and the switches, estimate the cost, then run the
smoke set, the tiny set (32 groups) and the full set:

```bash
python -m actspace_eval.providers.api_eval check --model sol
python -m actspace_eval.providers.api_eval estimate --model sol --set all --prefix all --repeats 3
python -m actspace_eval.providers.api_eval job --model sol --set smoke --prefix smoke --cap 5
python -m actspace_eval.providers.api_eval job --model sol --set tiny --prefix tiny --cap 40
python -m actspace_eval.providers.api_eval job --model sol --set all --prefix all --cap 400 --repeats 3
python -m actspace_eval.providers.api_eval truncation --model sol --prefix all
```

The paper's main table contains seven API models: sol, terra, luna, opus-5.5, sonnet-5.5 and haiku-5.5 on all 227 groups, and astra,
**evaluated on the 32-group tiny subset only** (Blind, random_walk_40 and Active v2; marked with † in the table, not compared with
the other rows). `paper_runs` in `configs/models_api.yaml` lists the runs behind each row.

A job first probes the API (images, JSON, usage reporting), then runs all settings in parallel within the model's concurrency and
writes a score table to `<results>/api/<model>/_reports/<prefix>.md`. After an interruption, rerun the same command with `--resume`.
With Slurm, `scripts/run_api_eval.sh sol all --cap 400 --repeats 3` does the checks and submits the job.
New providers are added in `src/actspace_eval/providers/__init__.py` (adapter) and the registry.

## Ablations and probes

These experiments are reported in Section 5.6 of the paper (ablations and cognitive-map probe). They reuse the sealed main runs of an
API model as sources (read only) and write their own runs with prefix `abl` to `<results>/api/<model>/<setting>/abl_r0/`. One job runs
all requested settings of a model under one cost cap; `--resume` continues an interrupted job.

| Experiment | Setting(s) | What changes | Paper |
|---|---|---|---|
| Memory-matched random walk | `rw40_last8_v2` (`rw40_last8`: earlier wording of revisit lines, kept as a check) | the random_walk_40 route of seed 0 (checked step by step against the source run), but photos only for START and the last 8 distinct viewpoints, as in Active | §5.6 (A); memory-matched exploration gain ΔRW_mem = Active − rw40_last8_v2 |
| START photo only | `start_only` | the Passive prompt with the START photo as the only photo; `cannot_determine` allowed | §5.6 (B) |
| Reasoning effort | `active_v2_medium` | Active v2 with reasoning effort medium instead of high | §5.6 (C) |
| Cognitive-map probe | `cogmap_active_v2`, `cogmap_oracle_route` | no new exploration: the answer-time context of the main run is rebuilt verbatim (photo hashes checked) and the model is asked for the position of every object in the group's questions relative to the start | §5.6 (D) |

The paper ran (A)–(D) for terra and haiku-5.5, and `rw40_last8_v2` for sol, terra, luna, opus-5.5, sonnet-5.5 and haiku-5.5; not for
astra (see `paper_runs` in `configs/models_api.yaml`).

```bash
read -s OPENAI_API_KEY && export OPENAI_API_KEY
python -m actspace_eval.providers.ablations check --model terra                          # no requests: checks routes and photo hashes
python -m actspace_eval.providers.ablations job --model terra --cap 30                   # all six settings
python -m actspace_eval.providers.ablations job --model sol --cap 15 --settings rw40_last8_v2
python -m actspace_eval.report_ablations                                                 # tables (A)–(D) and ΔRW_mem
```

`report_ablations` reads the runs and writes `<results>/api/_reports/ablations_api.md`, `ablations_api.csv` (every value with its
interval), `ablations_api.json` and `cogmap_api.csv` (one row per probed object). On the paper runs it reproduces the paper's
ablation tables value for value. A single setting with another reasoning effort can also be run directly with
`python -m actspace_eval.llm_run --effort medium ...`.

The other analyses of Section 5 read only finished runs and the benchmark data (no model calls); run them after the main runs:

| Paper | Command | Output (default) |
|---|---|---|
| §5.3 behaviour: random-walk evidence coverage, Active behaviour | `python -m actspace_eval.sec53_behavior --out results/_reports/sec53` | per-episode CSV, summaries |
| §5.3 Active vs random walk truncated to the same number of steps | `python -m actspace_eval.sec53_matched_ci --dir results/_reports/sec53` | paired differences with intervals |
| §5.4 where the evidence is at answer time | `python -m actspace_eval.sec54_evidence_state --out results/_reports/sec54` | per-question evidence states |
| §5.5 within-group consistency and abstention | `python -m actspace_eval.sec55_abstention --out results/_reports/sec55` (after sec54) | rates with intervals |
| §5.6 ranking consistency across the two datasets | `python -m actspace_eval.sec56_cross_source --out results/_reports/sec56` | Kendall τ, Spearman ρ |
| §5.7 error types and cases | `python -m actspace_eval.sec57_errors --out results/_reports/sec57` (after sec54) | per-question errors, case material |

## Reproducing the paper tables

```bash
python -m actspace_eval.report_paper --out-dir paper_table
```

reads the sealed runs (open models: `<results>/<model>/<setting>/full_r*|full_s*`; API models: `<results>/api/<model>/<setting>/<prefix>_r*`
with the prefixes listed in `CLOSED` in `src/actspace_eval/report_paper.py`) and writes `paper_main.md` (combined), `paper_main_arkit.md`,
`paper_main_scannetpp.md`, `paper_main.csv`, `paper_main.tex` (LaTeX, booktabs), `paper_appendix.md` and `self_check.md`. Models without
runs are left out. Models evaluated only on the 32-group tiny subset (astra) are listed with † and the footnote "evaluated on the
32-group tiny subset only", are not marked, and are checked separately by the self-check.

- Main tables: 227 groups (the 7 budget-infeasible ScanNet++ groups are excluded and reported in the appendix), 962 questions
  (Blind: the 899 answerable ones).
- Values are percentages with one decimal. Δ_RW = Active − random_walk_40 and Δ_O = oracle_route − Active are the differences of the
  two displayed values (each rounded to 0.1 first), so they can be recomputed from the table; `self_check.md` verifies this for every
  cell. Whether a difference is significant is judged from intervals computed on unrounded data (the per-model score tables; the
  unrounded differences are also in `paper_main.csv`).
- Bold / underline mark the best / second best per column within each model group (API, open), on unrounded values (Δ columns on
  their displayed values); for Δ_O smaller is better.
- `self_check.md` re-scores every question from the trajectories and the frozen data and checks counts, weights and rounding.
  To compare cell by cell with your own numbers, pass `--paper numbers.csv` (columns `scope, model, column, value`).
- Model sampling is not deterministic across hardware and library versions, so reruns of open and API models differ within the
  reported intervals; Blind/Passive/oracle inputs and random walks are identical by construction.

## Protocol summary

| Item | Value |
|---|---|
| Budget | 40 steps; each move, failed move or invalid reply costs 1; answering is free; 3 consecutive invalid replies end the episode |
| Image history (Active) | START photo + the last 8 photos (K = 8 + START); full text history of actions, photo IDs, odometry and poses |
| Images | upright, long side ≤ 640 px (ARKitScenes 640 × 480, ScanNet++ 1920 × 1440 → 640 × 480), JPEG |
| Output limit | `max_tokens = 32768` (including thinking); reasoning effort high / thinking on |
| Truncation | a reply that hits the limit has no answer: an invalid move, or 0 for the unanswered questions |
| Timeout | 600 s per call for API models and Cambrian-S; vLLM models 3 × 32768 / median decoding speed; scored as truncation, not retried |
| Context overflow | after the first Active step: the episode ends, unanswered questions score 0; at the first request: setting N/A |
| Open-model context | `max_model_len = min(native, 65536)` |
| Repeats | protocol: 3 runs for Active v2 and random_walk_40, 1 run for the other settings; the paper reports the number of runs per model in the table notes (`paper_runs` in `docs/models/models.yaml` and `configs/models_api.yaml`) |
| Intervals | scene bootstrap + repeats, 2000 resamples, paired across settings |

Metrics: **total** (mean score over all questions), **answer rate** (answerable questions that got an answer), **selective score**
(mean score over answered answerable questions), **correct abstention** (unanswerable questions answered `cannot_determine`),
**Δ_RW = Active − random_walk_40** (main metric: value of the model's own exploration), **Δ_O = oracle_route − Active** (evidence gap),
and totals per question type T1–T5. See [docs/evaluation_protocol.md](docs/evaluation_protocol.md) for parsing rules, MRA, the
treatment of loops, environment errors and chat templates.

## Rebuilding the dataset (optional)

The released annotations are all you need for evaluation. To rebuild the dataset from the raw data (ARKitScenes 3DOD + raw `vga_wide`,
ScanNet++ iPhone data, meshes and semantic annotations, plus the official toolboxes; set `data_raw`, `work`, `arkitscenes_repo` and
`scannetpp_toolbox` in `configs/paths.yaml`), follow [docs/data_construction.md](docs/data_construction.md): Steps 0–10 (pose
validation, keyframes, objects and vocabulary, view graph and visibility, clusters, questions, paraphrases, blind-test debiasing,
human verification, freezing, merging and release packaging). Several steps end at a **human checkpoint** (GATE A–E) where a person
reviews a report or annotates pages before the next step; these decisions are recorded in the configuration files and cannot be
replaced by code. **The official benchmark is defined by the released frozen annotations; the construction pipeline is published for transparency and for building extensions.** A rebuild produces a new version and is not expected to reproduce `merged_v1.0` byte for byte (see the limitations in the guide).

## FAQ

**The tests skip many items.** Tests that read the real data are skipped without it; the synthetic smoke tests and all unit tests
still run. With the annotations (and frames) configured in `configs/paths.yaml`, the skipped tests run as well.

**`FreezeError` at start-up.** A file of the annotation package differs from its `SHA256SUMS`, or (with the frozen build layout) the data
or a hashed configuration file (`configs/build.yaml`, vocabularies, ...) differs from the frozen version. The evaluation refuses to run
on modified data; restore the files.

**`vllm_env is not set`.** Set `vllm_env:` in `configs/paths.yaml` to the `actspace-vllm` environment (printed by `install_envs.sh`).

**A model reports N/A for a setting.** Its context or images-per-prompt limit is too small for that setting (for example Passive-40
sends 41 images). This is reported as N/A, not changed per model.

**Can I change the prompts, budget or K?** Not for comparable numbers: API runs check the protocol values in `configs/eval.yaml` and
refuse to run when they are changed. Use a different version name and report it as a different protocol.

**Out of GPU memory / another process holds the GPU.** The queue checks free memory before starting a server, uses only the GPUs
given to it, and treats such failures as environment errors (not counted against the model); free the GPU and rerun the same command.

**How much do API runs cost?** Run `api_eval estimate` first; prices are in `configs/models_api.yaml` (check them against the
provider's current price page and set `prices_confirmed`).

**Where are the logs?** Each run directory has `trajectories.jsonl`, `calls.jsonl`, `summary.json`, `status.json`, `env.json` and,
once complete, `SHA256SUMS`; `actspace_eval.trajectory.replay` re-runs a trajectory file and checks that every step is reproduced.

## Citation, license and data terms

```bibtex
@misc{actspace2026,
  title  = {ActSpace-Bench},
  note   = {Citation will be added upon publication},
  year   = {2026}
}
```

- Code: MIT License, see [LICENSE](LICENSE).
- Data: this repository contains no data from ARKitScenes or ScanNet++. Their use is governed by their own licenses and terms
  (ARKitScenes: see https://github.com/apple/ARKitScenes; ScanNet++: ScanNet++ Terms of Use). The ActSpace annotations will be
  released separately, under terms compatible with the licenses of the source datasets (to be specified).
- Model weights are not distributed; each model is subject to its own license on Hugging Face.
