# Building ActSpace-Bench from raw data

This guide describes how `merged_v1.0` was built and how to rerun the pipeline. **You do not need it to evaluate models.** The
released annotations plus `scripts/rebuild_frames.py` are enough for that.

**The official benchmark is defined by the released frozen annotations. The construction pipeline is published for transparency and
for building extensions; a rebuild is not expected to reproduce `merged_v1.0` byte for byte.**

Rerunning the pipeline produces a **new version**. Several steps depend on human decisions that are already recorded in the
configuration files, so a rebuild with unchanged configs aims to reproduce the frozen data. The frozen `merged_v1.0` is identified
only by the hashes in its `freeze_manifest.json`. See [Known limitations](#known-limitations-of-a-rebuild) before you start.

## Overview

```
Step 0  inventory and data preparation
Step 1  scene parsing, coordinate frames, upright images      -> GATE A  pose / convention validation (every pose)
Step 2  frame extraction and keyframe selection
Step 3  objects and category vocabulary                       -> GATE B  vocabulary review
Step 4  view graph and per-frame visibility                   -> visibility spot check (>= 95% agreement)
Step 5  object clusters and start views
Step 6  template questions, answers, constraints, evidence, unanswerable masking, deduplication, balancing
                                                              -> GATE C1 / C2  pilot acceptance (5 scenes per dataset)
Step 7  paraphrases (template-level paraphrase bank)
Step 8  blind test and debiasing
Step 9  human verification (two-phase pages)                  -> GATE D  human verification; freeze single-dataset version
Step 10 merge, rerun Step 8 on the merged pre-prune sets, freeze, package the release  -> GATE E  final acceptance
```

- Pilot first: until a dataset passes its pilot gate (C1 = ARKitScenes, C2 = ScanNet++), only its 5 pilot scenes are processed.
  Modules default to the pilot scenes (`arkitscenes.pilot_scenes`, `scannetpp.pilot_scenes` in `configs/build.yaml`) when `--scenes` is omitted.
- Each dataset is taken to Step 9 and frozen on its own (`arkit_only_v0.11`, `scannetpp_only_v0.2`). The merged version reruns Step 8
  on the union of both pre-prune sets (`configs/merge.yaml`) and is frozen as `merged_v1.0`.

## Setup

1. Install the `actspace` environment (`bash env/install_envs.sh --core`) and ffmpeg.
2. Get the raw data from the official sources (see the README) and lay it out as described in `configs/paths.example.yaml`
   (`data_raw`: ARKitScenes `3dod`, `raw`, `metadata.csv`, `3dod_train_val_splits.csv`; ScanNet++ `data/`, `splits/`, `metadata/`).
   The pipeline needs:

   | Dataset | Files per scene |
   |---|---|
   | ARKitScenes 3DOD (Validation) | `lowres_wide/`, `lowres_wide_intrinsics/`, `lowres_wide.traj`, `lowres_depth/`, `<id>_3dod_mesh.ply`, `<id>_3dod_annotation.json` |
   | ARKitScenes raw (Validation, selected videos only) | `vga_wide/`, `vga_wide_intrinsics/` |
   | ScanNet++ | `iphone/rgb.mkv`, `iphone/pose_intrinsic_imu.json`, `iphone/depth.bin`, `scans/mesh_aligned_0.05.ply`, `scans/mesh_aligned_0.05_semantic.ply`, `scans/segments.json`, `scans/segments_anno.json` |

3. Clone the official toolboxes and set `arkitscenes_repo` and `scannetpp_toolbox` in `configs/paths.yaml`
   (defaults: `third_party/ARKitScenes`, `third_party/scannetpp`):

   ```bash
   git clone https://github.com/apple/ARKitScenes.git third_party/ARKitScenes
   git clone https://github.com/scannetpp/scannetpp.git third_party/scannetpp
   ```

4. Set `work` (build outputs; the directory must be named `work`), `logs`, `sandbox` and `release` in `configs/paths.yaml`.
5. Run the pipeline inside a git checkout with committed code. Freezing and packaging record the git commit.

All commands run from the repository root with `export PYTHONPATH=src`. Below, `W` is `paths.work` and `R` is `reports/`, where all
step reports are written.

## Global conventions (implemented once, unit tested)

- **Coordinates**: gravity-aligned scene frame (z up), validated per pose in GATE A. Images are rotated upright according to the
  capture orientation; intrinsics and poses are rotated with them.
- **Directions** (T3/T4/T5): judged on the floor plane. For T3 and T5 the viewer stands at the center of the standing object and faces
  the center of the facing object. For T4 the viewer stands at the START camera position and faces the horizontal projection of the
  camera's forward axis. Quadrants are front-left / front-right / back-left / back-right; angles within 20° of a boundary are excluded.
- **Distances**: between object centers, in 3D (T1, T2) or horizontally (separation rules).
- **Language**: English questions. Objects are named only by unique category names; size references were removed.

## ARKitScenes

| Step | Command | Output | Human checkpoint |
|---|---|---|---|
| 0 | `python -m actspace_build.step0_inventory` (`--no-estimate` skips the HTTP size estimate) | `W/arkitscenes/_step0/` (one video per `visit_id`), `R/step0_inventory.md` | approve the download volume |
| 1 / A | `python -m actspace_build.full_arkit gate_a --workers 16` | per-scene `gate_a.json` (pose convention, up axis, rotation, per-pose depth check) | **GATE A**: review `R/step1_gateA.md` (`python -m actspace_build.reports gate_a`). Confirmed orientation conflicts go into `orientation.confirmed_conflict_scenes` in `configs/build.yaml` |
| 3 / B | `python -m actspace_build.full_arkit objects --workers 16` | `objects.json`, `object_points.npz` | **GATE B**: review `vocab/categories.tsv` and `vocab/label_mapping_arkitscenes.tsv`. The blacklist is `configs/object_blacklist.yaml` |
| raw download | `python -m actspace_build.full_arkit download --workers 16` | `vga_wide` frames via the official `download_data.py` into `<arkitscenes_repo>/data/raw/Validation/` | the same directory must be `data_raw/arkitscenes/raw` (symlink) |
| 1, 2, 4 | `python -m actspace_build.full_arkit scenes --workers 16` | `frames_all.json` (pose interpolation, upright K/T), `keyframes.json`, `view_graph.json`, `visibility.npz` | visibility spot check: `reports spotcheck_images`, judge the pairs in `R/visibility_spotcheck_judgments.tsv`, then `reports spotcheck` (needs ≥ 95%) |
| 4 | `python -m actspace_build.full_arkit check_graph` | stops if too many scenes have a disconnected graph | review if it stops |
| 5–6 | `python -m actspace_build.full_arkit questions` | `W/arkitscenes/_step6/test_groups.jsonl` | **GATE C1** on the pilot (`reports step6`, `reports gate_c`) before the full run |
| summary | `python -m actspace_build.full_arkit summary` and `python -m actspace_build.integrity check --stage _step6` | `R/arkit_full_summary.md`, integrity report | review |
| 7 | `python -m actspace_build.step7_rewrite validate` then `python -m actspace_build.step7_rewrite run` | `W/arkitscenes/_step7/test_groups.jsonl` (paraphrases from `configs/paraphrases.yaml`, chi-square check) | review when the paraphrase bank changes |
| 8 | `python -m actspace_build.step8_debias` | `W/arkitscenes/_step8/<debias.output_version>/` (feature-based blind test with controls, iterative pruning, rebalancing) | stops if a template loses > 40% |
| 9 / D | `python -m actspace_build.gate_d pages --version <ver>`, `... package --version <ver>`, then `... summary <annotation files>` | verification pages; report draft | **GATE D**: a person verifies a stratified sample in the browser (blind answer, then review of the gold answer and evidence views). Pass: ≥ 95% correct, ≤ 5% flagged. Flagged objects go into the blacklist and Steps 3–8 are rerun under a new version name |
| freeze | `python -m actspace_build.gate_d carryover --version <ver>`, `python -m actspace_build.freeze freeze --version <ver> --build-commit <sha>:<note>`, `python -m actspace_build.freeze verify --version <ver>` | `freeze_manifest.json`; version directory read-only | decision to freeze |

The pilot uses the per-step modules directly: `step1_arkit gate_a|frames`, `step2_keyframes`, `step3_objects vocab_stats|objects`,
`step4_graph_visibility`, `step56_questions` (all with `--scenes`), and `reports <name>` for the reports.

## ScanNet++

Scenes: `splits/nvs_sem_val.txt`. After GATE A, only the scenes in `W/scannetpp/_full/passed_scenes.txt` are processed.

| Step | Command | Output | Human checkpoint |
|---|---|---|---|
| 0 | `python -m actspace_build.scannetpp decode --scenes <scenes> --workers 8` | decoded RGB and depth (written into the raw tree `iphone/rgb`, `iphone/depth`, which must be writable) | — |
| 0 | `python -m actspace_build.scannetpp_step0` | pilot inventory `R/scannetpp_step0_inventory.md` | approve the download volume |
| 1 / A | `python -m actspace_build.scannetpp_step1 gate_a --scenes <scenes> --workers 5`, `python -m actspace_build.scannetpp_step1 report`, `python -m actspace_build.scannetpp_full gate_a_pass` | `gate_a.json`, `frames_all.json`, `passed_scenes.txt` (exits 3 if fewer than 50% pass) | **GATE A**: review `R/scannetpp_step1_gateA.md` |
| 2 | `python -m actspace_build.scannetpp_step2 run --scenes <passed>` | `keyframes.json`, `R/scannetpp_step2_keyframes.md` | — |
| 3 / B | `python -m actspace_build.scannetpp_vocab build`, `python -m actspace_build.scannetpp_step3 objects --scenes <passed> --workers 5`, `python -m actspace_build.scannetpp_step3_report` | `vocab/label_mapping_scannetpp.tsv`, `vocab/categories_scannetpp_draft.tsv`, objects | **GATE B**: review the vocabulary draft. Decisions are constants in `scannetpp_vocab.py`; the blacklist is `configs/object_blacklist_scannetpp.yaml` |
| 4 | `python -m actspace_build.scannetpp_step4 run --scenes <passed> --workers 5` | `view_graph.json`, `visibility.npz` (prints STOP if too many graphs are disconnected) | visibility spot check: `scannetpp_step4_report spotcheck_images` / `spotcheck` (≥ 95%) |
| 5–6 | `python -m actspace_build.scannetpp_step56 --scenes <passed>` | `W/scannetpp/_step6/test_groups.jsonl` | **GATE C2** on the pilot (`scannetpp_gateC`). After the full run: `scannetpp_full_report report`, object audit `scannetpp_audit tiles` / `report` |
| 7–8 | `python -m actspace_build.scannetpp_step78 step7`, `... step8`, `... integrity` | `W/scannetpp/_step8/scannetpp_only_v0.2/` | stops if a template loses > 40% |
| 9 / D | `python -m actspace_build.scannetpp_gateD sample`, `pages`, `package`, then `summary <files>` and `baseline <files>` | verification pages and report | **GATE D** as for ARKitScenes |
| freeze | `python -m actspace_build.scannetpp_freeze freeze`, `python -m actspace_build.scannetpp_freeze verify` | `freeze_manifest.json` (includes the pre-prune sets) | decision to freeze |

## Merge, freeze and release (Step 10)

| Command | What it does | Human checkpoint |
|---|---|---|
| `python -m actspace_build.merge_dryrun stats`, `sepfilter`, `step8`, `analyze` | optional dry run in the sandbox (`paths.sandbox`) | decisions recorded in `configs/merge.yaml` (e.g. `quadrant_rebalance_qids`) |
| `python -m actspace_build.merge_build build` | union of both pre-prune sets → standing-point separation filter → option-letter balance → Step 8 → fixed quadrant rebalancing → blind recheck → `W/merged/_step8/merged_v1.0/` | — |
| `python -m actspace_build.merge_build integrity` | independent recomputation of all answers and the integrity checks | review the report |
| `python -m actspace_build.merge_build freeze` / `verify` | freeze manifest (version dir, scene files, configs, both source manifests) and verification | decision to freeze |
| `python -m actspace_build.merge_release release` | verifies everything, then packages `<release>/merged_v1.0/` (`scenes/`, `questions/`, `build_info.json`, `SHA256SUMS`, `scripts/rebuild_frames.py`) without images | — |
| `python -m actspace_build.merge_release check`, `python -m actspace_build.merge_release rebuild-test` | checks the package (no images, tokens or personal paths; hashes); rebuilds frames in a clean directory and compares pixels | **GATE E**: final acceptance |

## Unanswerable questions and balance

- **Masking**: for a candidate object X (T4(X) or T1(X, cluster object)), every node where X is visible or partially visible is removed;
  if the remaining graph is no longer strongly connected, the part not connected to the start view is removed as well (the start view
  is never removed). The question then has the gold answer `cannot_determine`. Target 8–12% unanswerable questions overall.
- **Balance**: T3 and T4 quadrants 25% ± 5% each, T4 front/back 50% ± 5%, option letters balanced, T2 pair questions (constraints only) ≤ 20%.
- **Evidence**: `evidence_objs` (all objects in the question, including options), `oracle_nodes`, `link_views`, `path_views` (T4) and
  `oracle_steps`. These are used for the oracle settings and the evidence diagnostics.

## Known limitations of a rebuild

The pipeline was developed iteratively with human checkpoints. Some inputs and earlier versions exist only in the original working
copy, so a rebuild from this repository has these limitations:

1. **Git history.** `merge_dryrun regress` reads the frozen build configs with `git show 32a23ff:configs/build.yaml` and `7f0fceb`
   (`configs/merge.yaml`). These commits are not part of the public history, so `regress` cannot run. Freezing and packaging
   (`freeze`, `scannetpp_freeze`, `merge_build freeze`, `merge_release`) require a git checkout.
2. **Fixed question IDs.** `merge_build build` requires the 11 question IDs in `merge.quadrant_rebalance_qids`. A rebuild that does
   not reproduce the Step 8 output exactly must update this list (a human decision).
3. **GATE D tooling is tied to the original rounds.**
   - `gate_d` knows only the versions `arkit_only_v0.9`–`v0.11`. `gate_d sample` for v0.11 is disabled by design (no new sample).
     Later rounds need the sample lists of earlier rounds.
   - `gate_d blacklist` regenerates `configs/object_blacklist.yaml` from round 1 and would overwrite the final file. Do not run it.
   - `scannetpp_gateD` samples from `scannetpp_only_v0.1`.
   The human verification itself (annotations) is not part of this repository.
4. **Missing intermediate inputs.**
   - `W/arkitscenes/_step0/orientation_diagnostic.csv` is needed by `reports orientation_review`, `step3_objects referability_stats`
     and `reports step3`.
   - `W/scannetpp/_gateC/label_candidates.json` is needed by `scannetpp_gateC`.
   - The round-1 spot-check files are needed by `scannetpp_step4_report report`.
   These reports are diagnostics. The main path (`full_arkit`, `scannetpp_*` run commands, Steps 7–8, merge) does not need them.
5. **Layout assumptions.**
   - The ARKitScenes raw download directory and `data_raw/arkitscenes/raw` must be the same.
   - `arkit_3dod_root` must resolve to `<ARKitScenes>/data/3dod/Validation`.
   - The work directory must be named `work`.
   - `step7_rewrite run` writes `build_info.json` at the repository root (git-ignored).
6. **Order.** Pilot and full runs share `work/`. Write the GATE C reports before the full Step 6, which overwrites the pilot's `_step6`.
   Changing a blacklist after GATE D means rerunning Steps 3–8 (`full_arkit objects --force` → `questions` → `step7_rewrite run` →
   `step8_debias`) under a new version name (`debias.output_version` in `configs/build.yaml`; `VERSION` in `scannetpp_step78.py`).
7. **Hashed configuration.** `merged_v1.0` hashes `configs/build.yaml`, `configs/merge.yaml`, `configs/paraphrases.yaml`, both
   blacklists and the vocabularies. Any change, including regenerating the ScanNet++ vocabulary, makes the evaluation refuse the
   frozen data (start-up verification).
