# ActSpace-Bench evaluation protocol

This document fixes the evaluation protocol used for every model in the paper (closed API models and open models alike).
The configuration that encodes it is `configs/eval.yaml`; `actspace_eval.providers.registry.assert_protocol` refuses to run
API evaluations when the protocol values in that file are changed. Rendered system prompts (Appendix A) and all prompt templates (Appendix B) are reproduced verbatim
(a unit test checks that they match `src/actspace_eval/prompts.py`).

## 1. Data and principles

- **Frozen data, read only.** The evaluation reads the frozen version `merged_v1.0`: ARKitScenes 41 scenes / 103 groups /
  425 questions and ScanNet++ 31 scenes / 131 groups / 569 questions (234 groups, 994 questions). At start-up all files of the
  annotation package are checked against its `SHA256SUMS` (with the frozen build layout: the version directory, the per-scene files,
  the hashed configuration files and the two source freeze manifests against `freeze_manifest.json`); any mismatch stops the run.
  The official benchmark is the released frozen annotation package.
- **No leakage.** The agent (model) receives only the public observation (Section 4). Raw frame IDs, file names, the view-graph
  structure, object boxes, visibility, oracle information and ground-truth answers exist only inside the environment and in the
  private trajectory logs.
- **One protocol for all models.** Prompts, budget, image history, image size and scoring are identical for every model; nothing
  is changed for individual models. What a model cannot do (context too small, too few images per prompt) is reported as N/A
  for that setting, with the reason.

## 2. Unit of evaluation

- **One question group = one episode.** The 3–5 questions of a group are given together at the start and share one view graph.
- Groups with a mask (`view_graph_mask`, used to create unanswerable questions) use the masked view graph: masked nodes can never
  be reached or shown.
- The episode starts at the group's `init_frame` (the START view); T4 questions refer to this start pose.

## 3. Actions and budget

| Action | Meaning | Steps |
|---|---|---|
| `forward` / `backward` / `left` / `right` | translate along the view-graph edge of this action to the neighbouring view | 1 |
| `turn_left` / `turn_right` | rotate in place along the view-graph edge of this action | 1 |
| `answer` | submit the answers to all questions of the group; ends the episode | 0 |

- **Cannot move**: the current view has no edge for the action, or the edge leads to a masked view. The position does not change,
  the feedback says so, and the step **still costs 1**. Turns can fail in the same way.
- **Invalid reply**: an action string not in the table, or a reply that cannot be parsed. The position does not change and the step
  costs 1. **Three consecutive invalid replies end the episode** (unanswered questions score 0, `forced = invalid_replies`); a valid
  move (including "cannot move") resets the counter.
- **Notes**: a reply may carry an optional `notes` string (not scored; cut to 300 characters).
- **Budget: 40 steps** (moves only; looking is free). After the budget is used, the observation says `must_answer` and the prompt
  asks for the answer; a move after that ends the episode with an empty answer (`forced_reason = move_after_budget`).
- Seven ScanNet++ groups need more than 40 steps for the shortest evidence route (`env.budget_infeasible`, 41–50 steps). They are
  run as usual but excluded from the main tables and reported separately in the appendix.

## 4. Observations

- **Questions** are numbered Q1, Q2, ... within the group; question IDs, group IDs and question types are not shown.
- **Photos** are the upright frames (rotated according to the capture orientation), without any overlay, downscaled with
  `INTER_AREA` so that the long side is at most **640 px**: ARKitScenes `vga_wide` 640 × 480 as is, ScanNet++ 1920 × 1440 → 640 × 480.
  Annotations such as visibility were computed at the original resolution.
- **Photo IDs** are random (`photo_` + 6 digits), drawn per episode from the episode seed. Returning to a view gives the same ID
  within an episode; IDs carry no frame number, timestamp or file name.
- **Feedback** is English text: odometry after a successful move, `Cannot move: ...` or `Invalid action: ...`.
- **Odometry** (relative to the view before the move): horizontal translation projected on the previous heading (forward / left)
  and the change of heading (left positive), rounded to 0.1 m and 5°; zero components are omitted from the text.
- **Active v2 (main setting)** additionally gives, in every observation, the cumulative pose relative to START (computed from
  the true poses, not by summing rounded steps; rounded to 0.1 m and 5°, heading in (−180°, 180°]) and states the typical step size
  in the system prompt: 0.3 m per translation and 45° per turn for both datasets (medians over all view-graph edges of
  `merged_v1.0`, rounded; ARKitScenes 0.315 m / 44.2°, ScanNet++ 0.309 m / 44.3°). The true change varies from step to step.

## 5. Image history in multi-turn episodes

Active episodes are multi-turn conversations. Every request contains the full text history (actions, photo IDs, feedback, poses)
but only the **START photo plus the 8 most recent photos** (`llm.history: {mode: last_k, k: 8, pin_start: true}`, at most 9
images). Older photos are replaced by a one-line placeholder that keeps their photo ID.

## 6. Settings

All settings share the answer format (Section 7). Static settings show all photos at once and accept only `answer`.

| Setting | Run directory | Photos given | Interaction |
|---|---|---|---|
| Blind | `blind` | none, only the questions; `cannot_determine` not accepted | answer directly |
| Passive-k (k = 8, 20, 40) | `passive_8`, `passive_20`, `passive_40` | START + k views drawn uniformly at random from the unmasked views of the group; the draw depends only on the group and k, so all models see the same photos | answer directly |
| oracle_min | `oracle_min` | START + the target views of the scripted oracle (`oracle_nodes`, `link_views`, and `path_views` for T4), shuffled | answer directly |
| oracle_route | `oracle_route` | the views along the shortest group-level evidence route of the scripted oracle, in visiting order, with every action and the cumulative pose | answer directly |
| random_walk_40 | `random_walk_40` | 40 uniformly random successful moves from START, photos in visiting order with actions and cumulative poses; 3 seeds | answer directly |
| **Active v2** (main) | `active_v2` | interactive exploration (Sections 3–5) with step size and cumulative pose | moves, then answer |
| Active v1 (ablation) | — | as Active v2 without step size and cumulative pose | moves, then answer |

- **Main metric:** exploration gain **Δ_RW = Active − random_walk_40**: random_walk_40 receives the same kind of photos and poses
  for the same 40 steps; only the actions are not chosen by the model.
- **Evidence gap:** **Δ_O = oracle_route − Active** (smaller is better).
- Secondary: Active − Passive-40 (Passive-40 draws 40 views from the whole group as a full-coverage reference).
- **Repeats:** the protocol runs Active v2 and random_walk_40 3 times (`--run-seed 0/1/2`, run names `<prefix>_r0/_r1/_r2`); the run
  seed changes photo IDs, model sampling seeds and the random walk. The other settings run once. The paper reports the number of runs
  per model in the table notes (`paper_runs` in `docs/models/models.yaml` and `configs/models_api.yaml`); intervals of settings with
  one run do not include run-to-run variation.

## 7. Answers, parsing and scoring

- An answer is one JSON object, e.g. `{"action": "answer", "answers": {"Q1": "B", "Q2": 2.4, "Q3": "cannot_determine"}}`:
  multiple-choice questions take the option letter, T1 a number in meters, and every question may take `"cannot_determine"`
  (except in Blind).
- Parsing: the first complete JSON object of the reply is used; keys are case-insensitive; option values are stripped and upper-cased
  and must be one of the option letters (`"B. front-left"` is accepted); T1 values must be finite and positive (number or numeric
  string); `cannot_determine` is case-insensitive. A missing or unparsable answer scores 0; if no JSON can be parsed, the whole group
  scores 0.
- **Scoring per question**
  - Multiple choice (T2–T5): 1 if the letter is correct.
  - T1 (answerable): mean relative accuracy, MRA = mean over θ ∈ {0.50, 0.55, …, 0.95} of 1[|ŷ − y| / y < 1 − θ], with y the
    unrounded ground-truth distance (`answer_raw`).
  - Unanswerable questions: 1 only for `cannot_determine`. `cannot_determine` on an answerable question scores 0.
  - Blind is scored on answerable questions only (it cannot abstain).
- **Columns of every score table**
  - **Total**: mean score over all questions (answerable and unanswerable; Blind: answerable only).
  - **Answer rate**: share of answerable questions that received an answer (`cannot_determine` or missing counts as not answered).
  - **Selective score**: mean score over the answered answerable questions.
  - **Correct abstention**: share of unanswerable questions answered with `cannot_determine` (= their score).
  - Δ_RW, Δ_O and the per-type totals T1–T5.
- **Intervals**: scene bootstrap with repeats, 2000 resamples, fixed seed. Each resample draws scenes with replacement (within each
  dataset; the combined table is stratified by dataset; the same scene draw for all settings so that differences are paired) and,
  for each setting with R repeats, draws R runs with replacement independently. Settings with one run therefore have intervals
  without the sampling randomness of the model.

## 8. Truncation, timeouts and context overflow

1. **Output limit**: `max_tokens = 32768` for every model (including thinking tokens). A reply that hits the limit
   (`finish_reason = length`, Anthropic `max_tokens`) is a truncated reply with no answer: an invalid move in an Active step, or 0 for
   all unanswered questions if it was the answering call.
2. **Wall-clock limit per call**, only to stop stuck or runaway requests: 600 s for API models and Cambrian-S; for the other vLLM
   models 3 × 32768 / (the model's median decoding speed), measured on completed calls with ≥ 100 output tokens (first token time
   excluded), updated once after 50 calls of the formal run and recorded in `env.json` and in every call record. A timeout is scored
   exactly like a truncation (`timeout_truncated = true`, `finish_reason = length`) and is not retried.
3. **Context overflow**: if, after the first Active step (when the conversation already contains the model's own replies), a request
   exceeds the model's context, the episode ends there (`forced = context_overflow`), unanswered questions score 0 and the run continues.
   If already the first request does not fit, the setting is N/A for that model.
4. **Open-model context**: `max_model_len = min(native context, 65536)` (largest normal input observed: 43,150 tokens).
5. **Loops count**: degenerate repetition that ends in truncation is scored like any other truncation. Reports additionally give the
   truncation rate and, for analysis only, the selective score without truncated groups.
6. **Program problems vs model behaviour**: the smoke gate before a formal run only checks for program problems (server errors after
   all retries, incomplete records, image token counts that differ between landscape and portrait, wrong interleaving of images and
   text). Low parse rates are model behaviour and are scored as they are. Server failures caused by the machine (GPU memory held by
   other processes, driver errors) are environment errors and are not counted against the model.
7. **Identical interleaving**: every model receives the same interleaved content (each image placed between its frame and pose text).
   Models whose official chat template does not accept list content or moves images to the front (SenseNova-SI, LLaVA-OneVision,
   MiniCPM-V) use format-adapted templates in `configs/chat_templates/` that change only the transport format; `env.json` records the
   template hash.

## 9. Models and generation settings

- **Reasoning**: `reasoning_effort = high` for all API models and settings; open models with a thinking mode run with thinking on.
- **Open models**: served with vLLM 0.19.1 (OpenAI-compatible API) in bf16 (LLaVA-OneVision in its stored float16), no quantization,
  tensor parallel when one 40 GB GPU is not enough. Sampling parameters come from the model's `generation_config.json`; without one,
  temperature 0.7 and top_p 0.9. Per-call seeds are derived from run seed, group and call index. JSON mode (structured output) is on;
  thinking models use the reasoning parser so that only the reply after thinking is constrained. Weights are identified by repository,
  revision and SHA-256 (`docs/models/`).
- **API models**: `configs/models_api.yaml` (model ID, endpoint, reasoning request, JSON mode, image detail, prices). Keys are read
  from environment variables only and never written to logs; every run records only the variable name.

## 10. Paper tables

`python -m actspace_eval.report_paper` exports the paper tables for API and open models from the run directories:

- Main tables (combined, ARKitScenes, ScanNet++): Blind, Passive-40, random_walk_40, oracle_route, Active total, answer rate,
  selective score, correct abstention, Δ_RW, Δ_O, T1–T5, plus a Chance row (multiple choice 1 / number of options; T1 the expected MRA
  of a uniform guess within the range of the answerable T1 ground truths; unanswerable 0). Appendix: oracle_min, Passive-8/20,
  truncation rates, scores without truncated groups, the 7 budget-infeasible groups.
- 227 groups (234 minus the 7 budget-infeasible ones); 962 questions (Blind: 899 answerable questions).
- astra was evaluated on the 32-group tiny subset only (Blind, random_walk_40, Active v2). Its row is marked with † and the footnote
  "evaluated on the 32-group tiny subset only", is not compared or marked with the 227-group rows, and is checked separately.
- **Rounding**: percentages with one decimal. Δ_RW and Δ_O are the differences of the two displayed values (each rounded to 0.1
  first), so they can be recomputed from the table; the self-check verifies this exactly. Significance is judged from intervals
  computed on unrounded data (score tables); the unrounded differences are kept in `paper_main.csv`.
- **Marks**: within each model group (API, open), best per column in bold and second best underlined, on unrounded values (Δ columns on their displayed values); for Δ_O
  smaller is better, for all other columns larger is better; Chance is not marked.
- **Self check** (`self_check.md`): group and question counts, re-scoring every question from the trajectories and the frozen data and
  comparing with `summary.json`, consistency with the score tables, combined = question-weighted mean of the two datasets, differences
  and rounding, uniqueness of the marks.

## 11. Trajectory logs and reproducibility

Each run directory contains `trajectories.jsonl` (`episode_start`, `step`, `episode_end` records with raw view IDs, actions, photo IDs,
feedback, odometry, raw and parsed answers, scores and forced-end reasons), `calls.jsonl` (every model call and retry with usage,
finish reason and timing), `summary.json`, `status.json`, `env.json` (model, weights hash, chat-template hash, sampling, seeds, library
versions, GPUs, data version and manifest hash, code commit) and, when sealed, `SHA256SUMS`.
`actspace_eval.trajectory.replay` re-runs every episode from its seed and actions and requires identical views, photo IDs, feedback
and scores. Randomness comes only from photo IDs, random walks and model sampling, all derived from the config seed (20261102), the
run seed and the group ID.

## Appendix A: rendered system prompts (budget 40, K = 8)

The system prompts exactly as sent, rendered from the templates in Appendix B. `api_eval verify-wire` checks recorded requests against these blocks.

**Active v1 (ablation; START photo + the last 8 photos)**

```text
You are exploring an indoor scene in order to answer questions about it.

You start at a fixed viewpoint and see the photograph taken there. You can move between nearby viewpoints; after every move you see the photograph taken at your new viewpoint. All questions are given at the start and are answered together at the end.

Actions
- "forward", "backward", "left", "right": move to a nearby viewpoint in that direction, relative to the direction you are currently facing.
- "turn_left", "turn_right": turn to a nearby viewpoint that faces further to the left or right (your position may also shift slightly).
- "answer": submit your answers to all questions. This ends the task.

Budget
- You have 40 steps. Every move action uses one step, including a move that fails because there is no viewpoint in that direction ("Cannot move"); your position then stays the same.
- A reply that cannot be read as a single JSON object, or that does not name a valid action, is invalid: it also uses one step, and you do not move. After 3 invalid replies in a row the task ends, and all questions count as unanswered and wrong.
- Answering does not use a step. You may answer at any time. When no steps remain, you must answer.

Odometry
- After each successful move you are told how your position and heading changed, measured relative to the direction you were facing before the move: the forward/backward and left/right distance in meters (rounded to 0.1 m) and how far you turned left or right in degrees (rounded to 5 degrees). Components that round to zero are omitted.

Photographs
- Each photograph has an ID such as photo_123456. Returning to a viewpoint you have already visited shows the same ID.
- The START photograph is always shown. Of the other photographs, only the 8 most recent are shown as images; earlier ones remain listed by ID in the text record of your moves.

Conventions
- All questions refer to the same indoor scene. Each object name in a question refers to one specific object in the scene.
- Distances are measured between the centers of the two objects, in meters.
- Directions are judged on the floor plane (heights are ignored), for a viewer standing at the stated position and facing the stated direction: an object is in "front" if the direction to its center is less than 90 degrees from the facing direction and "back" if it is more than 90 degrees; "left" and "right" are the two sides of the facing direction. Standing at an object means standing at its center; facing an object means facing toward its center.
- Your starting position and starting viewing direction are the position of the camera that took the START photograph and the horizontal direction in which it was pointing.

Answers
- Multiple-choice questions: give the option letter, for example "B".
- Numeric questions: give a number in meters, for example 2.4.
- If the information available to you does not allow you to determine the answer to a question, give "cannot_determine" for that question.
- Answer every question, using the question IDs (Q1, Q2, ...).

Reply format
Reply with a single JSON object and nothing else.
- To move: {"action": "forward"} (or any other move action)
- To answer: {"action": "answer", "answers": {"Q1": "B", "Q2": 2.4, "Q3": "cannot_determine"}}
The JSON object may also contain an optional "notes" field: a short string for your own observations (at most 300 characters; longer notes are cut off). Notes are not scored.
```

**Active v1 with full image history (control (a)): the second item of "Photographs" becomes**

```text
- All photographs you have seen so far are shown.
```

**Active v2 (main setting; step size from `env.step_scale`)**

```text
You are exploring an indoor scene in order to answer questions about it.

You start at a fixed viewpoint and see the photograph taken there. You can move between nearby viewpoints; after every move you see the photograph taken at your new viewpoint. All questions are given at the start and are answered together at the end.

Actions
- "forward", "backward", "left", "right": move to a nearby viewpoint in that direction, relative to the direction you are currently facing. A move covers about 0.3 m.
- "turn_left", "turn_right": turn to a nearby viewpoint that faces further to the left or right, by about 45 degrees (your position may also shift by about 0.3 m).
- Viewpoints are not on a regular grid, so the actual change differs from move to move; the odometry below reports it.
- "answer": submit your answers to all questions. This ends the task.

Budget
- You have 40 steps. Every move action uses one step, including a move that fails because there is no viewpoint in that direction ("Cannot move"); your position then stays the same. Turns can fail in the same way as other moves.
- A reply that cannot be read as a single JSON object, or that does not name a valid action, is invalid: it also uses one step, and you do not move. After 3 invalid replies in a row the task ends, and all questions count as unanswered and wrong.
- Answering does not use a step. You may answer at any time. When no steps remain, you must answer.

Odometry
- After each successful move you are told how your position and heading changed, measured relative to the direction you were facing before the move: the forward/backward and left/right distance in meters (rounded to 0.1 m) and how far you turned left or right in degrees (rounded to 5 degrees). Components that round to zero are omitted.
- With every observation you are also given your pose relative to START, in fixed coordinates defined by your starting position and starting viewing direction, measured on the floor plane: "forward" is the distance along the starting viewing direction (negative = behind the starting position), "left" is the distance to the left of it (negative = to the right), both in meters (rounded to 0.1 m); "heading" is the angle from the starting viewing direction to your current viewing direction, positive = turned to the left, between -180 and 180 degrees (rounded to 5 degrees). At START the pose is forward 0.0 m, left 0.0 m, heading 0 degrees.

Photographs
- Each photograph has an ID such as photo_123456. Returning to a viewpoint you have already visited shows the same ID.
- The START photograph is always shown. Of the other photographs, only the 8 most recent are shown as images; earlier ones remain listed by ID in the text record of your moves.

Conventions
- All questions refer to the same indoor scene. Each object name in a question refers to one specific object in the scene.
- Distances are measured between the centers of the two objects, in meters.
- Directions are judged on the floor plane (heights are ignored), for a viewer standing at the stated position and facing the stated direction: an object is in "front" if the direction to its center is less than 90 degrees from the facing direction and "back" if it is more than 90 degrees; "left" and "right" are the two sides of the facing direction. Standing at an object means standing at its center; facing an object means facing toward its center.
- Your starting position and starting viewing direction are the position of the camera that took the START photograph and the horizontal direction in which it was pointing.

Answers
- Multiple-choice questions: give the option letter, for example "B".
- Numeric questions: give a number in meters, for example 2.4.
- If the information available to you does not allow you to determine the answer to a question, give "cannot_determine" for that question.
- Answer every question, using the question IDs (Q1, Q2, ...).

Reply format
Reply with a single JSON object and nothing else.
- To move: {"action": "forward"} (or any other move action)
- To answer: {"action": "answer", "answers": {"Q1": "B", "Q2": 2.4, "Q3": "cannot_determine"}}
The JSON object may also contain an optional "notes" field: a short string for your own observations (at most 300 characters; longer notes are cut off). Notes are not scored.
```

**random_walk_40 and oracle_route (identical text; only photos and route differ)**

```text
You are answering questions about an indoor scene from photographs taken along a path through it.

The path starts at the viewpoint of the START photograph, which shows your starting position and the direction you are facing. The record lists every move along the path in order. Each viewpoint's photograph is given once, in the order in which the viewpoints were first visited; a later return to the same viewpoint refers back to its photograph by ID. You cannot move or request more photographs.

Each entry of the record gives the pose at that viewpoint relative to START, in fixed coordinates defined by the starting position and starting viewing direction, measured on the floor plane: "forward" is the distance along the starting viewing direction (negative = behind the starting position), "left" is the distance to the left of it (negative = to the right), both in meters (rounded to 0.1 m); "heading" is the angle from the starting viewing direction to the viewing direction of the photograph, positive = turned to the left, between -180 and 180 degrees (rounded to 5 degrees). Moves are named relative to the direction faced before the move: "forward", "backward", "left", "right" move to a nearby viewpoint in that direction; "turn_left", "turn_right" turn to a nearby viewpoint that faces further to the left or right.

Conventions
- All questions refer to the same indoor scene. Each object name in a question refers to one specific object in the scene.
- Distances are measured between the centers of the two objects, in meters.
- Directions are judged on the floor plane (heights are ignored), for a viewer standing at the stated position and facing the stated direction: an object is in "front" if the direction to its center is less than 90 degrees from the facing direction and "back" if it is more than 90 degrees; "left" and "right" are the two sides of the facing direction. Standing at an object means standing at its center; facing an object means facing toward its center.
- Your starting position and starting viewing direction are the position of the camera that took the START photograph and the horizontal direction in which it was pointing.

Answers
- Multiple-choice questions: give the option letter, for example "B".
- Numeric questions: give a number in meters, for example 2.4.
- If the information available to you does not allow you to determine the answer to a question, give "cannot_determine" for that question.
- Answer every question, using the question IDs (Q1, Q2, ...).

Reply format
Reply with a single JSON object and nothing else:
{"action": "answer", "answers": {"Q1": "B", "Q2": 2.4, "Q3": "cannot_determine"}}
The JSON object may also contain an optional "notes" field: a short string for your own observations (at most 300 characters; longer notes are cut off). Notes are not scored.
```

**Passive-k and oracle_min (identical text; example with 41 photos)**

```text
You are answering questions about an indoor scene from photographs.

You are given 41 photographs taken at different viewpoints in the same scene. The first photograph, marked START, shows your starting position and the direction you are facing. The other photographs are not in any particular order. You cannot move or request more photographs.

Conventions
- All questions refer to the same indoor scene. Each object name in a question refers to one specific object in the scene.
- Distances are measured between the centers of the two objects, in meters.
- Directions are judged on the floor plane (heights are ignored), for a viewer standing at the stated position and facing the stated direction: an object is in "front" if the direction to its center is less than 90 degrees from the facing direction and "back" if it is more than 90 degrees; "left" and "right" are the two sides of the facing direction. Standing at an object means standing at its center; facing an object means facing toward its center.
- Your starting position and starting viewing direction are the position of the camera that took the START photograph and the horizontal direction in which it was pointing.

Answers
- Multiple-choice questions: give the option letter, for example "B".
- Numeric questions: give a number in meters, for example 2.4.
- If the information available to you does not allow you to determine the answer to a question, give "cannot_determine" for that question.
- Answer every question, using the question IDs (Q1, Q2, ...).

Reply format
Reply with a single JSON object and nothing else:
{"action": "answer", "answers": {"Q1": "B", "Q2": 2.4, "Q3": "cannot_determine"}}
The JSON object may also contain an optional "notes" field: a short string for your own observations (at most 300 characters; longer notes are cut off). Notes are not scored.
```

**Blind (must guess; `cannot_determine` not accepted; scored on answerable questions only)**

```text
You are answering questions about an indoor scene. No photographs of the scene are available in this setting; answer using general knowledge of how indoor spaces are typically arranged.

Conventions
- All questions refer to the same indoor scene. Each object name in a question refers to one specific object in the scene.
- Distances are measured between the centers of the two objects, in meters.
- Directions are judged on the floor plane (heights are ignored), for a viewer standing at the stated position and facing the stated direction: an object is in "front" if the direction to its center is less than 90 degrees from the facing direction and "back" if it is more than 90 degrees; "left" and "right" are the two sides of the facing direction. Standing at an object means standing at its center; facing an object means facing toward its center.
- Your starting position and starting viewing direction are the position of the camera that took the START photograph and the horizontal direction in which it was pointing.

Answers
- Multiple-choice questions: give the option letter, for example "B".
- Numeric questions: give a number in meters, for example 2.4.
- Give your best guess for every question; "cannot_determine" is not accepted in this setting.
- Answer every question, using the question IDs (Q1, Q2, ...).

Reply format
Reply with a single JSON object and nothing else:
{"action": "answer", "answers": {"Q1": "B", "Q2": 2.4}}
The JSON object may also contain an optional "notes" field: a short string for your own observations (at most 300 characters; longer notes are cut off). Notes are not scored.
```

## Appendix B: prompt templates (verbatim)

The templates below are the exact strings in `src/actspace_eval/prompts.py`; placeholders in braces are filled per episode.

### `CONVENTIONS`

```text
Conventions
- All questions refer to the same indoor scene. Each object name in a question refers to one specific object in the scene.
- Distances are measured between the centers of the two objects, in meters.
- Directions are judged on the floor plane (heights are ignored), for a viewer standing at the stated position and facing the stated direction: an object is in "front" if the direction to its center is less than 90 degrees from the facing direction and "back" if it is more than 90 degrees; "left" and "right" are the two sides of the facing direction. Standing at an object means standing at its center; facing an object means facing toward its center.
- Your starting position and starting viewing direction are the position of the camera that took the START photograph and the horizontal direction in which it was pointing.
```

### `ANSWER_RULES`

```text
Answers
- Multiple-choice questions: give the option letter, for example "B".
- Numeric questions: give a number in meters, for example 2.4.
- If the information available to you does not allow you to determine the answer to a question, give "cannot_determine" for that question.
- Answer every question, using the question IDs (Q1, Q2, ...).
```

### `ANSWER_RULES_BLIND`

```text
Answers
- Multiple-choice questions: give the option letter, for example "B".
- Numeric questions: give a number in meters, for example 2.4.
- Give your best guess for every question; "cannot_determine" is not accepted in this setting.
- Answer every question, using the question IDs (Q1, Q2, ...).
```

### `ANSWER_JSON`

```text
{"action": "answer", "answers": {"Q1": "B", "Q2": 2.4, "Q3": "cannot_determine"}}
```

### `ANSWER_JSON_BLIND`

```text
{"action": "answer", "answers": {"Q1": "B", "Q2": 2.4}}
```

### `NOTES_RULE`

```text
The JSON object may also contain an optional "notes" field: a short string for your own observations (at most 300 characters; longer notes are cut off). Notes are not scored.
```

### `SYSTEM_ACTIVE`

```text
You are exploring an indoor scene in order to answer questions about it.

You start at a fixed viewpoint and see the photograph taken there. You can move between nearby viewpoints; after every move you see the photograph taken at your new viewpoint. All questions are given at the start and are answered together at the end.

Actions
- "forward", "backward", "left", "right": move to a nearby viewpoint in that direction, relative to the direction you are currently facing.
- "turn_left", "turn_right": turn to a nearby viewpoint that faces further to the left or right (your position may also shift slightly).
- "answer": submit your answers to all questions. This ends the task.

Budget
- You have {budget} steps. Every move action uses one step, including a move that fails because there is no viewpoint in that direction ("Cannot move"); your position then stays the same.
- A reply that cannot be read as a single JSON object, or that does not name a valid action, is invalid: it also uses one step, and you do not move. After 3 invalid replies in a row the task ends, and all questions count as unanswered and wrong.
- Answering does not use a step. You may answer at any time. When no steps remain, you must answer.

Odometry
- After each successful move you are told how your position and heading changed, measured relative to the direction you were facing before the move: the forward/backward and left/right distance in meters (rounded to 0.1 m) and how far you turned left or right in degrees (rounded to 5 degrees). Components that round to zero are omitted.

Photographs
- Each photograph has an ID such as photo_123456. Returning to a viewpoint you have already visited shows the same ID.
{history_rule}

{conventions}

{answer_rules}

Reply format
Reply with a single JSON object and nothing else.
- To move: {{"action": "forward"}} (or any other move action)
- To answer: {answer_json}
{notes_rule}
```

### `SYSTEM_ACTIVE_V2`

```text
You are exploring an indoor scene in order to answer questions about it.

You start at a fixed viewpoint and see the photograph taken there. You can move between nearby viewpoints; after every move you see the photograph taken at your new viewpoint. All questions are given at the start and are answered together at the end.

Actions
- "forward", "backward", "left", "right": move to a nearby viewpoint in that direction, relative to the direction you are currently facing. A move covers about {move_m} m.
- "turn_left", "turn_right": turn to a nearby viewpoint that faces further to the left or right, by about {turn_deg} degrees (your position may also shift by about {move_m} m).
- Viewpoints are not on a regular grid, so the actual change differs from move to move; the odometry below reports it.
- "answer": submit your answers to all questions. This ends the task.

Budget
- You have {budget} steps. Every move action uses one step, including a move that fails because there is no viewpoint in that direction ("Cannot move"); your position then stays the same. Turns can fail in the same way as other moves.
- A reply that cannot be read as a single JSON object, or that does not name a valid action, is invalid: it also uses one step, and you do not move. After 3 invalid replies in a row the task ends, and all questions count as unanswered and wrong.
- Answering does not use a step. You may answer at any time. When no steps remain, you must answer.

Odometry
- After each successful move you are told how your position and heading changed, measured relative to the direction you were facing before the move: the forward/backward and left/right distance in meters (rounded to 0.1 m) and how far you turned left or right in degrees (rounded to 5 degrees). Components that round to zero are omitted.
{pose_rule}

Photographs
- Each photograph has an ID such as photo_123456. Returning to a viewpoint you have already visited shows the same ID.
{history_rule}

{conventions}

{answer_rules}

Reply format
Reply with a single JSON object and nothing else.
- To move: {{"action": "forward"}} (or any other move action)
- To answer: {answer_json}
{notes_rule}
```

### `POSE_RULE`

```text
- With every observation you are also given your pose relative to START, in fixed coordinates defined by your starting position and starting viewing direction, measured on the floor plane: "forward" is the distance along the starting viewing direction (negative = behind the starting position), "left" is the distance to the left of it (negative = to the right), both in meters (rounded to 0.1 m); "heading" is the angle from the starting viewing direction to your current viewing direction, positive = turned to the left, between -180 and 180 degrees (rounded to 5 degrees). At START the pose is forward 0.0 m, left 0.0 m, heading 0 degrees.
```

### `POSE_LINE`

```text
Pose relative to START: forward {forward_m:.1f} m, left {left_m:.1f} m, heading {heading_deg} degrees.
```

### `HISTORY_RULE_LAST_K_PIN`

```text
- The START photograph is always shown. Of the other photographs, only the {k} most recent are shown as images; earlier ones remain listed by ID in the text record of your moves.
```

### `HISTORY_RULE_LAST_K`

```text
- Only the {k} most recent photographs are shown as images. Earlier photographs remain listed by ID in the text record of your moves.
```

### `HISTORY_RULE_FULL`

```text
- All photographs you have seen so far are shown.
```

### `SYSTEM_STATIC`

```text
You are answering questions about an indoor scene from photographs.

You are given {n_photos} photographs taken at different viewpoints in the same scene. The first photograph, marked START, shows your starting position and the direction you are facing. The other photographs are not in any particular order. You cannot move or request more photographs.

{conventions}

{answer_rules}

Reply format
Reply with a single JSON object and nothing else:
{answer_json}
{notes_rule}
```

### `SYSTEM_ROUTE`

```text
You are answering questions about an indoor scene from photographs taken along a path through it.

The path starts at the viewpoint of the START photograph, which shows your starting position and the direction you are facing. The record lists every move along the path in order. Each viewpoint's photograph is given once, in the order in which the viewpoints were first visited; a later return to the same viewpoint refers back to its photograph by ID. You cannot move or request more photographs.

{pose_rule}

{conventions}

{answer_rules}

Reply format
Reply with a single JSON object and nothing else:
{answer_json}
{notes_rule}
```

### `POSE_RULE_ROUTE`

```text
Each entry of the record gives the pose at that viewpoint relative to START, in fixed coordinates defined by the starting position and starting viewing direction, measured on the floor plane: "forward" is the distance along the starting viewing direction (negative = behind the starting position), "left" is the distance to the left of it (negative = to the right), both in meters (rounded to 0.1 m); "heading" is the angle from the starting viewing direction to the viewing direction of the photograph, positive = turned to the left, between -180 and 180 degrees (rounded to 5 degrees). Moves are named relative to the direction faced before the move: "forward", "backward", "left", "right" move to a nearby viewpoint in that direction; "turn_left", "turn_right" turn to a nearby viewpoint that faces further to the left or right.
```

### `SYSTEM_BLIND`

```text
You are answering questions about an indoor scene. No photographs of the scene are available in this setting; answer using general knowledge of how indoor spaces are typically arranged.

{conventions}

{answer_rules}

Reply format
Reply with a single JSON object and nothing else:
{answer_json}
{notes_rule}
```

### `QUESTION_MC`

```text
{qid} (multiple choice) {text}
{options}
```

### `QUESTION_NUM`

```text
{qid} (numeric, meters) {text}
```

### `USER_START_ACTIVE`

```text
Questions
{questions}

You are at your starting viewpoint, {photo_id} (START).
{{image}}
Steps used: 0 of {budget}; {budget} remaining. Choose your next action.
```

### `STEP_MOVED`

```text
Step {t}: "{action}". {feedback} You are now at {photo_id}{revisit}.
{{image}}
```

### `STEP_CANNOT_MOVE`

```text
Step {t}: "{action}". Cannot move: no viewpoint in that direction. You are still at {photo_id}; no new photograph.
```

### `STEP_INVALID`

```text
Step {t}: your reply was not a valid action, so one step was used and you did not move. You are still at {photo_id}. Reply with a single JSON object as described in the instructions.
```

### `STEP_STATUS`

```text
Steps used: {used} of {budget}; {left} remaining.
```

### `STEP_MUST_ANSWER`

```text
No steps remain. Reply with the answer action now.
```

### `STEP_CHOOSE`

```text
Choose your next action.
```

### `IMAGE_OMITTED`

```text
[{photo_id}: image no longer shown]
```

### `USER_START_ACTIVE_V2`

```text
Questions
{questions}

You are at your starting viewpoint, {photo_id} (START).
{{image}}
{pose}
Steps used: 0 of {budget}; {budget} remaining. Choose your next action.
```

### `USER_ROUTE`

```text
Questions
{questions}

Path record ({n_moves} moves, {n_photos} photographs):
{record}
Answer all questions.
```

### `ROUTE_START`

```text
START {photo_id}. {pose}
{{image}}
```

### `ROUTE_MOVE_NEW`

```text
Move {t}: "{action}" to {photo_id}. {pose}
{{image}}
```

### `ROUTE_MOVE_REVISIT`

```text
Move {t}: "{action}" back to {photo_id} (photograph shown above). {pose}
```

### `USER_STATIC`

```text
Questions
{questions}

Photographs ({n_photos}):
{photos}
Answer all questions.
```

### `PHOTO_LINE_START`

```text
START {photo_id}
{{image}}
```

### `PHOTO_LINE`

```text
{photo_id}
{{image}}
```

### `USER_BLIND`

```text
Questions
{questions}

Answer all questions.
```
