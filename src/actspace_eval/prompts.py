"""Agent prompts (docs/evaluation_protocol.md; must not be used for real calls before review, see llm.prompts_approved in configs/eval.yaml).

Every template is defined once here; docs/evaluation_protocol.md reproduces them verbatim and a unit test checks that the two agree.
Messages are first built in a provider-independent structure: [{"role": ..., "parts": [("text", text) | ("image", photo ID, image)]}],
and models.py converts them to the format of the specific API. Prompts never mention scoring details or how the data were built.
"""
from __future__ import annotations

CONVENTIONS = """Conventions
- All questions refer to the same indoor scene. Each object name in a question refers to one specific object in the scene.
- Distances are measured between the centers of the two objects, in meters.
- Directions are judged on the floor plane (heights are ignored), for a viewer standing at the stated position and facing the stated direction: an object is in "front" if the direction to its center is less than 90 degrees from the facing direction and "back" if it is more than 90 degrees; "left" and "right" are the two sides of the facing direction. Standing at an object means standing at its center; facing an object means facing toward its center.
- Your starting position and starting viewing direction are the position of the camera that took the START photograph and the horizontal direction in which it was pointing."""

ANSWER_RULES = """Answers
- Multiple-choice questions: give the option letter, for example "B".
- Numeric questions: give a number in meters, for example 2.4.
- If the information available to you does not allow you to determine the answer to a question, give "cannot_determine" for that question.
- Answer every question, using the question IDs (Q1, Q2, ...)."""

ANSWER_RULES_BLIND = """Answers
- Multiple-choice questions: give the option letter, for example "B".
- Numeric questions: give a number in meters, for example 2.4.
- Give your best guess for every question; "cannot_determine" is not accepted in this setting.
- Answer every question, using the question IDs (Q1, Q2, ...)."""

ANSWER_JSON = """{"action": "answer", "answers": {"Q1": "B", "Q2": 2.4, "Q3": "cannot_determine"}}"""
ANSWER_JSON_BLIND = """{"action": "answer", "answers": {"Q1": "B", "Q2": 2.4}}"""

NOTES_RULE = """The JSON object may also contain an optional "notes" field: a short string for your own observations (at most 300 characters; longer notes are cut off). Notes are not scored."""

SYSTEM_ACTIVE = """You are exploring an indoor scene in order to answer questions about it.

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
{notes_rule}"""

# active_v2 is the main setting; active_v1 (SYSTEM_ACTIVE) is kept.
# Changes: the Actions / Budget sections give the per-step scale (measured median on the v0.11 viewpoint graphs, env.step_scale in configs/eval.yaml) and state that turns can also fail;
# the Odometry section adds the cumulative pose relative to START and its coordinate frame. Everything else is the same as v1.
SYSTEM_ACTIVE_V2 = """You are exploring an indoor scene in order to answer questions about it.

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
{notes_rule}"""

POSE_RULE = """- With every observation you are also given your pose relative to START, in fixed coordinates defined by your starting position and starting viewing direction, measured on the floor plane: "forward" is the distance along the starting viewing direction (negative = behind the starting position), "left" is the distance to the left of it (negative = to the right), both in meters (rounded to 0.1 m); "heading" is the angle from the starting viewing direction to your current viewing direction, positive = turned to the left, between -180 and 180 degrees (rounded to 5 degrees). At START the pose is forward 0.0 m, left 0.0 m, heading 0 degrees."""
POSE_LINE = """Pose relative to START: forward {forward_m:.1f} m, left {left_m:.1f} m, heading {heading_deg} degrees."""

HISTORY_RULE_LAST_K_PIN = """- The START photograph is always shown. Of the other photographs, only the {k} most recent are shown as images; earlier ones remain listed by ID in the text record of your moves."""
HISTORY_RULE_LAST_K = """- Only the {k} most recent photographs are shown as images. Earlier photographs remain listed by ID in the text record of your moves."""
HISTORY_RULE_FULL = """- All photographs you have seen so far are shown."""

SYSTEM_STATIC = """You are answering questions about an indoor scene from photographs.

You are given {n_photos} photographs taken at different viewpoints in the same scene. The first photograph, marked START, shows your starting position and the direction you are facing. The other photographs are not in any particular order. You cannot move or request more photographs.

{conventions}

{answer_rules}

Reply format
Reply with a single JSON object and nothing else:
{answer_json}
{notes_rule}"""

# Text shared by the two baselines random_walk_40 and oracle_route (they differ only in photos and route).
SYSTEM_ROUTE = """You are answering questions about an indoor scene from photographs taken along a path through it.

The path starts at the viewpoint of the START photograph, which shows your starting position and the direction you are facing. The record lists every move along the path in order. Each viewpoint's photograph is given once, in the order in which the viewpoints were first visited; a later return to the same viewpoint refers back to its photograph by ID. You cannot move or request more photographs.

{pose_rule}

{conventions}

{answer_rules}

Reply format
Reply with a single JSON object and nothing else:
{answer_json}
{notes_rule}"""

POSE_RULE_ROUTE = """Each entry of the record gives the pose at that viewpoint relative to START, in fixed coordinates defined by the starting position and starting viewing direction, measured on the floor plane: "forward" is the distance along the starting viewing direction (negative = behind the starting position), "left" is the distance to the left of it (negative = to the right), both in meters (rounded to 0.1 m); "heading" is the angle from the starting viewing direction to the viewing direction of the photograph, positive = turned to the left, between -180 and 180 degrees (rounded to 5 degrees). Moves are named relative to the direction faced before the move: "forward", "backward", "left", "right" move to a nearby viewpoint in that direction; "turn_left", "turn_right" turn to a nearby viewpoint that faces further to the left or right."""

SYSTEM_BLIND = """You are answering questions about an indoor scene. No photographs of the scene are available in this setting; answer using general knowledge of how indoor spaces are typically arranged.

{conventions}

{answer_rules}

Reply format
Reply with a single JSON object and nothing else:
{answer_json}
{notes_rule}"""

QUESTION_MC = """{qid} (multiple choice) {text}
{options}"""
QUESTION_NUM = """{qid} (numeric, meters) {text}"""

USER_START_ACTIVE = """Questions
{questions}

You are at your starting viewpoint, {photo_id} (START).
{{image}}
Steps used: 0 of {budget}; {budget} remaining. Choose your next action."""

STEP_MOVED = """Step {t}: "{action}". {feedback} You are now at {photo_id}{revisit}.
{{image}}"""
STEP_CANNOT_MOVE = """Step {t}: "{action}". Cannot move: no viewpoint in that direction. You are still at {photo_id}; no new photograph."""
STEP_INVALID = """Step {t}: your reply was not a valid action, so one step was used and you did not move. You are still at {photo_id}. Reply with a single JSON object as described in the instructions."""
STEP_STATUS = """Steps used: {used} of {budget}; {left} remaining."""
STEP_MUST_ANSWER = """No steps remain. Reply with the answer action now."""
STEP_CHOOSE = """Choose your next action."""
IMAGE_OMITTED = """[{photo_id}: image no longer shown]"""

USER_START_ACTIVE_V2 = """Questions
{questions}

You are at your starting viewpoint, {photo_id} (START).
{{image}}
{pose}
Steps used: 0 of {budget}; {budget} remaining. Choose your next action."""

USER_ROUTE = """Questions
{questions}

Path record ({n_moves} moves, {n_photos} photographs):
{record}
Answer all questions."""
ROUTE_START = """START {photo_id}. {pose}
{{image}}"""
ROUTE_MOVE_NEW = """Move {t}: "{action}" to {photo_id}. {pose}
{{image}}"""
ROUTE_MOVE_REVISIT = """Move {t}: "{action}" back to {photo_id} (photograph shown above). {pose}"""

USER_STATIC = """Questions
{questions}

Photographs ({n_photos}):
{photos}
Answer all questions."""
PHOTO_LINE_START = """START {photo_id}
{{image}}"""
PHOTO_LINE = """{photo_id}
{{image}}"""

USER_BLIND = """Questions
{questions}

Answer all questions."""

ALL_TEMPLATES = {k: v for k, v in dict(globals()).items() if k.isupper() and isinstance(v, str)}


# ---------------------------------------------------------------- assembly

def format_questions(qs: list[dict]) -> str:
    out = []
    for q in qs:
        if q["answer_type"] == "numeric":
            out.append(QUESTION_NUM.format(qid=q["id"], text=q["text"]))
        else:
            out.append(QUESTION_MC.format(qid=q["id"], text=q["text"], options="\n".join(f"  {o}" for o in q["options"])))
    return "\n".join(out)


ACTIVE_VERSIONS = ("v1", "v2")
ROUTE_MODES = ("random_walk", "oracle_route")


def system_prompt(mode: str, *, budget: int | None = None, history: dict | None = None, n_photos: int | None = None,
                  version: str = "v1", scale: dict | None = None) -> str:
    """mode: active / blind / passive / oracle (= oracle_min) / oracle_min / random_walk / oracle_route.
    version only affects active: v1 is the pilot prompt; v2 adds the per-step scale and cumulative pose (scale = env.step_scale)."""
    common = {"conventions": CONVENTIONS, "answer_rules": ANSWER_RULES, "answer_json": ANSWER_JSON, "notes_rule": NOTES_RULE}
    if mode == "active":
        h = history or {"mode": "last_k", "k": 8, "pin_start": True}
        if h["mode"] == "full":
            rule = HISTORY_RULE_FULL
        else:
            rule = (HISTORY_RULE_LAST_K_PIN if h.get("pin_start") else HISTORY_RULE_LAST_K).format(k=h["k"])
        if version == "v2":
            sc = scale or {"move_m": 0.3, "turn_deg": 45}
            return SYSTEM_ACTIVE_V2.format(budget=budget, history_rule=rule, pose_rule=POSE_RULE,
                                           move_m=sc["move_m"], turn_deg=sc["turn_deg"], **common)
        return SYSTEM_ACTIVE.format(budget=budget, history_rule=rule, **common)
    if mode == "blind":  # Blind must give a best guess; cannot_determine is not accepted
        return SYSTEM_BLIND.format(**{**common, "answer_rules": ANSWER_RULES_BLIND, "answer_json": ANSWER_JSON_BLIND})
    if mode in ROUTE_MODES:
        return SYSTEM_ROUTE.format(pose_rule=POSE_RULE_ROUTE, **common)
    return SYSTEM_STATIC.format(n_photos=n_photos, **common)


def pose_text(pose: dict) -> str:
    return POSE_LINE.format(**pose)


def _parts(text: str, images: list[tuple[str, object]]) -> list:
    """Split text containing {image} placeholders into parts, inserting images in order."""
    chunks = text.split("{image}")
    assert len(chunks) == len(images) + 1, (len(chunks), len(images))
    parts = []
    for i, c in enumerate(chunks):
        if c:
            parts.append(("text", c))
        if i < len(images):
            parts.append(("image", *images[i]))
    return parts


def user_static(obs: dict) -> list:
    qs = format_questions(obs["questions"])
    if not obs["photos"]:
        return _parts(USER_BLIND.format(questions=qs), [])
    lines = [(PHOTO_LINE_START if p["start"] else PHOTO_LINE).format(photo_id=p["photo_id"]) for p in obs["photos"]]
    text = USER_STATIC.format(questions=qs, n_photos=len(obs["photos"]), photos="\n".join(lines))
    return _parts(text, [(p["photo_id"], p["image"]) for p in obs["photos"]])


def user_route(obs: dict) -> list:
    """random_walk / oracle_route: the start photo, then each step in order (action, photo reached, cumulative pose); each viewpoint's photo is given only at its first visit."""
    lines, images, seen = [ROUTE_START.format(photo_id=obs["photos"][0]["photo_id"], pose=pose_text(obs["start_pose"]))], [], set()
    img = {p["photo_id"]: p["image"] for p in obs["photos"]}
    images.append((obs["photos"][0]["photo_id"], obs["photos"][0]["image"]))
    seen.add(obs["photos"][0]["photo_id"])
    for r in obs["route"]:
        if r["photo_id"] in seen:
            lines.append(ROUTE_MOVE_REVISIT.format(t=r["t"], action=r["action"], photo_id=r["photo_id"], pose=pose_text(r["pose"])))
        else:
            lines.append(ROUTE_MOVE_NEW.format(t=r["t"], action=r["action"], photo_id=r["photo_id"], pose=pose_text(r["pose"])))
            images.append((r["photo_id"], img[r["photo_id"]]))
            seen.add(r["photo_id"])
    text = USER_ROUTE.format(questions=format_questions(obs["questions"]), n_moves=len(obs["route"]), n_photos=len(obs["photos"]),
                             record="\n".join(lines))
    return _parts(text, images)


def user_active_start(obs: dict, version: str = "v1") -> list:
    if version == "v2":
        text = USER_START_ACTIVE_V2.format(questions=format_questions(obs["questions"]), photo_id=obs["photo_id"], budget=obs["budget"],
                                           pose=pose_text(obs["pose"]))
    else:
        text = USER_START_ACTIVE.format(questions=format_questions(obs["questions"]), photo_id=obs["photo_id"], budget=obs["budget"])
    return _parts(text, [(obs["photo_id"], obs["image"])])


def user_active_step(t: int, action: str, obs: dict, budget: int, revisit: bool, invalid: bool, version: str = "v1") -> list:
    """User message after a step: result of the previous action, odometry (in the feedback text), new photo (when the move succeeded), steps used / remaining.
    v2 adds a line with the cumulative pose relative to START after the photo and before the step count (given at every observation, including cannot-move and invalid replies)."""
    if invalid:
        head, images = STEP_INVALID.format(t=t, photo_id=obs["photo_id"]), []
    elif obs["moved"]:
        head = STEP_MOVED.format(t=t, action=action, feedback=obs["feedback"], photo_id=obs["photo_id"],
                                 revisit=" (visited before)" if revisit else "")
        images = [(obs["photo_id"], obs["image"])]
    else:
        head, images = STEP_CANNOT_MOVE.format(t=t, action=action, photo_id=obs["photo_id"]), []
    tail = STEP_STATUS.format(used=obs["steps_used"], budget=budget, left=obs["steps_left"])
    tail += " " + (STEP_MUST_ANSWER if obs["must_answer"] else STEP_CHOOSE)
    if version == "v2":
        tail = pose_text(obs["pose"]) + "\n" + tail
    return _parts(head + "\n" + tail, images)


def apply_history(turns: list[dict], history: dict) -> list[dict]:
    """Image history rule: last_k keeps only the latest k photos as images and replaces earlier ones with text placeholders; full keeps all.
    pin_start: the first image (START, in turn 0) is always kept, plus the latest k, at most k + 1 images;
    turn 0 therefore never changes, so the system prompt, questions and START image form a fixed prefix that can hit the prompt cache.
    The text record (each step's action, odometry, "cannot move", step count) is always kept in full."""
    if history["mode"] == "full":
        return turns
    seen, keep = 0, set()
    first = next(((ti, pi) for ti, t in enumerate(turns) for pi, p in enumerate(t.get("parts", [])) if p[0] == "image"), None)
    if history.get("pin_start") and first is not None:
        keep.add(first)
    for ti in range(len(turns) - 1, -1, -1):
        for pi in range(len(turns[ti].get("parts", [])) - 1, -1, -1):
            if turns[ti]["parts"][pi][0] == "image" and (ti, pi) not in keep:
                if seen < history["k"]:
                    keep.add((ti, pi))
                seen += 1
    out = []
    for ti, turn in enumerate(turns):
        if "parts" not in turn:
            out.append(turn)
            continue
        parts = [p if p[0] != "image" or (ti, pi) in keep else ("text", IMAGE_OMITTED.format(photo_id=p[1]))
                 for pi, p in enumerate(turn["parts"])]
        out.append({**turn, "parts": parts})
    return out
