"""Model interface probe (before comparing strong and weak models, confirm that the model accepts image input and reasoning_effort).

    python -m actspace_eval.probe --models gpt-6.1-terra gpt-6-luna gpt-6.1-sol --effort high --out <dir>

3 calls per model, with parameters identical to the real runs (Chat Completions, reasoning_effort, max_completion_tokens, JSON mode,
prompt_cache_key, image detail: high); the content is the first step of active_v2 on smoke group g_000000:
  A. text only (START image removed); B. with the START image; C. same as B (same prompt_cache_key), to check cache hits.
Records: whether accepted (raw HTTP error), finish_reason, whether the reply parses as a valid action, reasoning tokens, tokens per image (input difference B − A),
cache hits, and the raw usage object returned by the API (whether there are billing items such as cache writes). Cost cap per model: $0.25.
The key is read only from the environment variable OPENAI_API_KEY (models.OpenAIClient); this module never reads, writes or prints it.
Outputs: <dir>/<model>.jsonl (call log, same format as calls.jsonl) and <dir>/probe.json.
Exit code: 0 when all 3 calls of every model were accepted and all replies are valid actions, otherwise 1 (follow-up sbatch jobs can use --dependency=afterok).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import prompts as P
from .cost import CostTracker
from .data import MOVE_ACTIONS, FrozenData
from .env import ActSpaceEnv
from .llm_agent import parse_reply
from .llm_run import with_effort
from .models import ModelError, RealCallsDisabled, make_client

GROUP = "arkit_g_000000"   # group IDs carry a dataset prefix since merged_v1.0
CAP_USD = 0.25


def probe_messages(data, gid=GROUP):
    """(system prompt, text-only conversation, conversation with the START image) for the first step of active_v2."""
    ec = data.ecfg
    env = ActSpaceEnv(data, budget=ec["env"]["budget"], render=True, mode="active", cumulative_pose=True)
    obs = env.reset(gid)
    system = P.system_prompt("active", budget=ec["env"]["budget"], history=ec["llm"]["history"], version="v2",
                             scale=data.step_scale(gid))
    parts = P.user_active_start(obs, "v2")
    return system, [{"role": "user", "parts": [p for p in parts if p[0] == "text"]}], [{"role": "user", "parts": parts}]


def probe_model(data, model_key, effort, out: Path, messages, client_kwargs=None) -> dict:
    mcfg = with_effort(data.ecfg["models"][model_key], effort)
    res = {"model": model_key, "model_name": mcfg["model"], "reasoning_effort": mcfg.get("reasoning", {}).get("effort"),
           "params": None, "calls": []}
    log = out / f"{model_key}.jsonl"
    try:
        client = make_client(mcfg, data.ecfg["llm"], CostTracker(mcfg["prices_usd_per_mtok"], CAP_USD), log, **(client_kwargs or {}))
    except RealCallsDisabled as e:
        res["error"] = str(e)
        return res
    client.cache_key = f"actspace-probe-{model_key}"
    res["params"] = client.params()
    system, text_turns, image_turns = messages
    for name, turns in (("A_text", text_turns), ("B_image", image_turns), ("C_image_repeat", image_turns)):
        rec = {"call": name}
        try:
            comp = client.complete(system, turns, tag={"probe": name})
        except ModelError as e:
            rec.update(accepted=False, status=e.status, error=str(e)[:500])
            res["calls"].append(rec)
            continue
        act, _ = parse_reply(comp.text)
        rec.update(accepted=True, finish_reason=comp.finish_reason, action=act, usage=comp.usage,
                   usage_raw=comp.raw.get("usage"), cost_usd=round(comp.cost_usd, 6), reply=comp.text[:300])
        res["calls"].append(rec)
    c = {r["call"]: r for r in res["calls"]}
    if c.get("A_text", {}).get("accepted") and c.get("B_image", {}).get("accepted"):
        res["tokens_per_image"] = c["B_image"]["usage"]["input_tokens"] - c["A_text"]["usage"]["input_tokens"]
    ok = [r for r in res["calls"] if r.get("accepted")]
    res["all_accepted"] = len(ok) == 3
    res["reasoning_used"] = any(r["usage"]["reasoning_tokens"] > 0 for r in ok)
    res["valid_actions"] = sum(r["action"] in (*MOVE_ACTIONS, "answer") for r in ok)
    res["cached_on_repeat"] = c.get("C_image_repeat", {}).get("usage", {}).get("cached_input_tokens")
    res["spent_usd"] = round(sum(r.get("cost_usd", 0) for r in ok), 6)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--effort", choices=["low", "medium", "high"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = FrozenData()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    msgs = probe_messages(data)
    results = [probe_model(data, m, args.effort, out, msgs) for m in args.models]
    (out / "probe.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    for r in results:
        print(f"== {r['model']} ({r['model_name']}, effort={r['reasoning_effort']}) params={json.dumps(r['params'])}")
        if "error" in r:
            print("   not called:", r["error"])
            continue
        for c in r["calls"]:
            if c["accepted"]:
                print(f"   {c['call']}: ok finish={c['finish_reason']} action={c['action']} usage={c['usage']} raw_usage={json.dumps(c['usage_raw'])}")
            else:
                print(f"   {c['call']}: REJECTED status={c['status']} {c['error']}")
        print(f"   all_accepted={r['all_accepted']} reasoning_used={r['reasoning_used']} valid_actions={r['valid_actions']}/3 "
              f"tokens_per_image={r.get('tokens_per_image')} cached_on_repeat={r['cached_on_repeat']} spent=${r['spent_usd']}")
    sys.exit(0 if all(r.get("all_accepted") and r.get("valid_actions") == 3 for r in results) else 1)


if __name__ == "__main__":
    main()
