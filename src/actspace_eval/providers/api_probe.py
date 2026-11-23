"""Closed-model probe: a few real requests before the main run, with the same parameters as the main run; the content is the first
step of active_v2 on arkit_g_000000.

    python -m actspace_eval.providers.api_eval probe --model <short name> --prefix <prefix> --cap <USD>

A is text only; B carries the START image; C is identical to B (to check cache reads). On fatal errors such as HTTP 400 / 401 / 403
it stops immediately and does not send the remaining calls.
All of the following must pass (otherwise the one-click script stops):
- images: B is accepted, and the input difference B - A (tokens per image) is >= 100;
- thinking: all three calls are accepted (the request carries the thinking settings);
- json: all three replies parse into one JSON object containing a valid action;
- usage: the raw usage contains the fields needed for billing.
Reported only: whether thinking tokens or summaries appear (under adaptive, a simple step may involve no thinking), cache hits,
and the returned model ID.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..cost import CostCapExceeded
from ..llm_agent import parse_detail
from ..models import ModelError, RealCallsDisabled, make_client
from ..probe import probe_messages
from .base import ApiCostTracker

REQUIRED_USAGE = {"openai": ("prompt_tokens", "completion_tokens", "prompt_tokens_details", "completion_tokens_details"),
                  "anthropic": ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
MIN_TOKENS_PER_IMAGE = 100


def _call(client, name, system, turns) -> dict:
    rec = {"call": name}
    try:
        comp = client.complete(system, turns, tag={"probe": name})
    except ModelError as e:
        return {**rec, "accepted": False, "status": e.status, "error": str(e)[:500], "fatal": bool(getattr(e, "stop_status", None)),
                "stop_status": getattr(e, "stop_status", None)}
    except CostCapExceeded as e:
        return {**rec, "accepted": False, "status": None, "error": str(e)}
    return {**rec, "accepted": True, "finish_reason": comp.finish_reason, "parse": parse_detail(comp.text), "usage": comp.usage,
            "usage_raw": comp.raw.get("usage"), "model_returned": comp.raw.get("model"), "cost_usd": round(comp.cost_usd, 6),
            "reasoning_chars": comp.extra.get("reasoning_chars"), "reply": comp.text[:300]}


def probe_model(data, short: str, out: Path, cap: float, client_kwargs=None) -> dict:
    ec = data.ecfg
    mcfg = ec["models"][short]
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{short}.jsonl"
    res = {"model": short, "model_id": mcfg["model"], "provider": mcfg["provider"], "calls": [], "passed": False}
    tracker = ApiCostTracker(mcfg["prices_usd_per_mtok"], cap)
    try:
        client = make_client(mcfg, ec["llm"], tracker, log, **(client_kwargs or {}))
    except RealCallsDisabled as e:
        res["error"] = str(e)
        return res
    client.cache_key = f"actspace-probe-{short}"
    client.force_cache = True
    res["params"] = client.params()
    system, text_turns, image_turns = probe_messages(data)
    for name, turns in (("A_text", text_turns), ("B_image", image_turns), ("C_image_repeat", image_turns)):
        res["calls"].append(_call(client, name, system, turns))
        if res["calls"][-1].get("fatal"):
            res["fatal"], res["stop_status"] = True, res["calls"][-1]["stop_status"]
            break
    c = {r["call"]: r for r in res["calls"]}
    ok = [r for r in res["calls"] if r["accepted"]]
    a, b = c.get("A_text", {}), c.get("B_image", {})
    tpi = b["usage"]["input_tokens"] - a["usage"]["input_tokens"] if a.get("accepted") and b.get("accepted") else None
    raw = ok[-1]["usage_raw"] if ok else {}
    res.update(tokens_per_image=tpi, spent_usd=round(sum(r["cost_usd"] for r in ok), 6),
               checks={"images": tpi is not None and tpi >= MIN_TOKENS_PER_IMAGE, "thinking": len(ok) == 3,
                       "json": len(ok) == 3 and all(r["parse"]["json"] and r["parse"]["error"] is None for r in ok),
                       "usage": all(k in (raw or {}) for k in REQUIRED_USAGE[mcfg["provider"]])},
               reasoning_tokens=[r["usage"]["reasoning_tokens"] for r in ok], reasoning_chars=[r["reasoning_chars"] for r in ok],
               cache_read_on_repeat=c.get("C_image_repeat", {}).get("usage", {}).get("cached_input_tokens"),
               cache_write_on_first=b.get("usage", {}).get("cache_write_tokens"),
               model_returned=[r["model_returned"] for r in ok], output_tokens=[r["usage"]["output_tokens"] for r in ok])
    res["passed"] = all(res["checks"].values())
    return res


def report(r: dict) -> str:
    lines = [f"== probe {r['model']} ({r['model_id']})  params={json.dumps(r.get('params'))}"]
    if "error" in r:
        return "\n".join(lines + [f"   not called: {r['error']}"])
    for c in r["calls"]:
        if c["accepted"]:
            lines.append(f"   {c['call']}: ok finish={c['finish_reason']} parse={c['parse'].get('action')} usage={json.dumps(c['usage'])}")
        else:
            lines.append(f"   {c['call']}: REJECTED status={c['status']} {c['error']}")
    lines.append(f"   checks={r['checks']} tokens_per_image={r['tokens_per_image']} reasoning_tokens={r['reasoning_tokens']} "
                 f"reasoning_chars={r['reasoning_chars']} cache_write={r['cache_write_on_first']} cache_read={r['cache_read_on_repeat']} "
                 f"model_returned={r['model_returned']} spent=${r['spent_usd']}")
    return "\n".join(lines)
