"""Cost control and estimation (docs/evaluation_protocol.md).

- CostTracker: accumulates cost in real time at the configured prices (uncached input, cached input, cache writes and output priced separately; reasoning tokens are included in output).
  Cache writes: cache_write_tokens are part of the input tokens and are billed at the cache_write price (30-minute tier, 1.25× input)
  rather than as uncached input; the 12-hour tier (cache_write_12h_tokens) is billed at the cache_write_12h price. Without a configured write price they are billed as uncached input
  (as before). The pre-call check assumes "all input is written" (conservative).
  Pre-call check against the estimate: no call is made if spent + this estimate > cost_cap_usd; after a call, stop immediately if spent > cost_cap_usd.
  Both cases raise CostCapExceeded; the runner saves completed episodes and exits (resumable; on resume the spent amount is restored from the call log).
- estimate: assembles messages with the real prompt templates (without loading images) and estimates token counts and cost for the tiny set (or any question groups) × settings × budgets.
  Text at about 4 characters per token, images at the model's configured tokens_per_image, output at est_output_tokens.
  Cache hits are estimated via cached_prefix_tokens (in Active, the message prefix identical to the previous step's request; no hit below 1024 tokens).
  Cache writes: conservative; in every call the uncached input is counted entirely as writes whenever the whole request reaches 1024 tokens (WRITE_MIN_TOKENS).
"""
from __future__ import annotations

import json
from pathlib import Path

from . import prompts as P
from .data import dataset_of_scene, resolve_step_scale
from .llm_agent import setting_env, static_messages
from .models import cached_prefix_tokens, estimate_input_tokens


WRITE_MIN_TOKENS = 1024   # shortest cacheable request (OpenAI prompt caching docs)
TRUNCATED_FINISH = ("length", "max_tokens", "model_context_window_exceeded")   # finish / stop reasons for hitting the output limit


def est_writes(input_tokens: int, cached: int) -> int:
    """Estimate the cache writes of one call: when the request reaches 1024 tokens, all uncached input counts as writes (conservative)."""
    return input_tokens - cached if input_tokens >= WRITE_MIN_TOKENS else 0


class CostCapExceeded(RuntimeError):
    pass


class CostTracker:
    def __init__(self, prices: dict, cap_usd: float, spent: float = 0.0):
        self.prices, self.cap, self.spent = prices, float(cap_usd), float(spent)

    def cost_of(self, usage: dict) -> float:
        p = self.prices
        cached = usage.get("cached_input_tokens", 0)
        w = usage.get("cache_write_tokens") or 0
        w12 = usage.get("cache_write_12h_tokens") or 0
        return ((usage.get("input_tokens", 0) - cached - w) * p["input"] + cached * p["cached_input"]
                + (w - w12) * p.get("cache_write", p["input"]) + w12 * p.get("cache_write_12h", 2.5 * p["input"])
                + usage.get("output_tokens", 0) * p["output"]) / 1e6

    def add(self, usage: dict) -> float:
        c = self.cost_of(usage)
        self.spent += c
        return c

    def check_before(self, est_input_tokens: int, max_output_tokens: int):
        est = self.cost_of({"input_tokens": est_input_tokens, "cache_write_tokens": est_input_tokens, "output_tokens": max_output_tokens})
        if self.spent + est > self.cap:
            raise CostCapExceeded(f"spent ${self.spent:.4f} + at most ${est:.4f} for this call would exceed the cap ${self.cap:.2f}; no further calls")

    def check_after(self):
        if self.spent > self.cap:
            raise CostCapExceeded(f"spent ${self.spent:.4f}, exceeding the cap ${self.cap:.2f}; stopping now")

    @staticmethod
    def usage_in_log(path: Path) -> dict:
        """Total token usage in the call log (input includes cached and cache_write; output includes reasoning) and call / retry counts."""
        tot = {"calls": 0, "errors": 0, "input_tokens": 0, "cached_input_tokens": 0, "cache_write_tokens": 0,
               "cache_write_12h_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "possibly_billed_errors": 0, "truncated_calls": 0}
        if not Path(path).exists():
            return tot
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("type") == "error":
                    tot["errors"] += 1
                    tot["possibly_billed_errors"] += bool(r.get("possibly_billed"))
                if r.get("type") == "call":
                    tot["calls"] += 1
                    tot["truncated_calls"] += r.get("finish_reason") in TRUNCATED_FINISH
                elif not (r.get("type") == "error" and r.get("usage")):   # failed but billed attempts also count toward token usage
                    continue
                u = call_usage(r)
                for k in ("input_tokens", "cached_input_tokens", "cache_write_tokens", "cache_write_12h_tokens", "output_tokens",
                          "reasoning_tokens"):
                    tot[k] += u.get(k) or 0
        return tot

    @staticmethod
    def spent_in_log(path: Path) -> float:
        """Resume: restore the spent cost from the call log."""
        if not Path(path).exists():
            return 0.0
        total = 0.0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("type") in ("call", "error"):   # error records carry cost_usd only for failed but billed attempts
                    total += r.get("cost_usd") or 0.0
        return total


def call_usage(rec: dict) -> dict:
    """Usage of one call record. Older code did not write cache writes into usage: fill them in from usage_raw when present (same fields as models.OpenAIClient)."""
    u = dict(rec.get("usage") or {})
    if "cache_write_tokens" not in u:
        ptd = (rec.get("usage_raw") or {}).get("prompt_tokens_details") or {}
        u["cache_write_tokens"] = ptd.get("cache_write_tokens") or 0
        u["cache_write_12h_tokens"] = (ptd.get("cache_write_tokens_details") or {}).get("ttl_12h_tokens") or 0
    return u


def recost_log(path: Path, prices: dict) -> dict:
    """Recompute a run's cost including cache writes.
    logged: sum of cost_usd in the call log (old records exclude writes); cost: recomputed record by record via call_usage (writes count as 0 for old records without usage_raw);
    upper: upper bound; for records without raw usage, all uncached input of requests reaching 1024 tokens is billed as 30-minute-tier writes (est_writes; records with raw usage use the actual values);
    worst: regardless of raw usage, all uncached input of requests reaching 1024 tokens is billed as writes (in practice most API calls do not return cache_write_tokens);
    raw_calls / calls: number of calls with raw usage / total calls."""
    t = CostTracker(prices, float("inf"))
    out = {"logged": 0.0, "cost": 0.0, "upper": 0.0, "worst": 0.0, "calls": 0, "raw_calls": 0}
    if not Path(path).exists():
        return out
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") != "call":
                continue
            u = call_usage(r)
            out["calls"] += 1
            out["logged"] += r.get("cost_usd") or 0.0
            c = t.cost_of(u)
            out["cost"] += c
            w = t.cost_of({**u, "cache_write_tokens": est_writes(u.get("input_tokens", 0), u.get("cached_input_tokens", 0))})
            out["worst"] += max(c, w)
            if r.get("usage_raw") is not None or "cache_write_tokens" in (r.get("usage") or {}):
                out["raw_calls"] += 1
                out["upper"] += c
            else:
                out["upper"] += w
    return out


# ---------------------------------------------------------------- estimation

FEEDBACK_EXAMPLE = "Moved forward about 0.6 m and left about 0.1 m; turned left about 40 degrees."
POSE_EXAMPLE = {"forward_m": -1.2, "left_m": 2.3, "heading_deg": -135}


def estimate_episode(env_factory, group_id: str, setting: str, budget: int, mcfg: dict, history: dict, active_steps: int,
                     run_seed: int = 0, scale: dict | None = None) -> dict:
    """Call count and token estimate for one episode. Active assumes active_steps steps, each a successful move with a new photo (conservative), then an answer.
    setting is a setting name from llm_agent.SETTINGS; env_factory(setting, budget, run_seed) returns an environment (the photo count of random_walk depends on the seed)."""
    tpi, eo = mcfg["tokens_per_image"], mcfg["est_output_tokens"]
    mode, version, _ = setting_env(setting)
    env = env_factory(setting, budget, run_seed)
    obs = env.reset(group_id)
    scale = resolve_step_scale(scale, dataset_of_scene(env.data.groups[group_id]["scene_id"]))   # may be given per dataset
    nq = len(obs["questions"])
    ans_out = eo["answer_base"] + eo["answer_per_question"] * nq + eo.get("reasoning_per_call", 0)
    if mode != "active":
        system, turns = static_messages(mode, obs)
        inp = estimate_input_tokens(system, turns, tpi)
        return {"calls": 1, "images": len(obs["photos"]), "input_tokens": inp, "cached_input_tokens": 0,
                "cache_write_tokens": est_writes(inp, 0), "output_tokens": ans_out}
    system = P.system_prompt("active", budget=budget, history=history, version=version, scale=scale)
    turns = [{"role": "user", "parts": P.user_active_start(obs, version)}]
    calls = inp = out = imgs = cached = writes = 0
    prev = None
    for t in range(active_steps + 1):
        if t > 0:
            turns.append({"role": "assistant", "text": '{"action": "forward"}'})
            fake = {"moved": True, "feedback": FEEDBACK_EXAMPLE, "photo_id": f"photo_{t:06d}", "image": None,
                    "steps_used": t, "steps_left": budget - t, "must_answer": t >= budget, "pose": POSE_EXAMPLE}
            turns.append({"role": "user", "parts": P.user_active_step(t, "forward", fake, budget, False, False, version)})
        shown = P.apply_history(turns, history)
        calls += 1
        imgs += sum(p[0] == "image" for tt in shown for p in tt.get("parts", []))
        i, c = estimate_input_tokens(system, shown, tpi), cached_prefix_tokens(prev, (system, shown), tpi)
        inp, cached, writes = inp + i, cached + c, writes + est_writes(i, c)
        prev = (system, shown)
        out += (eo["action"] + eo.get("reasoning_per_call", 0)) if t < active_steps else ans_out
    return {"calls": calls, "images": imgs, "input_tokens": inp, "cached_input_tokens": cached, "cache_write_tokens": writes,
            "output_tokens": out}


def estimate(env_factory, group_ids, plan: list[tuple], mcfg: dict, history: dict, scale: dict | None = None) -> list[dict]:
    """plan: [(setting, budget, assumed Active steps[, list of run_seed])]. Returns total tokens and cost per item (at the model's configured prices):
    cost_usd prices the estimated cache hits and writes; cost_no_cache_usd ignores the cache discount (inputs that would hit are also billed as writes). With several run_seed values (random_walk, 3 seeds per group) they are summed."""
    tracker = CostTracker(mcfg["prices_usd_per_mtok"], cap_usd=float("inf"))
    rows = []
    for item in plan:
        mode, budget, steps = item[:3]
        seeds = item[3] if len(item) > 3 else (0,)
        tot = {"calls": 0, "images": 0, "input_tokens": 0, "cached_input_tokens": 0, "cache_write_tokens": 0, "output_tokens": 0}
        for seed in seeds:
            for gid in group_ids:
                e = estimate_episode(env_factory, gid, mode, budget, mcfg, history, steps, seed, scale)
                for k in tot:
                    tot[k] += e[k]
        tot["cost_usd"] = tracker.cost_of(tot)
        tot["cost_no_cache_usd"] = tracker.cost_of({**tot, "cached_input_tokens": 0,
                                                    "cache_write_tokens": tot["cache_write_tokens"] + tot["cached_input_tokens"]})
        act = setting_env(mode)[0] == "active"
        rows.append({"mode": mode, "budget": budget, "active_steps": steps if act else None, "seeds": len(seeds),
                     "history": history if act else None, "groups": len(group_ids), **tot})
    return rows


def format_estimate(rows, title="Cost estimate") -> str:
    lines = [title, f"{'setting':<16}{'groups':>7}{'calls':>7}{'photos':>8}{'input tokens':>14}{'cached':>12}{'writes':>12}{'output tokens':>14}{'cost USD':>11}{'no cache':>9}"]
    for r in rows:
        name = r["mode"] + (f"-{r['budget']}" if r["mode"].startswith(("active", "passive", "random_walk")) else "")
        if r.get("seeds", 1) > 1:
            name += f"×{r['seeds']} seeds"
        if r["active_steps"] is not None:
            name += f" ({r['active_steps']} steps, {r['history']['mode']}{r['history'].get('k', '') if r['history']['mode'] == 'last_k' else ''})"
        lines.append(f"{name:<16}{r['groups']:>7}{r['calls']:>7}{r['images']:>8}{r['input_tokens']:>14,}{r['cached_input_tokens']:>12,}{r.get('cache_write_tokens', 0):>12,}"
                     f"{r['output_tokens']:>14,}{r['cost_usd']:>11.2f}{r['cost_no_cache_usd']:>9.2f}")
    return "\n".join(lines)
