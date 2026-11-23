"""Reads the closed-model registry configs/models_api.yaml and converts it into the model configs used by llm_run / models. Reads no keys."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "configs" / "models_api.yaml"


def load_registry(path=None) -> dict:
    with open(path or REGISTRY) as f:
        return yaml.safe_load(f)


def entries(reg: dict) -> dict:
    """{short name: registry entry}, including mock_models."""
    return {**reg["models"], **(reg.get("mock_models") or {})}


def results_root(reg: dict) -> Path:
    """Results directory for closed models: env var API_OUT_ROOT (for tests) > results in configs/paths.yaml (<results>/api) >
    the registry's results_root (relative paths are relative to the repository root)."""
    from actspace_build.common import abs_path, site_paths
    if os.environ.get("API_OUT_ROOT"):
        return Path(os.environ["API_OUT_ROOT"])
    sp = site_paths()
    return sp["results"] / "api" if "results" in sp else abs_path(reg["results_root"])


def to_mcfg(short: str, e: dict) -> dict:
    """Registry entry -> model config for llm_run / the adapters."""
    p, r, img = e["prices"], e["reasoning"], e["image"]
    if e["json"]["mode"] == "schema" and e["provider"] != "anthropic":
        raise ValueError(f"{short}: json.mode schema is only for anthropic")
    return {"api_registry": True, "short": short, "provider": e["provider"], "model": e["model_id"], "base_url": e["base_url"],
            "key_env": e["key_env"], "api_version": e.get("api_version"), "temperature": None,
            "max_output_tokens": e["max_output_tokens"], "max_tokens_param": e.get("max_tokens_param", "max_completion_tokens"),
            "reasoning": {"enabled": True, "effort": r["target"], "approved": r["approved"], "display": (r["request"].get("thinking") or {}).get("display")},
            "thinking": {"supported": True, "enabled": True, "returns_thinking": r["returns_thinking"]},
            "json_mode": e["json"]["mode"] == "json_object", "json_schema": e["json"]["mode"] == "schema", "image_detail": img.get("detail", "auto"), "tokens_per_image": img["tokens_per_image"],
            "prompt_cache_key": e["prompt_cache"]["mode"] == "auto", "prompt_cache_mode": e["prompt_cache"]["mode"],
            "logprobs_top": e.get("logprobs_top"), "stream": bool(e.get("stream")), "request_timeout_s": e["request_timeout_s"],
            "prices_usd_per_mtok": {"input": p["input"], "cached_input": p["cache_read"], "cache_write": p["cache_write"],
                                    "cache_write_12h": p["cache_write_long"], "output": p["output"]},
            "prices_confirmed": e["prices_confirmed"], "est_output_tokens": e["est_output_tokens"], "mock": e.get("mock")}


def merge(data, reg=None) -> dict:
    """Adds the registry models to data.ecfg["models"] (in memory; configs/eval.yaml is not changed). Returns the registry."""
    reg = reg or load_registry()
    data.ecfg = {**data.ecfg, "models": {**data.ecfg["models"], **{k: to_mcfg(k, v) for k, v in entries(reg).items()}}}
    return reg


def env_info(short: str, e: dict) -> dict:
    """Model section of env.json. For the key, only the env var name is recorded."""
    return {"provider": e["provider"], "short": short, "model_id": e["model_id"], "base_url": e["base_url"], "api_version": e.get("api_version"),
            "key_env": e["key_env"], "reasoning": e["reasoning"], "max_tokens": e["max_output_tokens"], "prices_usd_per_mtok": e["prices"],
            "prices_confirmed": e["prices_confirmed"], "json": e["json"], "prompt_cache": e["prompt_cache"], "image": e["image"],
            "concurrency": e["concurrency"], "mock": e.get("mock"), "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest()}


def assert_protocol(ec: dict, version: str):
    """The fixed protocol must not change: checked before running; refuses to run if the config has been altered."""
    env, llm = ec["env"], ec["llm"]
    checks = {"data version merged_v1.0": version == "merged_v1.0", "budget 40 steps": env["budget"] == 40,
              "K = 8 + START": llm["history"] == {"mode": "last_k", "k": 8, "pin_start": True},
              "image long side 640": env["max_image_long_side"] == 640, "tiny 32 groups": len(llm["tiny_groups"]) == 32,
              "budget_infeasible 7 groups": len(env["budget_infeasible"]) == 7, "reasoning_effort = high": llm["reasoning_effort"] == "high"}
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise RuntimeError(f"config does not match the fixed protocol: {bad}")
