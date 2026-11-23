"""API evaluation of closed-source models.

- Registry configs/models_api.yaml: one entry per model (provider, official model ID, name of the env var holding the key, prices,
  reasoning effort, images, JSON, caching, concurrency). registry.merge adds the registry models to data.ecfg["models"]; from then on
  llm_run.run_llm takes the same path as for open-weight models (same trajectories, per-item records, env.json, seal).
- Adapters: openai_api.OpenAIAPIClient (subclass of models.OpenAIClient) and anthropic_api.AnthropicClient. The interface and shared
  behaviour (real-call switch, full message logging, retries on empty replies and server errors, cost reservation) live in
  base.ApiClientMixin. To add another provider: write an adapter and add it to ADAPTERS.
- Keys are read only from the env var named by the registry's key_env, go only into request headers, and are never written to any log or file.
- mock.py: fake transport layer; all of the real adapter code runs, only the HTTP step is replaced. scripts/run_api_eval.sh uses
  mock-openai / mock-anthropic to exercise the whole pipeline.
"""
from __future__ import annotations


def make_api_client(mcfg, llm_cfg, cost=None, log_path=None, **kw):
    from .anthropic_api import AnthropicClient
    from .openai_api import OpenAIAPIClient
    adapters = {"openai": OpenAIAPIClient, "anthropic": AnthropicClient}
    if mcfg["provider"] not in adapters:
        raise ValueError(f"unknown provider {mcfg['provider']}")
    if mcfg.get("mock") and kw.get("transport") is None:
        from .mock import make_transport
        kw = {**kw, "transport": make_transport(mcfg), "sleep": kw.get("sleep") or (lambda s: None)}
    return adapters[mcfg["provider"]](mcfg, llm_cfg, cost, log_path, **kw)
