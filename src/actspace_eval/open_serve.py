"""Launch arguments and server info for the open-model inference servers.

    python -m actspace_eval.open_serve args <model> <port>      # print the vllm serve arguments (one per line)
    python -m actspace_eval.open_serve wait <base_url> [secs]   # wait until the server is ready (/health and /v1/models)
    python -m actspace_eval.open_serve throughput <run dir…>    # per-run call latency, time to first token, token throughput, reasoning length

vllm serve: bf16 (override with vllm.dtype; LLaVA-OneVision uses its stored precision float16), no quantisation; tensor parallelism,
max_model_len, reasoning parser and trust_remote_code come from models.<model>.vllm in configs/eval.yaml;
--limit-mm-per-prompt allows at most open_models.limit_images_per_prompt (48) images per request (no video / audio input);
--generation-config auto (the model directory's generation_config.json supplies default sampling parameters; sampling parameters
given explicitly in the request take precedence, and the two agree in this evaluation); listens on 127.0.0.1 only.
--mm-processor-cache-gb 0 disables the multimodal processor cache: a request rejected for context length leaves the sender-side cache
out of sync with the engine, and the engine then reports "Expected a cached item" the next time the same image is used.
vllm.hf_overrides (max_dynamic_patch for the InternVL family; changes the in-memory config only); vllm.chat_template (template adaptation).
--chat-template-content-format openai is set explicitly: the Qwen3.5 / 3.6 templates process content with macros, vLLM's auto-detection
classifies them as string and moves all images of a message in front of the text, losing the correspondence between photo IDs and images;
with the openai format images stay in place, in the same order as in the closed-model requests.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .data import ROOT, load_eval_config


def vllm_args(ec: dict, key: str, port: int) -> list[str]:
    m = ec["models"][key]
    v = m["vllm"]
    om = ec["open_models"]
    if v is None:
        raise ValueError(f"{key} does not use vLLM (server: {m.get('server')})")
    mm = {"image": int(om["limit_images_per_prompt"]), "video": 0}
    if "gemma" in key.lower():
        mm["audio"] = 0
    a = ["serve", m["path"], "--served-model-name", m["model"], "--host", "127.0.0.1", "--port", str(port),
         "--dtype", str(v.get("dtype", "bfloat16")), "--tensor-parallel-size", str(v["tp"]),
         "--max-model-len", str(min(int(v["max_model_len"]), int(om["max_model_len_cap"]))),
         "--gpu-memory-utilization", str(v.get("gpu_memory_utilization", om["gpu_memory_utilization"])),
         "--limit-mm-per-prompt", json.dumps(mm), "--max-num-seqs", str(v.get("max_num_seqs", 32)),
         "--max-logprobs", "20", "--seed", "0", "--generation-config", "auto",
         "--mm-processor-cache-gb", str(om.get("mm_processor_cache_gb", 0)),
         "--chat-template-content-format", "openai"]
    if v.get("hf_overrides"):
        a += ["--hf-overrides", json.dumps(v["hf_overrides"])]
    if v.get("chat_template"):
        a += ["--chat-template", str(ROOT / v["chat_template"])]
    if v.get("trust_remote_code"):
        a.append("--trust-remote-code")
    if v.get("reasoning_parser"):
        a += ["--reasoning-parser", v["reasoning_parser"]]
    for x in v.get("extra_args") or []:
        a.append(str(x))
    return a


PLUGIN_DIR = ROOT / "vllm_plugins"


def server_env(ec: dict, key: str, env: dict) -> dict:
    """Environment of the server process: when the model config has vllm.plugins: true, vllm_plugins/ is added to PYTHONPATH
    and vLLM loads the plugins via entry points (LLaVA-OneVision: base resolution only). Other models load no plugins."""
    v = ec["models"][key].get("vllm") or {}
    if v.get("plugins"):
        env = {**env, "PYTHONPATH": f"{PLUGIN_DIR}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(PLUGIN_DIR)}
    return {**env, **{k: str(x) for k, x in (v.get("env") or {}).items()}}


def wait(base_url: str, timeout_s: float = 3600) -> dict:
    import requests
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if requests.get(base_url + "/health", timeout=5).status_code == 200:
                models = requests.get(base_url + "/v1/models", timeout=10).json()
                return {"ready_after_s": round(time.time() - t0, 1), "models": models}
        except requests.exceptions.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError(f"{base_url} not ready within {timeout_s} s")


def throughput(run_dir: Path) -> dict:
    """Call statistics of one run (reads calls.jsonl only): number of calls and errors, mean latency per call, time to first token,
    output token throughput (output tokens / latency per call, and total output tokens / run wall-clock seconds), reasoning length
    (characters and tokens), finish_reason distribution, number of parse failures."""
    import numpy as np
    calls, errors = [], []
    p = Path(run_dir) / "calls.jsonl"
    if not p.exists():
        return {"calls": 0}
    for line in open(p):
        r = json.loads(line)
        (calls if r.get("type") == "call" else errors).append(r)
    if not calls:
        return {"calls": 0, "errors": len(errors)}
    lat = np.array([c["latency_s"] for c in calls], float)
    out = np.array([c["usage"].get("output_tokens", 0) for c in calls], float)
    inp = np.array([c["usage"].get("input_tokens", 0) for c in calls], float)
    ttft = [c.get("ttft_s") for c in calls if c.get("ttft_s") is not None]
    rch = [c.get("reasoning_chars", 0) for c in calls]
    rtok = [c.get("reasoning_tokens") for c in calls if c.get("reasoning_tokens") is not None]
    st = json.loads((Path(run_dir) / "status.json").read_text()) if (Path(run_dir) / "status.json").exists() else {}
    fin = {}
    for c in calls:
        fin[str(c.get("finish_reason"))] = fin.get(str(c.get("finish_reason")), 0) + 1
    parse = [c.get("parse") or {} for c in calls]
    q = lambda xs, k: float(np.percentile(xs, k)) if len(xs) else None
    return {"calls": len(calls), "errors": len(errors), "status": st.get("status"), "na_reason": st.get("na_reason"),
            "wall_s": st.get("seconds"), "latency_mean_s": float(lat.mean()), "latency_p50_s": q(lat, 50), "latency_p90_s": q(lat, 90),
            "ttft_mean_s": float(np.mean(ttft)) if ttft else None,
            "input_tokens_mean": float(inp.mean()), "input_tokens_max": int(inp.max()),
            "output_tokens_mean": float(out.mean()), "output_tokens_max": int(out.max()),
            "tok_per_s_per_call": float((out / np.maximum(lat, 1e-6)).mean()),
            "tok_per_s_run": float(out.sum() / st["seconds"]) if st.get("seconds") else None,
            "reasoning_chars_mean": float(np.mean(rch)), "reasoning_chars_max": int(max(rch)),
            "reasoning_tokens_mean": float(np.mean(rtok)) if rtok else None,
            "finish_reason": fin, "json_fail": sum(not x.get("json", True) for x in parse),
            "invalid_replies": sum(bool(x.get("error")) for x in parse),
            "max_tokens_clamped": sum(bool(c.get("max_tokens_clamped")) for c in calls)}


def main():
    cmd = sys.argv[1]
    if cmd == "args":
        print("\n".join(vllm_args(load_eval_config(), sys.argv[2], int(sys.argv[3]))))
    elif cmd == "wait":
        print(json.dumps(wait(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 3600)))
    elif cmd == "throughput":
        for d in sys.argv[2:]:
            print(json.dumps({"run": d, **throughput(Path(d))}, ensure_ascii=False))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
