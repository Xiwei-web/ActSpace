"""Open-model integration: vLLM adapter, concurrency, detailed logging, N/A, budget_infeasible, score-table appendix.
Uses fake servers only (injected transport / http); starts no real server and uses no GPU."""
import gzip
import json
import os
import re
import stat

import pytest

from pathlib import Path

from actspace_eval.data import FrozenData, load_eval_config
from actspace_eval.llm_run import run_llm, setting_dir
from actspace_eval.meta import group_meta
from actspace_eval.models import OpenAIClient
from actspace_eval.models_vllm import (ModelLimit, VLLMClient, fold_repeats, is_image_limit_error, merge_chunks,
                                       parse_context_error, sample_seed)
from actspace_eval.open_serve import vllm_args
from actspace_eval.trajectory import complete_episodes, replay

ROOT = Path(__file__).resolve().parents[2]
TWO = ["arkit_g_000000", "scannetpp_g_000009"]
OPEN = ["Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.6-27B", "Qwen3.8-27B", "Qwen3.6-35B-A3B", "gemma-4-31B-it", "InternVL3_5-8B",
        "SenseNova-SI-1.2-InternVL3-8B", "SenseNova-SI-1.3-InternVL3-8B", "VST-7B-RL", "Cambrian-S-7B"]


@pytest.fixture()
def data():
    return FrozenData(verify=False)


def _text_of(msg):
    c = msg["content"]
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in c if p["type"] == "text")


class FakeVLLM:
    """Fake vLLM server: Active moves twice and then answers, static settings answer directly; streamed chunks carry reasoning,
    reply and logprobs."""

    def __init__(self, think=True, limit=None, images_max=None):
        self.bodies, self.think, self.limit, self.images_max = [], think, limit, images_max
        self.lock = __import__("threading").Lock()

    def reply(self, body):
        msgs = body["messages"]
        active = msgs[0]["content"].startswith("You are exploring")
        n_assist = sum(m["role"] == "assistant" for m in msgs)
        if active and n_assist < 2:
            return json.dumps({"action": ["forward", "turn_left"][n_assist], "notes": "x"})
        qs = re.findall(r"^(Q\d+) \((multiple choice|numeric, meters)\)", _text_of(msgs[1]), re.M)
        return json.dumps({"action": "answer", "answers": {q: (1.5 if k.startswith("numeric") else "A") for q, k in qs}})

    def __call__(self, path, body):
        with self.lock:
            self.bodies.append(body)
        n_img = sum(p.get("type") == "image_url" for m in body["messages"] if isinstance(m["content"], list) for p in m["content"])
        if self.images_max is not None and n_img > self.images_max:
            return {"_http_status": 400, "_text": json.dumps({"error": {"message": f"At most {self.images_max} image(s) may be provided in one prompt."}})}
        n_in = 1000 + 300 * n_img
        if self.limit is not None and n_in + body["max_tokens"] > self.limit:
            return {"_http_status": 400, "_text": json.dumps({"error": {"message":
                    f"'max_tokens' or 'max_completion_tokens' is too large: {body['max_tokens']}. This model's maximum context length is "
                    f"{self.limit} tokens and your request has {n_in} input tokens ({body['max_tokens']} > {self.limit} - {n_in})."}})}
        text = self.reply(body)
        lp = lambda tok: {"token": tok, "logprob": -0.1, "top_logprobs": [{"token": tok, "logprob": -0.1}] + [{"token": f"x{i}", "logprob": -3.0} for i in range(4)]}
        chunks = []
        if self.think:
            chunks += [{"id": "c1", "model": body["model"], "choices": [{"index": 0, "delta": {"reasoning_content": "Let me think. "}, "logprobs": {"content": [lp("Let"), lp(" me"), lp(" think")]}}]},
                       {"id": "c1", "choices": [{"index": 0, "delta": {"reasoning_content": "Okay."}, "logprobs": {"content": [lp("Okay")]}}]}]
        chunks += [{"id": "c1", "choices": [{"index": 0, "delta": {"content": text[:5]}, "logprobs": {"content": [lp(text[:5])]}}]},
                   {"id": "c1", "choices": [{"index": 0, "delta": {"content": text[5:]}, "logprobs": {"content": [lp(text[5:])]}, "finish_reason": "stop"}]},
                   {"id": "c1", "choices": [], "usage": {"prompt_tokens": n_in, "completion_tokens": 6 if self.think else 2, "total_tokens": n_in + 6}}]
        return chunks


def fake_http(path, body):
    if path == "/tokenize":
        toks = ["<|im_start|>", "system", "\n", "SYS", "<|image_pad|>", "<|im_end|>", "<think>\n"]
        ids = [0, 1, 2, 3] + [4] * 12 + [5, 6]
        return {"count": len(ids), "max_model_len": 65536, "tokens": ids, "token_strs": [toks[i] for i in ids]}
    if path == "/detokenize":
        return {"prompt": "".join(["<|im_start|>", "system", "\n", "SYS", "<|image_pad|>", "<|im_end|>", "<think>\n"][i] for i in body["tokens"])}
    raise AssertionError(path)


def test_open_models_configured(data):
    ms = data.ecfg["models"]
    for k in OPEN:
        m = ms[k]
        assert m["provider"] == "vllm" and m["max_output_tokens"] == 32768 and m["json_mode"] and m["logprobs_top"] == 5
        assert m["prices_usd_per_mtok"] == {"input": 0.0, "cached_input": 0.0, "output": 0.0}
        assert "temperature" in m["sampling"] and m["sampling_source"]
        assert m["path"].endswith("/" + k) and os.path.isdir(m["path"])
    # thinking enabled where supported; models that can only enable it via the system prompt keep the default
    assert all(ms[k]["chat_template_kwargs"] == {"enable_thinking": True} and ms[k]["thinking"]["enabled"]
               for k in ("Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.6-27B", "Qwen3.6-35B-A3B", "gemma-4-31B-it"))
    assert not any(ms[k]["thinking"]["enabled"] for k in ("InternVL3_5-8B", "VST-7B-RL", "SenseNova-SI-1.2-InternVL3-8B", "Cambrian-S-7B"))


def test_vllm_serve_args(data):
    ec = data.ecfg
    for k in OPEN:
        if ec["models"][k]["vllm"] is None:
            with pytest.raises(ValueError):
                vllm_args(ec, k, 8100)
            continue
        a = vllm_args(ec, k, 8100)
        get = lambda f: a[a.index(f) + 1]
        assert get("--dtype") == "bfloat16" and get("--host") == "127.0.0.1"
        assert get("--chat-template-content-format") == "openai"   # otherwise vLLM may move all images in front of the text
        assert json.loads(get("--limit-mm-per-prompt"))["image"] >= 45
        assert int(get("--max-model-len")) <= ec["open_models"]["max_model_len_cap"]
        assert get("--served-model-name") == ec["models"][k]["model"]
        assert ("--reasoning-parser" in a) == bool(ec["models"][k]["vllm"]["reasoning_parser"])
        assert ("--trust-remote-code" in a) == ("InternVL" in k)
    assert vllm_args(ec, "Qwen3.6-35B-A3B", 1)[vllm_args(ec, "Qwen3.6-35B-A3B", 1).index("--tensor-parallel-size") + 1] == "4"


def test_request_identical_to_closed_models(data):
    """Message structure identical to the closed models; parameters: JSON mode, thinking switch, logprobs top-5, sampling
    parameters and the derived seed."""
    fake = FakeVLLM()
    c = VLLMClient(data.ecfg["models"]["Qwen3.6-27B"], data.ecfg["llm"], transport=fake, http=fake_http)
    import numpy as np
    img = np.zeros((480, 640, 3), np.uint8)
    turns = [{"role": "user", "parts": [("text", "Q1 (multiple choice): which?\n"), ("image", "photo_000001", img)]}]
    c.run_seed = 2
    comp = c.complete("SYS", turns, tag={"episode": "arkit_g_000000", "t": 3, "mode": "passive"})
    body = fake.bodies[0]
    ref, _ = OpenAIClient(data.ecfg["models"]["gpt-6.1-sol"], data.ecfg["llm"], transport=lambda b: None).to_messages("SYS", turns)
    assert body["messages"] == ref
    assert body["response_format"] == {"type": "json_object"} and body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["logprobs"] is True and body["top_logprobs"] == 5 and body["max_tokens"] == 32768 and body["stream"] is True
    assert (body["temperature"], body["top_p"], body["top_k"]) == (1.0, 0.95, 20)
    assert body["seed"] == sample_seed(2, "arkit_g_000000", 3) != sample_seed(0, "arkit_g_000000", 3)
    assert comp.extra["reasoning"] == "Let me think. Okay." and comp.extra["reasoning_tokens"] == 3 + 1


def test_merge_chunks_and_helpers():
    raw, ttft = merge_chunks([(1.0, {"choices": [{"delta": {"reasoning_content": "a"}, "logprobs": {"content": [{"token": "a"}]}}]}),
                              (2.0, {"choices": [{"delta": {"content": "{}"}, "logprobs": {"content": [{"token": "{}"}]}, "finish_reason": "stop"}]}),
                              (2.5, {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}})], 0.5)
    m = raw["choices"][0]
    assert ttft == pytest.approx(0.5) and m["message"] == {"role": "assistant", "content": "{}", "reasoning_content": "a"}
    assert [x["_reasoning"] for x in m["logprobs"]["content"]] == [True, False] and m["finish_reason"] == "stop"
    assert fold_repeats(["a"] + ["<i>"] * 5 + ["b", "b"]) == "a<i>×5bb"
    assert parse_context_error("This model's maximum context length is 32768 tokens. However, your request has 40000 input tokens. "
                               "Please reduce the length of the input messages.") == (32768, 40000)
    assert parse_context_error("'max_tokens' or 'max_completion_tokens' is too large: 16384. This model's maximum context length is "
                               "32768 tokens and your request has 20000 input tokens (16384 > 32768 - 20000).") == (32768, 20000)
    assert parse_context_error("HTTP 500 boom") is None
    assert is_image_limit_error("At most 48 image(s) may be provided in one prompt.")


@pytest.mark.parametrize("setting,budget", [("active_v2", 40), ("passive", 40), ("random_walk", 40), ("blind", 40)])
def test_open_run_detailed_records(data, tmp_path, setting, budget):
    """Detailed logging: calls (message structure with frame IDs and full image SHA-256, reasoning text, time to first token,
    effective max_tokens and seed, parse result, rendered turn-0 prompt), logprobs, per-step and per-question fields, env.json;
    2 concurrent workers; SHA256SUMS written and files made read-only on completion; replayable."""
    fake = FakeVLLM()
    out = tmp_path / "Qwen3.5-9B" / setting_dir(setting, budget) / "smoke_r1"
    st = run_llm(data, "Qwen3.5-9B", setting, budget, TWO, out, print_estimate=False, run_seed=1, workers=2,
                 client_kwargs={"transport": fake, "http": fake_http}, env_info={"model_path": "x"}, seal=True)
    assert st["status"] == "complete" and st["completed"] == 2 and st["thinking"] is True
    calls = [json.loads(l) for l in open(out / "calls.jsonl")]
    eps = complete_episodes(out / "trajectories.jsonl")
    assert len({c["tag"]["episode"] for c in calls}) == 2
    nodes = {e[0]["group_id"]: {e[0]["start_node"]} | set(e[0].get("shown_nodes", [])) | {r["to"] for r in e if r["type"] == "step"} for e in eps}
    for c in calls:
        assert c["reasoning"] == "Let me think. Okay." and c["ttft_s"] is not None and c["max_tokens"] == 32768 and c["run_seed"] == 1
        assert c["parse"]["json"] and c["parse"]["error"] is None and c["finish_reason"] == "stop"
        assert c["seed"] == sample_seed(1, c["tag"]["episode"], c["tag"]["t"])
        imgs = [p for m in c["request"]["messages"] if isinstance(m["content"], list) for p in m["content"] if p["type"] == "image"]
        for p in imgs:
            assert re.fullmatch(r"[0-9a-f]{64}", p["sha256"]) and p["frame_id"] in nodes[c["tag"]["episode"]] and (p["w"], p["h"]) in ((640, 480), (480, 640))
        if c["tag"]["t"] == 0:
            assert c["rendered_prompt"]["text"] == "<|im_start|>system\nSYS<|image_pad|>×12<|im_end|><think>\n"
        assert "base64" not in json.dumps(c["request"])
    if setting != "blind":
        assert any(p["type"] == "image" for c in calls for m in c["request"]["messages"] if isinstance(m["content"], list) for p in m["content"])
    with gzip.open(out / "logprobs.jsonl.gz", "rt") as f:
        lps = [json.loads(l) for l in f]
    assert len(lps) == len(calls) and {x["call_id"] for x in lps} == {c["call_id"] for c in calls}
    assert all(len(x["top"][0]) == 5 and x["reasoning"][:4] == [True] * 4 and x["reasoning"][-1] is False for x in lps)
    for e in eps:
        assert e[0]["thinking"] is True and e[0]["sampling"] == {"temperature": 0.7, "top_p": 0.9} and e[0]["run_seed"] == 1
        assert e[0]["budget_infeasible"] is False
        for r in e:
            if r["type"] == "step":
                assert {"cannot_move", "invalid", "odometry", "pose", "photo_id", "to", "steps_used"} <= set(r)
    assert replay(data, out / "trajectories.jsonl")["mismatched_episodes"] == 0
    s = json.loads((out / "summary.json").read_text())
    for x in s["per_question"]:
        assert {"gold", "answer", "score", "cannot_determine", "evidence_hit", "dataset", "template", "budget_infeasible"} <= set(x)
    env = json.loads((out / "env.json").read_text())
    assert env["sampling"] == {"temperature": 0.7, "top_p": 0.9} and env["thinking"] is True and env["workers"] == 2
    assert env["data_version"] == "merged_v1.0" and len(env["freeze_manifest_sha256"]) == 64 and env["ended_at"] and env["code"]["commit"]
    sums = (out / "SHA256SUMS").read_text().split("\n")
    assert {l.split("  ")[1] for l in sums if l} == {"calls.jsonl", "env.json", "logprobs.jsonl.gz", "status.json", "summary.json", "trajectories.jsonl", "instances.jsonl"}
    assert sorted(json.loads(l)["group_id"] for l in open(out / "instances.jsonl")) == sorted(TWO)
    assert not (out / "summary.json").stat().st_mode & stat.S_IWUSR and not out.stat().st_mode & stat.S_IWUSR
    out.chmod(0o755)   # let pytest clean up the temporary directory
    for p in out.iterdir():
        p.chmod(0o644)


def test_resume_drops_incomplete_and_keeps_records(data, tmp_path):
    fake = FakeVLLM(think=False)
    out = tmp_path / "r"
    run_llm(data, "VST-7B-RL", "active_v2", 40, TWO[:1], out, print_estimate=False, client_kwargs={"transport": fake, "http": fake_http})
    with open(out / "trajectories.jsonl", "a") as f:   # simulate an interruption: one incomplete episode
        f.write(json.dumps({"type": "episode_start", "group_id": TWO[1]}) + "\n")
    st = run_llm(data, "VST-7B-RL", "active_v2", 40, TWO, out, resume=True, print_estimate=False, client_kwargs={"transport": fake, "http": fake_http})
    assert st["status"] == "complete" and st["completed"] == 2 and st["thinking"] is False
    assert replay(data, out / "trajectories.jsonl")["mismatched_episodes"] == 0


def test_context_clamped_then_na(data, tmp_path):
    """Context: if input + 32768 exceeds it, max_tokens is narrowed to the remaining space (recorded); if the remaining space is
    below min_output_tokens, the setting is N/A."""
    fake = FakeVLLM(think=False, limit=12000)       # passive_8: 9 images -> 1000 + 2700 = 3700 input tokens, 8300 left
    st = run_llm(data, "VST-7B-RL", "passive", 8, TWO, tmp_path / "p8", print_estimate=False,
                 client_kwargs={"transport": fake, "http": fake_http})
    assert st["status"] == "complete"
    calls = [json.loads(l) for l in open(tmp_path / "p8" / "calls.jsonl") if '"type": "call"' in l]
    assert all(c["max_tokens"] == 8300 and c["max_tokens_clamped"]["from"] == 32768 for c in calls)
    fake = FakeVLLM(think=False, limit=12000)       # passive_40: 41 images -> 13300 input tokens, over the limit
    st = run_llm(data, "VST-7B-RL", "passive", 40, TWO, tmp_path / "p40", print_estimate=False,
                 client_kwargs={"transport": fake, "http": fake_http}, workers=2)
    assert st["status"] == "not_applicable" and st["na_reason"]["kind"] == "context" and "12000" in st["na_reason"]["message"]


def test_context_at_least_retries_without_max_tokens(data):
    """When vLLM reports a truncated count ("at least N input tokens"), retry without max_tokens and let the server use the
    remaining space; record the actual remaining space."""
    seen = []

    def tr(path, body):
        seen.append(body)
        if "max_tokens" in body:
            return {"_http_status": 400, "_text": json.dumps({"error": {"message": "This model's maximum context length is 32768 tokens. However, "
                    "you requested 16384 output tokens and your prompt contains at least 16385 input tokens, for a total of at least 32769 tokens."}})}
        return FakeVLLM(think=False)(path, {**body, "max_tokens": 0})
    c = VLLMClient(data.ecfg["models"]["VST-7B-RL"], data.ecfg["llm"], transport=tr, http=fake_http)
    comp = c.complete("SYS", [{"role": "user", "parts": [("text", "Q1 (multiple choice): x")]}], tag={"episode": "g", "t": 1})
    assert "max_tokens" not in seen[1] and comp.extra["max_tokens_clamped"]["to"] == "auto" and comp.extra["max_tokens"] == 32768 - 1000


def test_max_tokens_equal_to_context_is_not_sent(data):
    """SenseNova: context 32768, max_tokens 32768. vLLM computes zero input space and rejects every request (which once caused
    16 runs to be wrongly recorded as not_applicable)."""
    seen = []
    msg = ("This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens and your prompt contains 2365 characters "
           "(more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens.")

    def tr(path, body):
        seen.append(body)
        if "max_tokens" in body:
            return {"_http_status": 400, "_text": json.dumps({"error": {"message": msg}})}
        return FakeVLLM(think=False)(path, {**body, "max_tokens": 0})
    m = data.ecfg["models"]["SenseNova-SI-1.2-InternVL3-8B"]
    assert m["max_output_tokens"] == m["vllm"]["max_model_len"] == 32768
    comp = VLLMClient(m, data.ecfg["llm"], transport=tr, http=fake_http).complete("SYS", [{"role": "user", "parts": [("text", "Q1 (multiple choice): x")]}], tag={"episode": "g", "t": 1})
    assert len(seen) == 1 and "max_tokens" not in seen[0]
    assert comp.extra["max_tokens_clamped"]["to"] == "auto" and comp.extra["max_tokens"] == 32768 - comp.usage["input_tokens"]
    c = VLLMClient({**m, "vllm": {**m["vllm"], "max_model_len": 40960}}, data.ecfg["llm"], transport=tr, http=fake_http)
    c.complete("SYS", [{"role": "user", "parts": [("text", "Q1 (multiple choice): x")]}], tag={"episode": "g", "t": 1})   # the error wording itself takes the same path
    assert "max_tokens" in seen[1] and "max_tokens" not in seen[2]


def test_image_limit_na(data, tmp_path):
    fake = FakeVLLM(think=False, images_max=30)
    st = run_llm(data, "VST-7B-RL", "passive", 40, TWO, tmp_path / "p40", print_estimate=False, client_kwargs={"transport": fake, "http": fake_http})
    assert st["status"] == "not_applicable" and st["na_reason"]["kind"] == "images"
    errs = [json.loads(l) for l in open(tmp_path / "p40" / "calls.jsonl")]
    assert errs and errs[0]["type"] == "error" and errs[0]["limit_kind"] == "images" and errs[0]["gave_up"]


def test_concurrent_mock_runs_match_sequential(data, tmp_path):
    """Concurrent (workers > 1) and sequential runs give identical trajectories (mock agent; episodes are independent)."""
    gids = ["arkit_g_000000", "arkit_g_000072", "scannetpp_g_000009"]
    a = run_llm(data, "mock", "active_v2", 40, gids, tmp_path / "seq", print_estimate=False, render=False)
    b = run_llm(data, "mock", "active_v2", 40, gids, tmp_path / "par", print_estimate=False, render=False, workers=3)
    assert a["status"] == b["status"] == "complete"
    key = lambda p: {e[0]["group_id"]: [(r["type"], r.get("action"), r.get("to")) for r in e] for e in complete_episodes(p)}
    assert key(tmp_path / "seq" / "trajectories.jsonl") == key(tmp_path / "par" / "trajectories.jsonl")


def test_budget_infeasible_matches_routes(data):
    meta = group_meta(data)
    over = sorted(g for g in data.group_ids if meta[g]["group_route_steps"] > data.ecfg["env"]["budget"])
    assert over == sorted(data.ecfg["env"]["budget_infeasible"]) and len(over) == 7
    assert not set(over) & set(data.ecfg["llm"]["tiny_groups"])


def test_report_eval_infeasible_appendix(data, tmp_path):
    from actspace_eval import report_eval as RE
    gids = ["arkit_g_000000", "scannetpp_g_000009", "scannetpp_g_000000"]
    for setting, name, seed in [("random_walk", "rw0", 0), ("active_v2", "a0", 0), ("passive", "p40", 0)]:
        assert run_llm(data, "mock", setting, 40, gids, tmp_path / name, print_estimate=False, render=False, run_seed=seed)["status"] == "complete"
    s = json.loads((tmp_path / "a0" / "summary.json").read_text())
    assert s["budget_infeasible_groups"] == ["scannetpp_g_000000"]
    assert [e["efficiency"] for e in s["per_episode"] if e["group_id"] == "scannetpp_g_000000"] == [None]
    R = RE.build(tmp_path, {"random_walk_40": ["rw0"], "active_v2": ["a0"], "passive_40": ["p40"]})
    assert all(x["group_id"] != "scannetpp_g_000000" for rs in R["runs"].values() for r in rs for x in r["per_question"])
    assert {x["group_id"] for r in R["infeasible"]["runs"]["active_v2"] for x in r["per_question"]} == {"scannetpp_g_000000"}
    text = "\n".join(RE.render(R))
    assert "## Appendix. Groups not completable within the budget (1 groups)" in text and "cannot be completed within 40 steps" in text and "scannetpp_g_000000" in text
    # --root and run names of the form "<setting>/<run>" (open-model results directory)
    (tmp_path / "M" / "active_v2").mkdir(parents=True)
    (tmp_path / "a0").rename(tmp_path / "M" / "active_v2" / "a0")
    R2 = RE.build(tmp_path / "M", {"active_v2": ["active_v2/a0"]})
    assert len(R2["runs"]["active_v2"][0]["per_question"]) == len(R["runs"]["active_v2"][0]["per_question"])


def test_sample_seed_stable():
    assert sample_seed(0, "g", 1) == sample_seed(0, "g", 1) and len({sample_seed(k, "g", 1) for k in (0, 1, 2)}) == 3
    assert 0 <= sample_seed(2, "g", 5) < 2 ** 31


def test_model_limit_is_not_retryable():
    e = ModelLimit("context: x", "context")
    assert not e.retryable and e.kind == "context"


# ---------------------------------------------------------------- fixes after the first smoke test

class Flaky(FakeVLLM):
    """The first n requests fail according to kind, then everything is normal."""

    def __init__(self, n, kind, **kw):
        super().__init__(think=False, **kw)
        self.n, self.kind = n, kind

    def __call__(self, path, body):
        with self.lock:
            bad = self.n > 0
            self.n -= 1
        if not bad:
            return super().__call__(path, body)
        self.bodies.append(body)
        if self.kind == "empty":      # seen with InternVL3.5 in the smoke test: the engine rejects, the stream only has usage
            return [{"id": "c", "choices": [], "usage": {"prompt_tokens": 8000, "completion_tokens": 0, "total_tokens": 8000}}]
        if self.kind == "stream_error":
            return [{"object": "error", "message": "EngineCore died", "code": 500}]
        if self.kind == "cut":        # the stream breaks off mid-generation
            return [{"id": "c", "choices": [{"index": 0, "delta": {"content": "{"}, "logprobs": {"content": []}}]}]
        return {"_http_status": 500, "_text": "boom"}


def nosleep(s):
    return None


@pytest.mark.parametrize("kind", ["empty", "stream_error", "cut", "http500"])
def test_server_errors_are_retried_not_invalid_replies(data, tmp_path, kind):
    """Recovery after retrying server errors: no step counted, not an invalid reply, errors written to the call log."""
    st = run_llm(data, "VST-7B-RL", "active_v2", 40, TWO[:1], tmp_path / "r", print_estimate=False,
                 client_kwargs={"transport": Flaky(2, kind), "http": fake_http, "sleep": nosleep})
    assert st["status"] == "complete" and st["usage"]["errors"] == 2
    s = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert s["invalid_steps"] == 0 and s["forced_answers"] == {"None": 1}
    assert all((json.loads(l).get("parse") or {}).get("error") is None for l in open(tmp_path / "r" / "calls.jsonl"))


def test_retries_exhausted_stopped_server_error_then_resume(data, tmp_path):
    """Retries exhausted: stopped_server_error, the unfinished episode is not written to the trajectories; --resume completes it
    once the server recovers."""
    st = run_llm(data, "VST-7B-RL", "active_v2", 40, TWO[:1], tmp_path / "r", print_estimate=False,
                 client_kwargs={"transport": Flaky(10 ** 6, "empty"), "http": fake_http, "sleep": nosleep})
    assert st["status"] == "stopped_server_error" and st["completed"] == 0 and st["remaining"] == TWO[:1]
    assert complete_episodes(tmp_path / "r" / "trajectories.jsonl") == []
    st = run_llm(data, "VST-7B-RL", "active_v2", 40, TWO[:1], tmp_path / "r", resume=True, print_estimate=False,
                 client_kwargs={"transport": FakeVLLM(think=False), "http": fake_http, "sleep": nosleep})
    assert st["status"] == "complete" and st["completed"] == 1


def test_truncated_thinking_is_a_model_reply_not_server_error(data, tmp_path):
    """Reasoning that fills the limit (finish_reason = length, empty content) is a model reply: counted as no answer, not retried."""
    def tr(path, body):
        return [{"id": "c", "choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}, "logprobs": {"content": [{"token": "hmm", "logprob": -1.0, "top_logprobs": []}]}, "finish_reason": "length"}]},
                {"id": "c", "choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 16384}}]
    st = run_llm(data, "Qwen3.5-9B", "passive", 8, TWO[:1], tmp_path / "r", print_estimate=False,
                 client_kwargs={"transport": tr, "http": fake_http, "sleep": nosleep})
    assert st["status"] == "complete" and st["usage"]["errors"] == 0
    c = json.loads(open(tmp_path / "r" / "calls.jsonl").readline())
    assert c["finish_reason"] == "length" and c["parse"]["error"] == "empty reply"
    assert json.loads((tmp_path / "r" / "summary.json").read_text())["mean_score_all_questions"] == 0


def test_input_length_error_is_na(data, tmp_path):
    """Error wording observed in the smoke test: context exceeded after expanding images -> N/A (not a server error)."""
    msg = "Input length (137568) exceeds model's maximum context length (40960)."
    assert parse_context_error(msg) == (40960, 137568)
    st = run_llm(data, "InternVL3_5-8B", "passive", 40, TWO[:1], tmp_path / "r", print_estimate=False,
                 client_kwargs={"transport": lambda p, b: {"_http_status": 400, "_text": json.dumps({"error": {"message": msg}})}, "http": fake_http, "sleep": nosleep})
    assert st["status"] == "not_applicable" and st["na_reason"]["kind"] == "context" and st["usage"]["errors"] == 1


def test_silently_capped_output_is_recorded_and_na_when_cut_off(data):
    """For multi-image requests vLLM does not fail but squeezes the output into the remaining space: record effective_max_tokens;
    truncated with less than min_output_tokens left counts as insufficient context."""
    m = {**data.ecfg["models"]["SenseNova-SI-1.2-InternVL3-8B"], "max_output_tokens": 16384}

    def tr(n_in, fin):
        return lambda p, b: [{"id": "c", "choices": [{"index": 0, "delta": {"content": "{}"}, "logprobs": {"content": []}, "finish_reason": fin}]},
                             {"id": "c", "choices": [], "usage": {"prompt_tokens": n_in, "completion_tokens": 5}}]
    turns = [{"role": "user", "parts": [("text", "x")]}]
    c = VLLMClient(m, data.ecfg["llm"], transport=tr(30000, "stop"), http=fake_http).complete("S", turns, tag={"episode": "g", "t": 1})
    assert c.extra["effective_max_tokens"] == 2768 and c.extra["max_tokens"] == 16384
    with pytest.raises(ModelLimit):
        VLLMClient(m, data.ecfg["llm"], transport=tr(32700, "length"), http=fake_http).complete("S", turns, tag={"episode": "g", "t": 1})


def test_internvl_family_tiles_and_serve_flags(data):
    ec = data.ecfg
    get = lambda a, f: a[a.index(f) + 1]
    assert json.loads(get(vllm_args(ec, "InternVL3_5-8B", 1), "--hf-overrides")) == {"max_dynamic_patch": 1}
    for k in ("SenseNova-SI-1.2-InternVL3-8B", "SenseNova-SI-1.3-InternVL3-8B"):
        a = vllm_args(ec, k, 1)
        assert json.loads(get(a, "--hf-overrides")) == {"max_dynamic_patch": 1} and "--mm-processor-kwargs" not in a
        assert os.path.isfile(get(a, "--chat-template")) and get(a, "--max-model-len") == "32768"
    assert all(get(vllm_args(ec, k, 1), "--mm-processor-cache-gb") == "0" for k in OPEN if ec["models"][k]["vllm"])
    assert "--hf-overrides" not in vllm_args(ec, "Qwen3.5-9B", 1)


@pytest.mark.parametrize("model", ["SenseNova-SI-1.2-InternVL3-8B", "SenseNova-SI-1.3-InternVL3-8B"])
def test_sensenova_template_verbatim(data, model):
    """The adapted template renders the openai format (system prompt as a list) exactly like the official template renders a
    string system prompt."""
    jinja2 = pytest.importorskip("jinja2")
    m = data.ecfg["models"][model]
    official = json.load(open(m["path"] + "/tokenizer_config.json"))["chat_template"]
    served = open(ROOT / m["vllm"]["chat_template"]).read()
    user = [{"type": "text", "text": "Q1\n"}, {"type": "image"}, {"type": "text", "text": "photo_1\n"}, {"type": "image"}]
    conv = [{"role": "user", "content": user}, {"role": "assistant", "content": "{}"}, {"role": "user", "content": user}]
    r = lambda t, sysc: jinja2.Environment().from_string(t).render(messages=[{"role": "system", "content": sysc}] + conv, add_generation_prompt=True)
    want = r(official, "SYSTEM TEXT\nline 2")
    assert r(served, [{"type": "text", "text": "SYSTEM TEXT\nline 2"}]) == want and r(served, "SYSTEM TEXT\nline 2") == want
    assert want.startswith("<|im_start|>system\nSYSTEM TEXT\nline 2<|im_end|>\n<|im_start|>user\nQ1\n<image>\nphoto_1\n<image>\n<|im_end|>")
    with pytest.raises(TypeError):
        r(official, [{"type": "text", "text": "x"}])   # the error from the first smoke test


def test_max_tokens_override_recorded(data, tmp_path):
    fake = FakeVLLM()
    st = run_llm(data, "Qwen3.5-9B", "passive", 8, TWO[:1], tmp_path / "r", print_estimate=False, max_tokens=32768,
                 client_kwargs={"transport": fake, "http": fake_http}, env_info={})
    assert st["status"] == "complete" and fake.bodies[0]["max_tokens"] == 32768
    assert json.loads((tmp_path / "r" / "env.json").read_text())["max_output_tokens"] == 32768
    assert complete_episodes(tmp_path / "r" / "trajectories.jsonl")[0][0]["max_output_tokens"] == 32768


def test_cambrian_non_stream_path(data, tmp_path):
    """Cambrian-S wrapper server: non-streaming; turn 0 asks for the rendered prompt; time to first token and tokens per image come
    from the server."""
    seen = []

    def tr(path, body):
        seen.append(body)
        text = FakeVLLM().reply(body)
        return {"id": "c", "model": "Cambrian-S-7B", "usage": {"prompt_tokens": 3000, "completion_tokens": 9, "total_tokens": 3009},
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text},
                             "logprobs": {"content": [{"token": "{", "logprob": -0.1, "top_logprobs": [{"token": "{", "logprob": -0.1}]}]}}],
                "x_timing": {"ttft_s": 0.5, "total_s": 1.0}, "x_image_tokens": [2788], "x_image_mode": "anyres", "x_json_mode": "enforced",
                **({"x_prompt_text": "<|im_start|>system"} if body.get("return_prompt_text") else {})}
    st = run_llm(data, "Cambrian-S-7B", "active_v2", 40, TWO[:1], tmp_path / "r", print_estimate=False, client_kwargs={"transport": tr, "sleep": nosleep})
    assert st["status"] == "complete"
    calls = [json.loads(l) for l in open(tmp_path / "r" / "calls.jsonl")]
    assert seen[0]["return_prompt_text"] is True and "return_prompt_text" not in seen[1] and "stream" not in seen[0]
    assert calls[0]["rendered_prompt"]["text"] == "<|im_start|>system" and calls[0]["ttft_s"] == 0.5
    assert calls[0]["x_image_tokens"] == [2788] and calls[0]["x_json_mode"] == "enforced" and calls[0]["logprobs_tokens"] == 1
    assert (seen[0]["top_k"], seen[0]["repetition_penalty"]) == (20, 1.05)


def test_repeat_ratio_separates_loops():
    from actspace_eval.report_open_smoke import repeat_ratio
    loop = " ".join(f"Image {i}: Banjo on floor." for i in range(600))
    rng = __import__("random").Random(0)
    prose = " ".join("".join(rng.choice("abcdefghij") for _ in range(5)) for _ in range(3000))
    assert repeat_ratio(loop) > 0.9 and repeat_ratio(prose) < 0.1 and repeat_ratio("") == 0


def test_interleave_check_flags_images_moved_to_front(tmp_path):
    from actspace_eval.report_open_smoke import interleave_ok
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": [{"type": "text", "text": "x"}, {"type": "image", "photo_id": "photo_1"}, {"type": "image", "photo_id": "photo_2"}]}]

    def run(name, text):
        d = tmp_path / name
        d.mkdir()
        (d / "calls.jsonl").write_text(json.dumps({"type": "call", "tag": {"t": 0}, "request": {"messages": msgs}, "rendered_prompt": {"text": text}}) + "\n")
        return interleave_ok(d)
    assert run("good", "START photo_1\n<|image_pad|>×300\nphoto_2\n<|image_pad|>×300\nAnswer") == {"checked": 1, "ok": 1}
    assert run("bad", "<|image_pad|>×300\n<|image_pad|>×300\nSTART photo_1\n\n\nphoto_2\n\nAnswer") == {"checked": 1, "ok": 0}
    assert run("revisit", "START photo_1\n<|image_pad|>×300\nphoto_2\n<|image_pad|>×300\nback to photo_1 (shown above)\nback to photo_2\nAnswer") == {"checked": 1, "ok": 1}


def _usage_run(d, seconds, n_calls, wall):
    d.mkdir(parents=True)
    with open(d / "calls.jsonl", "w") as f:
        for g in ("g0", "g1", "g2"):
            for _ in range(n_calls):
                f.write(json.dumps({"type": "call", "tag": {"group": g, "episode": g}, "group_id": g, "latency_s": seconds, "finish_reason": "stop",
                                    "usage": {"input_tokens": 1000, "output_tokens": 100}}) + "\n")
    (d / "status.json").write_text(json.dumps({"status": "complete", "seconds": wall}))


def test_gpu_hours_uses_peak_prefill_and_covers_serial_server(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from actspace_eval import report_open_smoke as r
    monkeypatch.setattr(r, "per_episode_usage", lambda d: {g: {"input_tokens": 1000, "output_tokens": 100, "seconds": 10.0} for g in ("g0", "g1", "g2")})
    _usage_run(tmp_path / "blind", 10, 1, 30)
    dirs = {"blind": tmp_path / "blind"}
    ec = {"models": {"m": {"vllm": {"tp": 2}}, "c": {"vllm": None}}, "llm": {"tiny_groups": ["g"] * 6}}
    data = SimpleNamespace(group_ids=["g"] * 6)
    srv = {"gen_tok_s_busy_median": 100.0, "prompt_tok_s_busy_median": 1.0, "prompt_tok_s_p90": 1.0, "prompt_tok_s_max": 1000.0, "prefix_hit_last_pct": 0}
    x = r.estimate_gpu_hours(ec, data, "m", dirs, srv)["tiny32"]
    assert abs(x["throughput_bound_h"] - (600 / 100 + 6000 / 1000) / 3600) < 1e-9   # prefill uses the peak window, not the sparse window mean
    assert abs(x["linear_bound_h"] - 30 * 6 / 3 / 3600) < 1e-9
    assert x["wall_h"] == x["linear_bound_h"] and x["gpu_hours"] == x["wall_h"] * 2
    y = r.estimate_gpu_hours(ec, data, "c", dirs, {})["tiny32"]                     # serial server: calls × wall clock per call
    assert abs(y["wall_h"] - 6 * (30 / 3) / 3600) < 1e-9 and y["gpus"] == 1


# ---------------------------------------------------------------- task queue
def _plan(**k):
    return {"prefix": "full", "seeds": [0], "models": ["M1", "M2"], "settings": ["blind", "random_walk_40", "active_v2"], "smoke_gate": {"M2": "smoke4"},
            "groups_tiny": ["g0"], "groups_all": ["g0", "g1"], "groups_smoke": ["g0"], **k}


def _traj(d, groups, partial=None):
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    pad = {"run": "r", "agent": "a" * 500}      # in real records, type comes after the extra fields of write_episode
    for g in groups:
        rows += [{**pad, "group_id": g, "type": "episode_start"}, {"type": "episode_end"}]
    if partial:
        rows.append({**pad, "group_id": partial, "type": "episode_start"})
    (d / "trajectories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_queue_plan_names_and_seeds():
    from actspace_eval.open_queue import plan_tasks
    names = {(t.model, t.setting, t.run, t.kind) for t in plan_tasks(_plan())}
    assert ("M1", "active_v2", "full_r0", "full") in names and ("M1", "random_walk_40", "full_s0", "full") in names
    assert ("M2", "blind", "smoke4_r0", "smoke") in names and not any(n[0] == "M1" and n[3] == "smoke" for n in names)
    more = [t.run for t in plan_tasks(_plan(seeds=[0, 1, 2], models=["M1"], smoke_gate={})) if t.setting == "active_v2"]
    assert more == ["full_r0", "full_r1", "full_r2"]
    assert [t.run for t in plan_tasks(_plan(seeds=[0, 1], models=["M1"], smoke_gate={})) if t.setting == "blind"] == ["full_r0"]


def test_queue_restores_state_from_disk(tmp_path):
    from actspace_eval.open_queue import DONE, FAILED, PENDING, plan_tasks, read_tsv, restore, write_tsv
    plan = _plan(models=["M1"], smoke_gate={})
    tasks = plan_tasks(plan)
    _traj(tmp_path / "M1/blind/full_r0", ["g0", "g1"])
    (tmp_path / "M1/blind/full_r0/SHA256SUMS").write_text("x")
    _traj(tmp_path / "M1/active_v2/full_r0", ["g0"], partial="g1")      # tiny done, 2nd group interrupted
    tasks[1].state, tasks[1].reason = FAILED, "boom"
    write_tsv(tmp_path / "tasks.tsv", tasks, "1")
    fresh = plan_tasks(plan)
    restore(fresh, tmp_path, plan, read_tsv(tmp_path / "tasks.tsv"))
    by = {t.setting: t for t in fresh}
    assert by["blind"].state == DONE and by["blind"].done == 2
    assert by["random_walk_40"].state == FAILED and by["random_walk_40"].reason == "boom"
    assert by["active_v2"].state == PENDING and by["active_v2"].done == 1 and by["active_v2"].tiny


def test_queue_workers_per_model():
    from actspace_eval.open_queue import worker_count
    q = load_eval_config()["open_models"]["queue"]
    assert worker_count(q, "VST-7B-RL", "passive_8") == 4 and worker_count(q, "VST-7B-RL", "active_v2") == 8
    assert worker_count(q, "Cambrian-S-7B", "active_v2") == 1 and worker_count(q, "Qwen3.6-27B", "blind") == 3
    assert set(q["order"]) == {k for k, m in load_eval_config()["models"].items() if m["provider"] == "vllm"}   # Qwen3.8 is part of the main queue


def _tok_run(d, entries):
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for gid, w, h, n in entries:
        rows.append({"type": "call", "tag": {"t": 0, "episode": gid}, "group_id": gid, "rendered_prompt": {"text": f"photo_1 <IMG>×{n} end"},
                     "request": {"messages": [{"role": "user", "content": [{"type": "image", "w": w, "h": h, "photo_id": "photo_1"}]}]}})
    (d / "calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_gate_needs_both_orientations_with_equal_tokens(tmp_path):
    from actspace_eval.report_open_full import gate_result
    _tok_run(tmp_path / "M/passive_8/s", [("arkit_g_1", 480, 640, 256), ("scannetpp_g_1", 640, 480, 256)])
    g = gate_result(tmp_path, "M", "s", ["passive_8"])
    assert g["pass"] and g["tokens"] == {"arkit 480×640": [256], "scannetpp 640×480": [256]}
    _tok_run(tmp_path / "M/passive_20/s", [("arkit_g_1", 480, 640, 768), ("scannetpp_g_1", 640, 480, 256)])
    assert not gate_result(tmp_path, "M", "s", ["passive_20"])["pass"]
    _tok_run(tmp_path / "M/passive_40/s", [("scannetpp_g_1", 640, 480, 256)])
    assert not gate_result(tmp_path, "M", "s", ["passive_40"])["pass"]      # no portrait images: not confirmed


def test_runs_merge_automatically_and_tiny_uses_unsealed(tmp_path):
    from actspace_eval.report_open_full import model_spec
    for name, sealed in (("full_s0", True), ("full_s1", True), ("full_s2", False), ("smoke_r0", True)):
        d = tmp_path / "M/random_walk_40" / name
        d.mkdir(parents=True)
        (d / "summary.json").write_text("{}")
        if sealed:
            (d / "SHA256SUMS").write_text("x")
    assert model_spec(tmp_path, "M", "full") == {"random_walk_40": ["random_walk_40/full_s0", "random_walk_40/full_s1"]}
    assert len(model_spec(tmp_path, "M", "full", tiny=True)["random_walk_40"]) == 3


def test_resume_moves_unfinished_calls_and_trims_partial_line(tmp_path):
    from actspace_eval.llm_run import drop_unfinished_calls, trim_partial_line
    p = tmp_path / "calls.jsonl"
    p.write_text(json.dumps({"tag": {"episode": "g0"}}) + "\n" + json.dumps({"tag": {"episode": "g1"}}) + "\n" + '{"tag": {"epi')
    trim_partial_line(p)
    drop_unfinished_calls(p, {"g0"})
    assert [json.loads(x)["tag"]["episode"] for x in p.read_text().splitlines()] == ["g0"]
    assert json.loads((tmp_path / "calls_dropped.jsonl").read_text())["tag"]["episode"] == "g1"


def test_single_run_note_in_report():
    from actspace_eval.report_eval import render
    R = {"scenes": {}, "boots": {}, "runs": {"active_v2": [{}], "blind": [{}]}}
    assert "**Single run so far: intervals do not include model sampling randomness.**" in render(R)
    R["runs"]["active_v2"] = [{}, {}]
    assert "**Single run so far: intervals do not include model sampling randomness.**" not in render(R)


def _fake_slurm(tmp_path, monkeypatch, squeue_out="", squeue_rc=0):
    """Fake squeue / sbatch / scancel: record the calls."""
    import subprocess as sp
    from types import SimpleNamespace
    calls = []

    def run(cmd, **k):
        calls.append(cmd)
        if cmd[0] == "squeue":
            return SimpleNamespace(returncode=squeue_rc, stdout=squeue_out, stderr="")
        return SimpleNamespace(returncode=0, stdout="777\n", stderr="")
    monkeypatch.setattr(sp, "run", run)
    return calls


def _queue(tmp_path, monkeypatch, job="100", **plan):
    from actspace_eval.open_queue import Queue
    (tmp_path / "_queue").mkdir(exist_ok=True)
    (tmp_path / "_queue/plan.json").write_text(json.dumps(_plan(models=["VST-7B-RL"], smoke_gate={}, report_dir=str(tmp_path), **plan)))
    monkeypatch.setenv("SLURM_JOB_ID", job)
    monkeypatch.setenv("QUEUE_STOP_AFTER_S", "1000")
    return Queue(tmp_path)


def test_next_job_is_submitted_once_with_dependency(tmp_path, monkeypatch):
    q = _queue(tmp_path, monkeypatch)
    calls = _fake_slurm(tmp_path, monkeypatch)
    q.ensure_next()
    sb = [c for c in calls if c[0] == "sbatch"]
    assert len(sb) == 1 and "--dependency=afterany:100" in sb[0] and sb[0][-1] == "scripts/run_open_queue.sbatch"


@pytest.mark.parametrize("out,rc", [("100\n200\n", 0), ("", 1)])
def test_no_second_next_job_and_no_submit_when_squeue_fails(tmp_path, monkeypatch, out, rc):
    q = _queue(tmp_path, monkeypatch)
    calls = _fake_slurm(tmp_path, monkeypatch, out, rc)
    q.ensure_next()
    assert not [c for c in calls if c[0] == "sbatch"]


def test_self_does_not_count_as_next_job(tmp_path, monkeypatch):
    q = _queue(tmp_path, monkeypatch)
    calls = _fake_slurm(tmp_path, monkeypatch, "100\n")
    q.ensure_next()
    assert len([c for c in calls if c[0] == "sbatch"]) == 1


def test_job_limit_stops_the_chain(tmp_path, monkeypatch):
    from actspace_eval.open_queue import MAX_JOBS
    q = _queue(tmp_path, monkeypatch, jobs=[{"job": str(i)} for i in range(MAX_JOBS)])
    calls = _fake_slurm(tmp_path, monkeypatch)
    q.ensure_next()
    q.ensure_next()
    assert not [c for c in calls if c[0] == "sbatch"]
    assert (tmp_path / "open_models_progress.md").read_text().count("no continuation job submitted") == 1


def test_continuation_with_nothing_to_do_exits_without_next_job(tmp_path, monkeypatch):
    q = _queue(tmp_path, monkeypatch)
    for t in q.tasks:
        t.state = "paused"
    calls = _fake_slurm(tmp_path, monkeypatch)
    (tmp_path / "_queue/DONE").write_text("x")
    (tmp_path / "open").mkdir()
    (tmp_path / "open/VST-7B-RL.md").write_text("x")
    q.plan["tiny_reported"] = ["VST-7B-RL"]
    q.run()
    assert not [c for c in calls if c[0] in ("sbatch", "scancel")]


def test_finished_queue_cancels_the_next_job(tmp_path, monkeypatch):
    q = _queue(tmp_path, monkeypatch)
    calls = _fake_slurm(tmp_path, monkeypatch, "100\n200\n")
    q.cancel_next()
    assert [c for c in calls if c[0] == "scancel"] == [["scancel", "200"]]


@pytest.mark.parametrize("state,out,rc,submits", [("pending", "", 0, 1), ("running", "", 0, 1), ("pending", "375467\n", 0, 0), ("pending", "", 1, 0), ("done", "", 0, 0), ("paused", "", 0, 0)])
def test_watchdog(tmp_path, monkeypatch, state, out, rc, submits):
    from argparse import Namespace
    from actspace_eval.open_queue import cmd_watchdog, plan_tasks, write_tsv
    (tmp_path / "_queue").mkdir()
    plan = _plan(models=["M1"], smoke_gate={}, report_dir=str(tmp_path))
    (tmp_path / "_queue/plan.json").write_text(json.dumps(plan))
    tasks = plan_tasks(plan)
    for t in tasks:
        t.state = state
    write_tsv(tmp_path / "_queue/tasks.tsv", tasks, "1")
    calls = _fake_slurm(tmp_path, monkeypatch, out, rc)
    cmd_watchdog(Namespace(root=str(tmp_path)))
    assert len([c for c in calls if c[0] == "sbatch"]) == submits
    assert (tmp_path / "open_models_progress.md").exists() == bool(submits)
    assert (tmp_path / "_queue/watchdog.log").exists()


def test_qwen38_differs_from_qwen36_only_by_name_and_path(data):
    a, b = (dict(data.ecfg["models"][m]) for m in ("Qwen3.6-27B", "Qwen3.8-27B"))
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"model", "path", "call_timeout_proxy", "report_note"}
    assert b["call_timeout_proxy"] == "Qwen3.6-27B" and "high" in b["report_note"]
    q = data.ecfg["open_models"]["queue"]
    assert q["workers_by_model"]["Qwen3.8-27B"] == q["workers_by_model"]["Qwen3.6-27B"]
    assert "Qwen3.8-27B" in q["smoke_gate"] and "Qwen3.8-27B" in q["order"]


def _q38(tmp_path, monkeypatch, state="pending"):
    """Independent queue: QUEUE_DIR points to _queue_q38; the main queue directory only holds one file that must not be touched."""
    from actspace_eval.open_queue import plan_tasks, write_tsv
    (tmp_path / "_queue").mkdir()
    (tmp_path / "_queue/plan.json").write_text("main queue")
    (tmp_path / "_queue_q38").mkdir()
    plan = _plan(models=["VST-7B-RL"], smoke_gate={}, report_dir=str(tmp_path), job_name="open_q38",
                 sbatch_args=["-J", "open_q38", "--gres=gpu:2"], final_queues=["_queue", "_queue_q38"])
    (tmp_path / "_queue_q38/plan.json").write_text(json.dumps(plan))
    tasks = plan_tasks(plan)
    for t in tasks:
        t.state = state
    write_tsv(tmp_path / "_queue_q38/tasks.tsv", tasks, "1")
    monkeypatch.setenv("QUEUE_DIR", "_queue_q38")
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setenv("QUEUE_STOP_AFTER_S", "1000")


def test_separate_queue_keeps_to_its_own_dir_and_job_name(tmp_path, monkeypatch):
    from argparse import Namespace
    from actspace_eval.open_queue import Queue, cmd_watchdog
    _q38(tmp_path, monkeypatch)
    q = Queue(tmp_path)
    q.flush()
    q.log("x")
    assert q.qdir == tmp_path / "_queue_q38" and (tmp_path / "_queue_q38/events.log").exists()
    calls = _fake_slurm(tmp_path, monkeypatch)
    q.ensure_next()
    sb = [c for c in calls if c[0] == "sbatch"]
    assert len(sb) == 1 and sb[0][2:5] == ["--dependency=afterany:100", "-J", "open_q38"] and "--gres=gpu:2" in sb[0]
    calls.clear()
    cmd_watchdog(Namespace(root=str(tmp_path)))
    assert any(c[0] == "squeue" and "open_q38" in c for c in calls)
    assert [c for c in calls if c[0] == "sbatch"] == [["sbatch", "--parsable", "-J", "open_q38", "--gres=gpu:2", "scripts/run_open_queue.sbatch"]]
    assert (tmp_path / "_queue_q38/watchdog.log").exists()
    assert sorted(x.name for x in (tmp_path / "_queue").iterdir()) == ["plan.json"] and (tmp_path / "_queue/plan.json").read_text() == "main queue"


def test_final_report_covers_both_queues(tmp_path, monkeypatch):
    from actspace_eval.open_queue import plan_tasks, write_tsv
    from actspace_eval.report_open_full import cmd_final
    n = {}
    for d, m in (("_queue", "M1"), ("_queue_q38", "M2")):
        (tmp_path / d).mkdir()
        plan = _plan(models=[m], smoke_gate={}, report_dir=str(tmp_path), jobs=[{"job": "1", "gpus": 2, "seconds": 3600}])
        (tmp_path / d / "plan.json").write_text(json.dumps(plan))
        tasks = plan_tasks(plan)
        n[d] = len(tasks)
        for t in tasks:
            t.state = "done"
        write_tsv(tmp_path / d / "tasks.tsv", tasks, "1")
    cmd_final(tmp_path, "full", tmp_path, queues=("_queue", "_queue_q38"))
    head = (tmp_path / "open_models_full.md").read_text()
    assert f"{n['_queue'] + n['_queue_q38']} tasks" in head and "Used 2 jobs" in head and "4 allocated GPU hours" in head


# ---------------- per-call 600 s limit, context overflow, extra report columns, rerunning paused tasks
def test_call_timeout_config():
    ec = load_eval_config()
    assert ec["llm"]["call_timeout_s"] == 600 and ec["models"]["Cambrian-S-7B"]["call_timeout_s"] == 600
    assert ec["models"]["Cambrian-S-7B"]["read_timeout_s"] >= 8 * 600
    assert all(ec["models"][k].get("call_timeout_s") == "auto" for k in OPEN if k != "Cambrian-S-7B")
    assert ec["llm"]["call_timeout_auto"] == {"factor": 3, "tokens": 32768, "min_calls": 50, "min_output_tokens": 100}


def test_cambrian_timeout_is_a_truncated_reply(data, tmp_path):
    """Server-side timeout: finish_reason length, timeout_truncated; not retried and not a server error; the step is an invalid reply."""
    seen = []

    def tr(path, body):
        seen.append(body)
        text = FakeVLLM().reply(body)
        cut = len(seen) == 1
        return {"id": "c", "model": "Cambrian-S-7B", "usage": {"prompt_tokens": 3000, "completion_tokens": 9, "total_tokens": 3009},
                "choices": [{"index": 0, "finish_reason": "length" if cut else "stop",
                             "message": {"role": "assistant", "content": '{"action": "forw' if cut else text}}],
                "x_timing": {"ttft_s": 0.5, "total_s": 600.0}, "x_image_tokens": [2788], "x_timeout_truncated": cut, "x_time_limit_s": 600.0}
    st = run_llm(data, "Cambrian-S-7B", "active_v2", 40, TWO[:1], tmp_path / "r", print_estimate=False, client_kwargs={"transport": tr, "sleep": nosleep})
    assert st["status"] == "complete" and seen[0]["x_time_limit_s"] == 600.0
    calls = [json.loads(l) for l in open(tmp_path / "r" / "calls.jsonl") if '"type": "call"' in l]
    assert calls[0]["timeout_truncated"] is True and calls[0]["finish_reason"] == "length" and calls[0]["retries"] == 0
    assert calls[1]["timeout_truncated"] is False
    steps = [json.loads(l) for l in open(tmp_path / "r" / "trajectories.jsonl") if '"type": "step"' in l]
    assert steps[0]["invalid"] is True
    from actspace_eval.report_open_full import run_extra
    assert run_extra(tmp_path / "r")["timeout"] == 1


def test_vllm_client_timeout_keeps_partial_reply():
    from actspace_eval.models_vllm import timed_out
    t0 = 0.0
    raw, ttft = timed_out([(1.0, {"id": "c", "choices": [{"index": 0, "delta": {"reasoning_content": "thinking"}}]})], t0)
    assert raw["choices"][0]["finish_reason"] == "length" and raw["x_timeout_truncated"] and raw["choices"][0]["message"]["reasoning_content"] == "thinking"
    raw, _ = timed_out([], t0)      # timed out while queued in vLLM, not a single token
    assert raw["choices"][0]["finish_reason"] == "length" and raw["choices"][0]["message"]["content"] == ""


def test_closed_api_timeout_is_truncated_not_retried(data):
    from actspace_eval.models import CallTimeout
    n = []

    def tr(body):
        n.append(1)
        raise CallTimeout("timeout_truncated: no reply within 600 s")
    c = OpenAIClient(data.ecfg["models"]["gpt-6.1-sol"], data.ecfg["llm"], transport=tr, sleep=nosleep)
    assert c.call_timeout_s() == 600
    comp = c.complete("SYS", [{"role": "user", "text": "hi"}], tag={"episode": "g", "t": 0})
    assert len(n) == 1 and comp.finish_reason == "length" and comp.text == "" and comp.extra["timeout_truncated"] is True


class _OverflowVLLM(FakeVLLM):
    """Active reports context overflow from step 1 on (after the model's own replies enter the history)."""

    def __init__(self, at=1, **k):
        super().__init__(think=False, **k)
        self.at = at

    def __call__(self, path, body):
        if sum(m["role"] == "assistant" for m in body["messages"]) >= self.at:
            return {"_http_status": 400, "_text": json.dumps({"error": {"message": "Input length (32870) exceeds model's maximum context length (32768)."}})}
        return super().__call__(path, body)


def test_context_overflow_ends_episode_and_run_continues(data, tmp_path):
    out = tmp_path / "a"
    st = run_llm(data, "VST-7B-RL", "active_v2", 40, TWO, out, print_estimate=False,
                 client_kwargs={"transport": _OverflowVLLM(), "http": fake_http, "sleep": nosleep})
    assert st["status"] == "complete" and st["completed"] == 2
    ends = [json.loads(l) for l in open(out / "trajectories.jsonl") if '"episode_end"' in l]
    assert [e["forced"] for e in ends] == ["context_overflow"] * 2 and all(v == 0 for e in ends for v in e["scores"].values())
    assert replay(data, out / "trajectories.jsonl")["mismatched_episodes"] == 0
    from actspace_eval.report_open_full import run_extra
    assert run_extra(out)["overflow"] == 2


def test_context_overflow_at_first_call_is_still_na(data, tmp_path):
    st = run_llm(data, "VST-7B-RL", "active_v2", 40, TWO[:1], tmp_path / "a", print_estimate=False,
                 client_kwargs={"transport": _OverflowVLLM(at=0), "http": fake_http, "sleep": nosleep})
    assert st["status"] == "not_applicable" and st["na_reason"]["kind"] == "context"


def test_requeue_file_reopens_paused_task(tmp_path, monkeypatch):
    from actspace_eval.open_queue import Queue, plan_tasks, write_tsv
    (tmp_path / "_queue").mkdir()
    plan = _plan(models=["VST-7B-RL"], smoke_gate={}, report_dir=str(tmp_path))
    (tmp_path / "_queue/plan.json").write_text(json.dumps(plan))
    tasks = plan_tasks(plan)
    for t in tasks:
        t.state = "done" if t.setting != "active_v2" else "paused"
    write_tsv(tmp_path / "_queue/tasks.tsv", tasks, "1")
    (tmp_path / "_queue/requeue.tsv").write_text("# model\tsetting\trun\nVST-7B-RL\tactive_v2\tfull_r0\n")
    monkeypatch.setenv("SLURM_JOB_ID", "200")
    monkeypatch.setenv("QUEUE_STOP_AFTER_S", "1000")
    q = Queue(tmp_path)
    assert [t.state for t in q.tasks if t.setting == "active_v2"] == ["pending"] and q.requeued == [("VST-7B-RL", "active_v2", "full_r0")]
    assert not (tmp_path / "_queue/requeue.tsv").exists() and (tmp_path / "_queue/requeue.done.200.tsv").exists()


def test_selective_score_drops_truncated_groups(tmp_path):
    from actspace_eval.report_open_full import run_extra, selective
    d = tmp_path / "r"
    d.mkdir()
    recs = [{"type": "call", "group_id": "g1", "finish_reason": "length"}, {"type": "call", "group_id": "g1", "finish_reason": "stop"},
            {"type": "call", "group_id": "g2", "finish_reason": "length", "timeout_truncated": True}]
    (d / "calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    e = run_extra(d)
    assert e["trunc_groups"] == {"g2"} and e["timeout"] == 1
    R = {"spec": {"blind": ["blind/r"]}, "runs": {"blind": [{"per_question": [{"group_id": "g1", "score": 1.0}, {"group_id": "g2", "score": 0.0}]}]}}
    rows = [{"setting": "blind", "trunc_groups": {"blind/r": e["trunc_groups"]}}]
    assert selective(R, rows)["blind"] == (2, 1, 0.5, 1.0)


# ---------------- per-call limit of vLLM models = 3 × 32768 ÷ normal decoding speed
def _speed_calls(d, n, out_tokens, seconds, ttft=1.0):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "calls.jsonl", "a") as f:
        for _ in range(n):
            f.write(json.dumps({"type": "call", "usage": {"output_tokens": out_tokens}, "latency_s": seconds + ttft, "ttft_s": ttft}) + "\n")


def test_call_limit_sources_and_auto_update(tmp_path, data):
    from actspace_eval.call_limit import CallLimit, call_speed
    llm = data.ecfg["llm"]
    assert call_speed(50, 10, 1) is None and call_speed(1000, 11, 1) == 100
    _speed_calls(tmp_path / "P" / "blind" / "full_r0", 60, 2000, 100)          # proxy model: 20 tokens/s
    m = tmp_path / "M"
    cl = CallLimit(m, {"call_timeout_proxy": "P"}, llm)
    assert cl.ensure()["stage"] == "proxy" and cl.limit_s() == round(3 * 32768 / 20, 1)
    _speed_calls(m / "blind" / "smoke1_r0", 5, 1000, 25)                        # its own smoke test: 40 tokens/s
    cl2 = CallLimit(m, {"call_timeout_proxy": "P"}, llm)
    assert cl2.ensure()["stage"] == "smoke" and cl.limit_s() == round(3 * 32768 / 40, 1)   # other processes re-read by modification time
    for i in range(49):
        cl.record(10.0)
    assert cl.state["stage"] == "smoke"
    cl2.record(10.0)                                                              # 50th call of the evaluation run: updated once automatically
    assert cl.limit_s() == round(3 * 32768 / 10, 1) and cl.state["stage"] == "formal" and len(cl.state["history"]) == 3
    cl.record(99.0)
    assert len(cl.samples.read_text().split()) == 50                              # no more samples recorded after the update
    smoke = CallLimit(tmp_path / "S", {}, llm, formal=False)
    smoke.ensure()
    smoke.record(5.0)
    assert smoke.state["stage"] == "none" and smoke.limit_s() is None and not smoke.samples.exists()


def test_vllm_run_writes_limit_to_env_and_calls(data, tmp_path):
    root = tmp_path / "res"
    _speed_calls(root / "VST-7B-RL" / "blind" / "smoke_r0", 3, 1000, 50)          # smoke: 20 tokens/s -> 4915.2 s
    out = root / "VST-7B-RL" / "active_v2" / "full_r0"
    st = run_llm(data, "VST-7B-RL", "active_v2", 40, TWO[:1], out, print_estimate=False, env_info={},
                 client_kwargs={"transport": FakeVLLM(think=False), "http": fake_http})
    assert st["status"] == "complete"
    env = json.loads((out / "env.json").read_text())
    assert env["call_timeout"]["limit_s"] == 4915.2 and env["call_timeout"]["stage"] == "smoke" and env["call_timeout_end"]["limit_s"] == 4915.2
    calls = [json.loads(l) for l in open(out / "calls.jsonl") if '"type": "call"' in l]
    assert all(c["call_timeout_s"] == 4915.2 for c in calls)


def test_template_table_is_percent_one_decimal():
    from actspace_eval.report_eval import pct
    assert pct({"mean": 0.29357, "lo": 0.2201, "hi": 0.36}) == "29.4 [22.0, 36.0]" and pct(None) == "—"


# ---------------------------------------------------------------- multiple instances and scheduling
def test_control_file_spreads_groups_over_instances_once_each(data, tmp_path, monkeypatch):
    """The control file starts with no instances and two are added later: each group runs exactly once, instances.jsonl records the
    instance of each group, env.json records the control file."""
    import threading
    monkeypatch.setenv("ACTSPACE_CONTROL_POLL_S", "0.2")
    fake = FakeVLLM(think=False)
    ctl = tmp_path / "ctl.json"
    ctl.write_text(json.dumps({"workers": {"static": 1, "active_v2": 1}, "servers": []}))
    gids = ["arkit_g_000000", "arkit_g_000072", "scannetpp_g_000009", "scannetpp_g_000016"]
    servers = [{"id": "VST-7B-RL@8100", "url": "http://127.0.0.1:8100", "gpus": "0"}, {"id": "VST-7B-RL@8300", "url": "http://127.0.0.1:8300", "gpus": "1"}]
    threading.Timer(0.5, lambda: ctl.write_text(json.dumps({"workers": {"static": 1, "active_v2": 2}, "servers": servers}))).start()
    out = tmp_path / "VST-7B-RL" / "active_v2" / "full_r0"
    st = run_llm(data, "VST-7B-RL", "active_v2", 40, gids, out, print_estimate=False, workers=8,
                 client_kwargs={"transport": fake, "http": fake_http}, env_info={"model_path": "x"}, control=ctl)
    assert st["status"] == "complete" and st["completed"] == 4
    eps = complete_episodes(out / "trajectories.jsonl")
    assert sorted(e[0]["group_id"] for e in eps) == sorted(gids)
    inst = [json.loads(l) for l in open(out / "instances.jsonl")]
    assert sorted(x["group_id"] for x in inst) == sorted(gids) and {x["instance"] for x in inst} <= {s["id"] for s in servers}
    env = json.loads((out / "env.json").read_text())
    assert env["control"]["path"] == str(ctl) and sum(v["episodes"] for v in env["instances_used"].values()) == 4


def test_per_instance_workers_and_bad_control(tmp_path):
    from actspace_eval.llm_run import per_instance, read_control
    assert per_instance({"static": 3, "active_v2": 8}, "active_v2") == 8 and per_instance({"static": 3, "active_v2": 8}, "passive_40") == 3
    assert per_instance({"static": 3, "active_v2": 8, "blind": 5}, "blind") == 5 and per_instance(2, "blind") == 2
    (tmp_path / "c.json").write_text("{bad")
    assert read_control(tmp_path / "c.json", "blind") is None and read_control(tmp_path / "missing.json", "blind") is None


def _sched(tmp_path, monkeypatch, models, free="0,1,2,3"):
    from actspace_eval.open_queue import Queue
    (tmp_path / "_queue").mkdir(exist_ok=True)
    (tmp_path / "_queue/plan.json").write_text(json.dumps(_plan(models=models, smoke_gate={}, report_dir=str(tmp_path))))
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setenv("QUEUE_STOP_AFTER_S", "100000")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", free)
    q = Queue(tmp_path)
    started = []
    monkeypatch.setattr(q, "model_thread", lambda m, g, p: started.append(("new", m, tuple(g))))
    monkeypatch.setattr(q, "replica_thread", lambda m, g, p, ev: started.append(("more", m, tuple(g))))
    return q, started


def _running(q, model, gpus_left, done=0):
    import threading
    q.running.add(model)
    q.launched.add(model)
    q.ctl[model] = []
    q.n_inst[model] = 1
    q.batch_end[model] = threading.Event()
    for t in q.of_model(model, "full"):
        t.state, t.done, t.total = "running", done, 234
    q.free = list(gpus_left)


def test_free_gpus_go_first_to_earlier_models_in_order(tmp_path, monkeypatch):
    """Order 27B (2 GPUs) -> Cambrian (1 GPU) -> gemma (4 GPUs): one freed GPU adds a Cambrian instance, two freed GPUs add a
    27B instance."""
    import time as _t
    q, started = _sched(tmp_path, monkeypatch, ["Qwen3.6-27B", "Cambrian-S-7B", "gemma-4-31B-it"])
    # the order is fixed by the test and does not follow queue.order in configs/eval.yaml
    monkeypatch.setattr(q, "reload_config", lambda: None)
    q.q = {**q.q, "order": ["Qwen3.6-27B", "Cambrian-S-7B", "gemma-4-31B-it"]}
    _running(q, "Qwen3.6-27B", [])
    _running(q, "Cambrian-S-7B", ["3"])
    q.launch()
    _t.sleep(0.1)
    assert started == [("more", "Cambrian-S-7B", ("3",))] and q.n_inst["Cambrian-S-7B"] == 2
    started.clear()
    q.free = ["2", "3"]
    q.launch()
    _t.sleep(0.1)
    assert started == [("more", "Qwen3.6-27B", ("2", "3"))]


def test_no_extra_instance_when_little_work_left_or_gpus_held(tmp_path, monkeypatch):
    import time as _t
    q, started = _sched(tmp_path, monkeypatch, ["Cambrian-S-7B"])
    _running(q, "Cambrian-S-7B", ["3"], done=233)           # 1 group left, 1 episode in flight
    q.launch()
    assert started == []
    for t in q.of_model("Cambrian-S-7B", "full"):
        t.done = 100
    (tmp_path / "_queue/hold_gpus").write_text("3\n")
    q.launch()
    assert started == []
    (tmp_path / "_queue/hold_gpus").unlink()
    q.launch()
    _t.sleep(0.1)
    assert started == [("more", "Cambrian-S-7B", ("3",))]
    q.replica_fail["Cambrian-S-7B"] = 2
    started.clear()
    q.free = ["2"]
    q.launch()
    assert started == []


def test_new_model_launches_when_it_fits(tmp_path, monkeypatch):
    import time as _t
    q, started = _sched(tmp_path, monkeypatch, ["SenseNova-SI-1.2-InternVL3-8B", "Qwen3.6-27B"])
    _running(q, "Qwen3.6-27B", ["3"])
    q.launch()
    _t.sleep(0.1)
    assert started == [("new", "SenseNova-SI-1.2-InternVL3-8B", ("3",))]


def test_control_file_and_workers_override(tmp_path, monkeypatch):
    from actspace_eval.open_queue import Server
    q, _ = _sched(tmp_path, monkeypatch, ["Qwen3.6-27B"])
    q.ctl["Qwen3.6-27B"] = [Server(q, "Qwen3.6-27B", ["0", "1"], 8106), Server(q, "Qwen3.6-27B", ["2", "3"], 8300, tag="_p8300")]
    q.write_control("Qwen3.6-27B")
    doc = json.loads(q.ctl_path("Qwen3.6-27B").read_text())
    assert doc["workers"] == {"static": 3, "active_v2": 8} and [s["id"] for s in doc["servers"]] == ["Qwen3.6-27B@8106", "Qwen3.6-27B@8300"]
    assert doc["servers"][1]["info"].endswith("server_Qwen3.6-27B_p8300.json")
    (tmp_path / "_queue/workers.json").write_text(json.dumps({"Qwen3.6-27B": {"active_v2": 10}}))
    q.sync_controls()
    assert json.loads(q.ctl_path("Qwen3.6-27B").read_text())["workers"] == {"static": 3, "active_v2": 10}


def test_spawn_passes_control_only_for_full_tasks(tmp_path, monkeypatch):
    import subprocess as sp
    from actspace_eval.open_queue import Server
    q, _ = _sched(tmp_path, monkeypatch, ["Qwen3.6-27B"])
    cmds = []

    class P:
        def __init__(self, cmd, **k):
            cmds.append(cmd)

        def wait(self):
            return 0
    monkeypatch.setattr(sp, "Popen", P)
    srv = Server(q, "Qwen3.6-27B", ["0", "1"], 8106)
    t = q.of_model("Qwen3.6-27B", "full")[0]
    q.spawn(t, srv, ["g"], False)
    assert "--control" not in cmds[-1]
    q.ctl["Qwen3.6-27B"] = [srv]
    q.spawn(t, srv, ["g"], False)
    assert cmds[-1][cmds[-1].index("--control") + 1].endswith("ctl_Qwen3.6-27B.json")


@pytest.mark.parametrize("out,rc,err,cancel", [("batch\nextern\nopen_takeover\n", 0, "", False), ("batch\nextern\n", 0, "", True),
                                                ("", 0, "", False), ("", 1, "slurm_load_jobs error: Invalid job id specified", False)])
def test_takeover_guard(tmp_path, monkeypatch, out, rc, err, cancel):
    import subprocess as sp
    from types import SimpleNamespace
    from actspace_eval.open_queue import takeover_guard
    (tmp_path / "takeover.json").write_text(json.dumps({"job": "375467", "old_pid": 1}))
    calls = []

    def run(cmd, **k):
        calls.append(cmd)
        return SimpleNamespace(returncode=rc, stdout=out, stderr=err)
    monkeypatch.setattr(sp, "run", run)
    msg = takeover_guard(tmp_path)
    assert bool([c for c in calls if c[0] == "scancel"]) == cancel
    assert (tmp_path / "takeover.json").exists() == ("open_takeover" in out)
    assert msg
