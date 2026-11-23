#!/usr/bin/env python
"""OpenAI-compatible (Chat Completions, non-streaming) server for Cambrian-S-7B.

Follows the official inference path of the Cambrian-S repo (commit 058b1b4):
  * model loading      : cambrian/model/builder.py::load_pretrained_model (replicated, bf16 instead of fp16)
  * image preprocessing: cambrian/mm_utils.py::process_images
                         - exactly 1 image in the request  -> "anyres" (snapshot + up to 9 sub-crops)
                         - >1 images                       -> use_pad=True (expand2square + resize 384)
                         (same dispatch as lmms-eval/lmms_eval/models/simple/cambrians.py:261-271)
  * placeholder text   : "<image>" + "\n" (no "\n" if the following text already starts with "\n"),
                         lmms-eval cambrians.py:296-312
  * conversation       : cambrian/conversation.py conv_templates["qwen_2"] (CHATML)
  * tokenisation       : cambrian/mm_utils.py::tokenizer_image_token (IMAGE_TOKEN_INDEX=-200)
  * embedding merge    : CambrianMetaForCausalLM.prepare_inputs_labels_for_multimodal_for_generation
  * generation         : HF GenerationMixin.generate(inputs_embeds=...), as CambrianQwenForCausalLM.generate
See CAMBRIAN_SERVER.md for details.
"""
import argparse
import base64
import io
import math
import os
from pathlib import Path
import sys
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

CAMBRIAN_REPO = os.environ.get("CAMBRIAN_REPO", str(Path(__file__).resolve().parents[2] / "third_party" / "cambrian-s"))
if CAMBRIAN_REPO not in sys.path:
    sys.path.insert(0, CAMBRIAN_REPO)

import torch
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
from transformers import RepetitionPenaltyLogitsProcessor

from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates
from cambrian.mm_utils import process_images, tokenizer_image_token
from cambrian.model.cambrian_arch import unpad_image

MODEL_ID = "Cambrian-S-7B"
MAX_MODEL_LEN = 32768
EOS_IDS = [151645, 151643]  # generation_config.json
PAD_ID = 151643
GEN_DEFAULTS = dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05)  # generation_config.json

STATE: Dict[str, Any] = {}
LOCK = threading.Lock()


class BadRequest(Exception):
    pass


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ----------------------------------------------------------------------------------------------
# Model loading: replica of cambrian/model/builder.py::load_pretrained_model with bf16
# (builder hard-codes torch.float16 at builder.py:44,51-52,73; user requirement is bf16).
# ----------------------------------------------------------------------------------------------
def load_model(model_path: str, device: str = "cuda:0"):
    from cambrian.model.language_model.cambrian_qwen2 import CambrianQwenForCausalLM
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)            # builder.py:50
    model = CambrianQwenForCausalLM.from_pretrained(                                   # builder.py:53-57
        model_path, low_cpu_mem_usage=True, device_map=device, torch_dtype=dtype)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)          # builder.py:60-65
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    for vt in model.get_vision_tower_aux_list():                                        # builder.py:67-73
        if not vt.is_loaded:
            vt.load_model(device_map=device)
        vt.to(device=device, dtype=dtype)
    image_processor = [vt.image_processor for vt in model.get_vision_tower_aux_list()]
    # lmms-eval cambrians.py:141-148 with the values used by evaluate_all_in_one.sh
    # (miv_token_len=64, si_token_len=729, image_aspect_ratio=anyres, anyres_max_subimages=9)
    model.config.miv_token_len = 64
    model.config.si_token_len = 729
    model.config.image_aspect_ratio = "anyres"
    model.config.anyres_max_subimages = 9
    model.config.video_max_frames = 32
    model.config.video_fps = 1
    model.config.video_force_sample = False
    model.config.add_time_instruction = False
    model.eval()
    return tokenizer, model, image_processor


def vision_weight_check(model, model_path):
    """Compare the HF-downloaded SigLIP2 tower (what the official code actually uses) against the copy
    stored inside the Cambrian-S checkpoint (which the official loader ignores because of delay_load)."""
    try:
        import json
        from safetensors import safe_open
        idx = json.load(open(os.path.join(model_path, "model.safetensors.index.json")))["weight_map"]
        sd = model.get_vision_tower_aux_list()[0].vision_tower.state_dict()
        out = {}
        for k in ["vision_model.embeddings.patch_embedding.weight", "vision_model.encoder.layers.10.mlp.fc1.weight",
                  "vision_model.post_layernorm.weight"]:
            ck = "model.vision_tower_aux_list.0.vision_tower." + k
            if ck not in idx or k not in sd:
                out[k] = "missing"
                continue
            with safe_open(os.path.join(model_path, idx[ck]), "pt") as f:
                t = f.get_tensor(ck)
            out[k] = float((t.to(sd[k].device, torch.float32) - sd[k].float()).abs().max())
        n_layers = len(model.get_vision_tower_aux_list()[0].vision_tower.vision_model.encoder.layers)
        n_ck_layers = len({k.split(".")[6] for k in idx if ".encoder.layers." in k and "vision_tower" in k})
        return {"max_abs_diff_vs_checkpoint_copy": out, "hf_tower_layers": n_layers, "ckpt_tower_layers": n_ck_layers}
    except Exception as e:  # informational only
        return {"error": repr(e)}


# ----------------------------------------------------------------------------------------------
# JSON-mode constrained decoding (lm-format-enforcer, generic {"type":"object"} schema)
# ----------------------------------------------------------------------------------------------
def build_json_tokenizer_data(model_path, vocab_size):
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
    from lmformatenforcer.integrations.transformers import _build_regular_tokens_list, _decode_function
    import functools
    # fast tokenizer only for speed of building the token table (identical vocab / decoding)
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    regular = _build_regular_tokens_list(tok, vocab_size)
    return TokenEnforcerTokenizerData(regular, functools.partial(_decode_function, tok), EOS_IDS, False, vocab_size)


class JsonObjectLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer_data):
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.tokenenforcer import TokenEnforcer
        self.enforcer = TokenEnforcer(tokenizer_data, JsonSchemaParser({"type": "object"}))

    def __call__(self, input_ids, scores):
        # input_ids = generated tokens only (generation runs from inputs_embeds)
        allowed = self.enforcer.get_allowed_tokens(input_ids[0].tolist()).allowed_tokens
        mask = torch.full_like(scores, float("-inf"))
        if len(allowed) == 0:
            allowed = EOS_IDS
        mask[:, torch.tensor(allowed, device=scores.device, dtype=torch.long)] = 0
        return scores + mask


class RawLogprobRecorder(LogitsProcessor):
    """First logits processor in the chain: sees the *raw* model logits (before repetition penalty,
    JSON mask, temperature, top-k, top-p).  Records TTFT and, if asked, raw log-softmax logprobs."""

    def __init__(self, t0, want_logprobs, top_n):
        self.t0 = t0
        self.ttft = None
        self.want = want_logprobs
        self.top_n = top_n
        self.prev_lp = None
        self.records = []  # list of (sampled_lp, top_vals, top_idx) filled lazily

    def _flush(self, sampled_id):
        lp = self.prev_lp
        top = None
        if self.top_n > 0:
            v, i = torch.topk(lp, self.top_n)
            top = (v.tolist(), i.tolist())
        self.records.append((float(lp[sampled_id]), sampled_id, top))
        self.prev_lp = None

    def __call__(self, input_ids, scores):
        if self.ttft is None:
            torch.cuda.synchronize()
            self.ttft = time.time() - self.t0
        if self.want:
            if self.prev_lp is not None:
                self._flush(int(input_ids[0, -1]))
            self.prev_lp = torch.log_softmax(scores[0].float(), dim=-1)
        return scores

    def finalize(self, last_token_id):
        if self.want and self.prev_lp is not None:
            self._flush(int(last_token_id))


class StopStringCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stops):
        self.tok = tokenizer
        self.stops = stops
        self.hit = None

    def __call__(self, input_ids, scores, **kw):
        text = self.tok.decode(input_ids[0, -64:], skip_special_tokens=True)
        for s in self.stops:
            if s in text:
                self.hit = s
                return True
        return False


class TimeLimitCriteria(StoppingCriteria):
    """Wall-clock limit per call. Timed from when the server starts processing the request (t0 of run_chat, excluding time
    queued before the serialization lock); when it expires, generation stops and the reply is treated as truncated
    (finish_reason = length, x_timeout_truncated = true)."""

    def __init__(self, t0, limit_s):
        self.t0, self.limit = t0, float(limit_s)
        self.hit = False

    def __call__(self, input_ids, scores, **kw):
        if time.time() - self.t0 >= self.limit:
            self.hit = True
            return True
        return False


# ----------------------------------------------------------------------------------------------
# Request -> prompt
# ----------------------------------------------------------------------------------------------
def decode_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        try:
            b64 = url.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception as e:
            raise BadRequest(f"could not decode image data URL: {e}")
    if url.startswith("file://") or os.path.isabs(url):
        return Image.open(url[7:] if url.startswith("file://") else url).convert("RGB")
    raise BadRequest("only data: URLs (base64) or local file paths are supported for image_url")


def content_to_text(content, images: List[Image.Image], allow_images: bool) -> str:
    """Render OpenAI content into Cambrian text with <image> placeholders at the image positions."""
    if content is None:
        return ""
    if isinstance(content, str):
        if DEFAULT_IMAGE_TOKEN in content:
            raise BadRequest("literal '<image>' in text is not allowed")
        return content
    if not isinstance(content, list):
        raise BadRequest("content must be a string or a list of parts")
    # first normalise parts into a list of ("text", str) / ("image", PIL)
    parts = []
    for p in content:
        t = p.get("type")
        if t == "text":
            txt = p.get("text", "")
            if DEFAULT_IMAGE_TOKEN in txt:
                raise BadRequest("literal '<image>' in text is not allowed")
            if parts and parts[-1][0] == "text":  # adjacent text parts: join with newline (vLLM convention)
                parts[-1] = ("text", parts[-1][1] + "\n" + txt)
            else:
                parts.append(("text", txt))
        elif t in ("image_url", "image"):
            if not allow_images:
                raise BadRequest("images are only supported in user messages")
            iu = p.get("image_url", p.get("image"))
            url = iu.get("url") if isinstance(iu, dict) else iu
            parts.append(("image", decode_image(url)))
        else:
            raise BadRequest(f"unsupported content part type: {t}")
    out = ""
    for i, (kind, val) in enumerate(parts):
        if kind == "text":
            out += val
        else:
            images.append(val)
            # lmms-eval cambrians.py:296-306: "<image>" + "\n" unless next text already starts with "\n"
            nxt = parts[i + 1] if i + 1 < len(parts) else None
            if nxt is not None and nxt[0] == "text" and nxt[1].startswith("\n"):
                out += DEFAULT_IMAGE_TOKEN
            else:
                out += DEFAULT_IMAGE_TOKEN + "\n"
    return out


def build_prompt(messages):
    if not isinstance(messages, list) or not messages:
        raise BadRequest("messages must be a non-empty list")
    conv = conv_templates["qwen_2"].copy()
    images: List[Image.Image] = []
    idx = 0
    sys_texts = []
    while idx < len(messages) and messages[idx].get("role") == "system":
        sys_texts.append(content_to_text(messages[idx].get("content"), images, False))
        idx += 1
    if sys_texts:
        conv.system = "<|im_start|>system\n" + "\n".join(sys_texts)
    for m in messages[idx:]:
        role = m.get("role")
        if role == "user":
            conv.append_message(conv.roles[0], content_to_text(m.get("content"), images, True))
        elif role == "assistant":
            conv.append_message(conv.roles[1], content_to_text(m.get("content"), images, False))
        elif role == "system":  # non-leading system message: rendered as its own CHATML turn (Qwen2.5 template does the same)
            conv.append_message("<|im_start|>system", content_to_text(m.get("content"), images, False))
        else:
            raise BadRequest(f"unsupported role: {role}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), images


def image_token_counts(image_sizes, cfg):
    """Number of LLM positions each image occupies, mirroring cambrian_arch.py:571-641 shape logic."""
    side = int(cfg.si_token_len ** 0.5)  # 27 (729 SigLIP2 tokens, no interpolation)
    counts = []
    for s in image_sizes:
        if len(s) == 2:  # pad path (multi-image): unpad 27x27 grid, + newline per row
            t = unpad_image(torch.zeros(1, side, side, 1), s)
            counts.append(t.shape[1] * (t.shape[2] + 1))
        elif len(s) == 4:  # anyres: snapshot + (H*27 x W*27) grid, both unpadded, + newline per row
            snap = unpad_image(torch.zeros(1, side, side, 1), s[:2])
            grid = unpad_image(torch.zeros(1, s[2] * side, s[3] * side, 1), s[:2])
            counts.append(snap.shape[1] * (snap.shape[2] + 1) + grid.shape[1] * (grid.shape[2] + 1))
        else:
            raise RuntimeError(f"unexpected image size entry {s}")
    return counts


# ----------------------------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------------------------
def err(status, msg, typ="BadRequestError"):
    return JSONResponse(status_code=status, content={"error": {"message": msg, "type": typ, "code": status}})


def _num(v, default, cast=float):
    return default if v is None else cast(v)


def run_chat(body: Dict[str, Any], t_recv: float):
    tok, model, improc = STATE["tokenizer"], STATE["model"], STATE["image_processor"]
    t0 = time.time()
    n = body.get("n", 1)
    if n not in (None, 1):
        raise BadRequest("only n=1 is supported")
    prompt, images = build_prompt(body.get("messages"))

    temperature = _num(body.get("temperature"), GEN_DEFAULTS["temperature"])
    top_p = _num(body.get("top_p"), GEN_DEFAULTS["top_p"])
    top_k = _num(body.get("top_k"), GEN_DEFAULTS["top_k"], int)
    if top_k is not None and top_k < 0:
        top_k = 0  # vLLM uses -1 for "disabled"; HF uses 0
    rep = _num(body.get("repetition_penalty"), GEN_DEFAULTS["repetition_penalty"])
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    seed = body.get("seed")
    want_lp = bool(body.get("logprobs", False))
    top_n = int(body.get("top_logprobs") or 0) if want_lp else 0
    if top_n > 20:
        raise BadRequest("top_logprobs must be <= 20")
    rf = body.get("response_format") or {}
    json_mode = isinstance(rf, dict) and rf.get("type") in ("json_object",)
    if isinstance(rf, dict) and rf.get("type") not in (None, "text", "json_object"):
        raise BadRequest(f"response_format type {rf.get('type')} not supported (only text / json_object)")
    stops = body.get("stop")
    if isinstance(stops, str):
        stops = [stops]
    stops = [s for s in (stops or []) if s]

    # ---- images (lmms-eval cambrians.py:258-271) ----
    model.config.image_aspect_ratio = "anyres"  # process_images mutates this when >1 image (mm_utils.py:266-267); reset per request
    if len(images) == 1:
        vis, vis_sizes = process_images(images, improc, model.config)
    elif len(images) > 1:
        vis, vis_sizes = process_images(images, improc, model.config, use_pad=True)
    else:
        vis, vis_sizes = None, None
    model.config.image_aspect_ratio = "anyres"

    input_ids = tokenizer_image_token(prompt, tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    n_placeholders = int((input_ids == IMAGE_TOKEN_INDEX).sum())
    if n_placeholders != len(images):
        raise BadRequest(f"internal: {n_placeholders} placeholders vs {len(images)} images")
    img_counts = image_token_counts(vis_sizes, model.config) if vis is not None else []
    text_tokens = input_ids.shape[1] - n_placeholders
    prompt_tokens = text_tokens + sum(img_counts)
    if prompt_tokens >= MAX_MODEL_LEN:
        raise BadRequest(f"This model's maximum context length is {MAX_MODEL_LEN} tokens. "
                         f"However, your request has {prompt_tokens} input tokens.")
    room = MAX_MODEL_LEN - prompt_tokens
    max_new = room if max_tokens is None else max(1, min(int(max_tokens), room))

    if seed is not None:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))

    torch.cuda.reset_peak_memory_stats()
    prefill_mode = body.get("x_prefill_logits", STATE["prefill_logits"])
    STATE["last_only"] = (prefill_mode == "last")

    with torch.inference_mode():
        # == CambrianQwenForCausalLM.generate (cambrian_qwen2.py:495-527) ==
        if vis is not None:
            vis = [v.to(device="cuda", dtype=torch.bfloat16) for v in vis]  # lmms-eval casts to model dtype (.half()); model is bf16 here
            _, position_ids, attention_mask, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal_for_generation(
                input_ids, None, None, None, None, vis, image_sizes=vis_sizes)
        else:
            position_ids, attention_mask = None, None
            inputs_embeds = model.get_model().embed_tokens(input_ids)
        assert inputs_embeds.shape[1] == prompt_tokens, (inputs_embeds.shape, prompt_tokens)

        rec = RawLogprobRecorder(t0, want_lp, top_n)
        procs = [rec]
        if rep is not None and rep != 1.0:
            procs.append(RepetitionPenaltyLogitsProcessor(rep))
        if json_mode:
            procs.append(JsonObjectLogitsProcessor(STATE["json_tokdata"]))
        stop_crit = StopStringCriteria(tok, stops) if stops else None
        do_sample = temperature > 0
        gen_kwargs = dict(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids,
            use_cache=True, max_new_tokens=max_new, eos_token_id=EOS_IDS, pad_token_id=PAD_ID,
            repetition_penalty=1.0,  # applied by our processor (after the raw-logprob recorder)
            logits_processor=LogitsProcessorList(procs), num_beams=1, do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p if top_p is not None else 1.0, top_k=top_k or 0)
        else:
            gen_kwargs.update(temperature=None, top_p=None, top_k=None)
        limit = body.get("x_time_limit_s")
        time_crit = TimeLimitCriteria(t0, limit) if limit else None
        crits = [c for c in (stop_crit, time_crit) if c is not None]
        if crits:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(crits)
        out_ids = super(type(model), model).generate(**gen_kwargs)
        torch.cuda.synchronize()
    t_end = time.time()

    gen = out_ids[0].tolist()
    rec.finalize(gen[-1] if gen else EOS_IDS[0])
    if gen and gen[-1] in EOS_IDS:
        finish = "stop"
        content_ids = gen[:-1]
    elif stop_crit is not None and stop_crit.hit is not None:
        finish = "stop"
        content_ids = gen
    else:
        finish = "length"
        content_ids = gen
    text = tok.decode(content_ids, skip_special_tokens=True)
    if stop_crit is not None and stop_crit.hit is not None and stop_crit.hit in text:
        text = text[: text.index(stop_crit.hit)]

    choice = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
    if want_lp:
        lp_content = []
        for (lp, tid, top) in rec.records[: len(gen)]:
            item = {"token": tok.decode([tid]), "logprob": lp,
                    "bytes": list(tok.decode([tid]).encode("utf-8"))}
            if top is not None:
                item["top_logprobs"] = [{"token": tok.decode([i]), "logprob": v, "bytes": list(tok.decode([i]).encode("utf-8"))}
                                        for v, i in zip(*top)]
            else:
                item["top_logprobs"] = []
            lp_content.append(item)
        choice["logprobs"] = {"content": lp_content}
    else:
        choice["logprobs"] = None

    resp = {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [choice],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(gen), "total_tokens": prompt_tokens + len(gen)},
        "x_timing": {"ttft_s": rec.ttft, "total_s": t_end - t0, "queue_s": t0 - t_recv},
        "x_image_tokens": img_counts,
        "x_num_images": len(images),
        "x_image_mode": ("anyres" if len(images) == 1 else "pad") if images else None,
        "x_max_tokens_effective": max_new,
        "x_gpu_peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "x_logprobs_semantics": "raw" if want_lp else None,
    }
    if json_mode:
        resp["x_json_mode"] = "enforced"
    if time_crit is not None:
        resp["x_time_limit_s"] = time_crit.limit
        resp["x_timeout_truncated"] = bool(time_crit.hit and finish == "length")
    if body.get("return_prompt_text"):
        resp["x_prompt_text"] = prompt
    STATE["global_peak"] = max(STATE.get("global_peak", 0.0), resp["x_gpu_peak_mem_gib"])
    return resp


# ----------------------------------------------------------------------------------------------
app = FastAPI()


@app.get("/health")
def health():
    if not STATE.get("ready"):
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "nyu-visionx",
                                        "max_model_len": MAX_MODEL_LEN}]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    t_recv = time.time()
    try:
        body = await request.json()
    except Exception:
        return err(400, "invalid JSON body")
    if body.get("stream"):
        return err(400, "streaming not supported")
    if not STATE.get("ready"):
        return err(503, "model loading", "ServiceUnavailable")

    def work():
        with LOCK:
            return run_chat(body, t_recv)

    import anyio
    try:
        resp = await anyio.to_thread.run_sync(work)
    except BadRequest as e:
        return err(400, str(e))
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return err(500, f"CUDA out of memory: {e}", "InternalServerError")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return err(500, repr(e), "InternalServerError")
    return resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("CAMBRIAN_MODEL"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prefill-logits", choices=["last", "all"], default="last",
                    help="'all' = official behaviour (lm_head over every prompt position, fp32 upcast); "
                         "'last' = only the last position's logits are computed during prefill (same next-token logits, far less memory)")
    args = ap.parse_args()
    STATE["prefill_logits"] = args.prefill_logits
    STATE["last_only"] = args.prefill_logits == "last"

    t = time.time()
    tok, model, improc = load_model(args.model_path)
    STATE.update(tokenizer=tok, model=model, image_processor=improc)
    log(f"model loaded in {time.time()-t:.1f}s; attn_impl={getattr(model.config, '_attn_implementation', None)}; "
        f"vocab(after resize)={model.get_input_embeddings().weight.shape[0]}; len(tokenizer)={len(tok)}; "
        f"dtype={next(model.parameters()).dtype}")
    log("vision tower check:", vision_weight_check(model, args.model_path))

    def lm_head_pre_hook(module, args_):
        h = args_[0]
        if STATE.get("last_only") and h.dim() == 3 and h.shape[1] > 1:
            return (h[:, -1:, :],)
        return None
    model.lm_head.register_forward_pre_hook(lm_head_pre_hook)

    t = time.time()
    STATE["json_tokdata"] = build_json_tokenizer_data(args.model_path, len(tok))
    log(f"json enforcer tokenizer data built in {time.time()-t:.1f}s")
    log(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)} "
        f"mem_after_load={torch.cuda.memory_allocated()/2**30:.2f}GiB")
    STATE["ready"] = True
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
