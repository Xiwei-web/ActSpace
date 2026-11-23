"""Check the format-adapted chat templates against real requests. Run it in the vLLM environment (it uses transformers' own
template rendering function):

    <actspace-vllm>/bin/python scripts/check_chat_templates.py <run dir> [<run dir> …]

For every call in each run's calls.jsonl (including the multi-turn Active calls), the recorded request is converted to the vLLM
openai format (text {"type": "text"}, images {"type": "image"}) and rendered with both the official and the new template:
- requests without images: the new template is identical to the official one (LLaVA with list content; MiniCPM with the string
  content the official template accepts);
- requests with images: LLaVA is identical to the official template once <image> is removed, and the text before each <image>
  matches the text before that image in the request; MiniCPM is identical to the official template rendering "the string built by
  the official chat() (images replaced by (<image>./</image>), joined with newlines)".
"""
import json
import sys
from collections import Counter
from pathlib import Path

from transformers.utils.chat_template_utils import render_jinja_template

T = Path(__file__).resolve().parents[1] / "configs" / "chat_templates"
TPL = {k: (T / f"official/{a}").read_text() for k, a in (("llava_off", "llava_onevision_7b.jinja"), ("minicpm_off", "minicpm_v_4_5.jinja"))}
TPL["llava_new"] = (T / "llava_onevision_interleaved.jinja").read_text()
TPL["minicpm_new"] = (T / "minicpm_v_interleaved.jinja").read_text()


def rend(tpl, msgs, **kw):
    out, _ = render_jinja_template(conversations=[msgs], chat_template=TPL[tpl], add_generation_prompt=True, **kw)
    return out[0]


def vllm_msgs(req):
    out = []
    for m in req["messages"]:
        c = m["content"]
        parts = [{"type": "text", "text": c}] if isinstance(c, str) else [
            {"type": "image"} if q["type"] == "image" else {"type": "text", "text": q["text"]} for q in c]
        out.append({"role": m["role"], "content": parts})
    return out


def join(parts):
    return "\n".join("(<image>./</image>)" if q["type"] == "image" else q["text"] for q in parts)


def check(req) -> dict:
    msgs = vllm_msgs(req)
    n_img = sum(q["type"] == "image" for m in msgs for q in m["content"])
    res = {}
    # LLaVA
    new, off = rend("llava_new", msgs), rend("llava_off", msgs)
    ok = new.replace("<image>", "") == off.replace("<image>", "") and new.count("<image>") == n_img
    if n_img == 0:
        ok = ok and new == off
    else:   # the text right before each placeholder = the nearest text part before that image in the request
        pieces, prev, k = new.split("<image>"), [], 0
        for m in msgs:
            last = None
            for q in m["content"]:
                if q["type"] == "text" and q["text"].strip():
                    last = q["text"]
                elif q["type"] == "image":
                    prev.append(last)
        ok = ok and all(p is None or pieces[i].rstrip("\n").endswith(p.rstrip("\n")) or pieces[i].endswith(p) or pieces[i] == "" for i, p in enumerate(prev))
    res["llava"] = ok
    # MiniCPM (thinking on, as in the evaluation setting)
    kw = {"enable_thinking": True}
    new = rend("minicpm_new", msgs, **kw)
    off = rend("minicpm_off", [{"role": m["role"], "content": join(m["content"])} for m in msgs], **kw)
    res["minicpm"] = new == off and new.count("(<image>./</image>)") == n_img
    res["image"] = n_img > 0
    return res


def main():
    tot = Counter()
    for d in sys.argv[1:]:
        c = Counter()
        for line in open(Path(d) / "calls.jsonl"):
            r = json.loads(line)
            if r.get("type") != "call":
                continue
            x = check(r["request"])
            kind = "with images" if x["image"] else "text only"
            c[f"{kind} calls"] += 1
            c[f"{kind} LLaVA identical"] += x["llava"]
            c[f"{kind} MiniCPM identical"] += x["minicpm"]
        print(d, dict(c), flush=True)
        tot.update(c)
    print("total", dict(tot))
    bad = any(tot[f"{k} calls"] != tot[f"{k} {m} identical"] for k in ("with images", "text only") for m in ("LLaVA", "MiniCPM") if tot[f"{k} calls"])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
