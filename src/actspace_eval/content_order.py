"""Check of the image/text interleaving order in the rendered prompt.

All models receive the same interleaved image/text content. This module checks it against the rendered prompt saved for turn 0
of every episode (rendered_prompt.text in calls.jsonl): each model's image placeholders are replaced by a marker, the text
segments are located one by one in request order, and we check whether there is an image between every two text segments
(and after the last one). If this matches the request, the interleaving order is correct. Several adjacent images merged into
one segment also count as a match. The rendered text folds runs of identical tokens into "x×N" (models_vllm.rendered_prompt);
text is folded the same way before searching.

Results: interleaved (matches), images_first (all images moved before the first text segment), mismatch (any other difference),
no_image (the request has no image), no_render (no rendered prompt saved, e.g. the render endpoint timed out),
text_not_found (a text segment could not be located; no verdict).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .data import results_root

# Image placeholders of each model (rendered, expanded): Qwen family / RoboBrain / VST, InternVL / SenseNova, GLM, gemma,
# MiniCPM-V (including slices), LLaVA, Cambrian-S
PLACEHOLDER = re.compile(
    r"<\|vision_start\|>.*?<\|vision_end\|>|<img>.*?</img>|<\|begin_of_image\|>.*?<\|end_of_image\|>|<\|image>.*?<image\|>"
    r"|<image>.*?</image>|<slice>.*?</slice>|\(<image>\./</image>\)|<image>(?:×\d+)?|<\|image_pad\|>(?:×\d+)?|<IMG_CONTEXT>(?:×\d+)?", re.S)
MARK = "\x00"
TURN_END = ("<|im_end|>", "<end_of_turn>", "<|eot_id|>", "<|user|>", "<|assistant|>", "<turn|>")


def fold(s: str) -> str:
    """Like rendered_prompt, write runs of >= 3 identical characters as "x×N" (digits tokenised one by one get folded,
    e.g. photo_111147 -> photo_1×447)."""
    return re.sub(r"(.)\1{2,}", lambda m: f"{m.group(1)}×{len(m.group(0))}", s)


def _find(text: str, line: str, pos: int, head: bool) -> tuple[int, str]:
    """Find the first (head) or last 30 characters of a text line after pos; if not found, fold first and then cut
    (cutting must not split a folded string)."""
    for x in (line, fold(line)):
        x = x[:30] if head else x[-30:]
        i = text.find(x, pos)
        if i >= 0:
            return i, x
    return -1, line


def check_call(r: dict) -> str:
    """Interleaving order of one call (a call record in calls.jsonl)."""
    segs, pending = [], 0
    for m in r["request"]["messages"]:
        if not isinstance(m["content"], list):
            continue
        for q in m["content"]:
            if q["type"] == "image":
                pending += 1
            elif q["type"] == "text" and q["text"].strip():
                segs.append((pending, q["text"].strip()))
                pending = 0
    if sum(b for b, _ in segs) + pending == 0:
        return "no_image"
    text = (r.get("rendered_prompt") or {}).get("text") or ""
    if not text:
        return "no_render"
    t = PLACEHOLDER.sub(MARK, text)
    exp, obs, pos = [], [], 0
    for before, v in segs:
        lines = [x for x in v.split("\n") if x.strip()]
        i, first = _find(t, lines[0], pos, True)
        if i < 0:
            return "text_not_found"
        exp.append(before > 0)
        obs.append(MARK in t[pos:i])
        j, last = _find(t, lines[-1], i, False)
        pos = j + len(last) if j >= 0 else i + len(first)
    end = min([k for k in (t.find(x, pos) for x in TURN_END) if k >= 0] or [len(t)])
    exp.append(pending > 0)
    obs.append(MARK in t[pos:end])
    if exp == obs:
        return "interleaved"
    if obs[0] and not any(obs[1:]):
        return "images_first"
    return "mismatch"


def check_run(run_dir: Path) -> Counter:
    """Counts of the turn-0 results over all episodes of one run."""
    c = Counter()
    p = Path(run_dir) / "calls.jsonl"
    if not p.exists():
        return c
    for line in open(p):
        if not line.endswith("\n"):
            break
        r = json.loads(line)
        if r.get("type") == "call" and r["tag"].get("t") == 0:
            c[check_call(r)] += 1
    return c


def order_ok(counts: Counter) -> bool:
    """Gate: no images_first / mismatch / text_not_found; settings with images need at least one interleaved check
    (no_render gives no verdict)."""
    bad = counts["images_first"] + counts["mismatch"] + counts["text_not_found"]
    has_img = sum(v for k, v in counts.items() if k != "no_image")
    return bad == 0 and (has_img == 0 or counts["interleaved"] > 0)


def main():
    """Per-run audit: python -m actspace_eval.content_order [--root <results root>] [--models …] [--runs full_ smoke2_]"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="results root (default: results from the config)")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--runs", nargs="*", default=["full_"], help="run name prefixes")
    a = ap.parse_args()
    root = Path(a.root) if a.root else results_root()
    for m in a.models or sorted(x.name for x in root.iterdir() if x.is_dir() and not x.name.startswith("_") and x.name != "api"):
        for d in sorted(p for p in (root / m).glob("*/*") if p.is_dir() and any(p.name.startswith(r) for r in a.runs)):
            c = check_run(d)
            print(f"{m}\t{d.parent.name}\t{d.name}\t{'OK' if order_ok(c) else 'MISMATCH'}\t{dict(c)}", flush=True)


if __name__ == "__main__":
    main()
