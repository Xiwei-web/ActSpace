"""Answer parsing and per-question scoring (docs/evaluation_protocol.md)."""
from __future__ import annotations

import json
import math
import re

CANNOT = "cannot_determine"
DEFAULT_THRESH = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def public_questions(group: dict) -> list[dict]:
    """Public question view: Q1…Qn, question text, answer type, options; no qid, template, evidence or answer."""
    out = []
    for i, q in enumerate(group["questions"], 1):
        numeric = q["answer_type"] == "numeric"
        out.append({"id": f"Q{i}", "text": q["text"], "answer_type": "numeric" if numeric else "multiple_choice",
                    "unit": "m" if numeric else None, "options": None if numeric else list(q["options"])})
    return out


def _first_json_object(text: str):
    """The first parseable JSON object in the raw text (other text or code-fence markers may surround it)."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_value(v, q_public: dict):
    """Return (parsed value, error message). The value is an option letter, a positive number (metres) or CANNOT; None if parsing failed."""
    if isinstance(v, str) and v.strip().lower() == CANNOT:
        return CANNOT, None
    if q_public["answer_type"] == "numeric":
        if isinstance(v, bool):
            return None, "not a number"
        if isinstance(v, (int, float)):
            x = float(v)
        elif isinstance(v, str):
            m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(m|meters?|metres?)?\s*", v)
            if not m:
                return None, "not a number"
            x = float(m.group(1))
        else:
            return None, "not a number"
        return (x, None) if math.isfinite(x) and x > 0 else (None, "not a positive finite number")
    if not isinstance(v, str):
        return None, "not a letter"
    letters = [o.split(".", 1)[0].strip().upper() for o in q_public["options"]]
    m = re.fullmatch(r"\s*\(?([A-Za-z])\)?(?:\s*[.:)]\s*.*)?\s*", v, re.S)
    if not m or m.group(1).upper() not in letters:
        return None, "not one of the option letters"
    return m.group(1).upper(), None


def parse_answers(content, q_public: list[dict]) -> dict:
    """content: raw text or a dict. Returns {"ok": whether the whole reply parsed as a JSON object, "answers": {Qi: value or None}, "errors": {Qi: message}}."""
    obj = content if isinstance(content, dict) else _first_json_object(content) if isinstance(content, str) else None
    if obj is None:
        return {"ok": False, "answers": {q["id"]: None for q in q_public}, "errors": {q["id"]: "no JSON object" for q in q_public}}
    lower = {str(k).strip().lower(): v for k, v in obj.items()}
    answers, errors = {}, {}
    for q in q_public:
        if q["id"].lower() not in lower:
            answers[q["id"]], errors[q["id"]] = None, "missing"
            continue
        answers[q["id"]], err = parse_value(lower[q["id"].lower()], q)
        if err:
            errors[q["id"]] = err
    return {"ok": True, "answers": answers, "errors": errors}


def gold(q: dict):
    """Gold answer: unanswerable → CANNOT; T1 → answer_raw (metres); multiple choice → letter."""
    if q["evidence_label"] == "unanswerable":
        return CANNOT
    return float(q["answer_raw"]) if q["answer_type"] == "numeric" else q["answer"]


def score_question(q: dict, pred, thresholds=DEFAULT_THRESH) -> float:
    """Per-question score: multiple choice 0/1; answerable T1 MRA; unanswerable scores 1 only for CANNOT; an answerable question answered CANNOT or left unanswered scores 0."""
    y = gold(q)
    if y == CANNOT:
        return float(pred == CANNOT)
    if pred is None or pred == CANNOT:
        return 0.0
    if q["answer_type"] == "numeric":
        return float(sum(abs(pred - y) / y < 1 - t for t in thresholds) / len(thresholds))
    return float(pred == y)
