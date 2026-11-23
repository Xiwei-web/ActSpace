"""Step 7: question rewriting with a template-level paraphrase bank (written directly by Claude; no external API calls).

  validate  checks every sentence in configs/paraphrases.yaml with the Step 7 automatic validation rules
  apply     for each question, picks one paraphrase of its template at random with a scene-level seed, fills in object names and writes text; the original template sentence is kept in text_template
            input work/arkitscenes/_step6/test_groups.jsonl, output work/arkitscenes/_step7/test_groups.jsonl
  report    reports/step7_rewrite.md (full paraphrase bank, validation pass rate, 50 random before/after pairs, chi-square test of paraphrase ID × correct answer)
Also writes build_info.json (repository root) recording the rewrite method, the authoring model and the hash of the paraphrase file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import yaml

from .common import ROOT, load_config, path_of, read_json, scene_rng, work_dir, write_json

DATASET = "arkitscenes"
BANK = ROOT / "configs" / "paraphrases.yaml"
AUTHOR_MODEL = "Claude Opus 5.5 (claude-opus-5-5)"
# Revisions after review (IDs start at 0, matching # in the report)
REVISIONS = [
    "Review passed overall. Required changes: T2_closest #3 → \"Select, from the options below, the object that is closest to the {OBJ1}.\"; "
    "T2_farthest #3 → \"Select, from the options below, the object that is farthest from the {OBJ1}.\" (the original lacked the \"among the options\" restriction).",
    "Suggested change (adopted): T3 #6 → \"Consider standing at the {OBJ1}, facing toward the {OBJ2}. Where is the {OBJ3} relative to you?\" (the answer depends on the facing direction).",
    "New validation rule: T2 closest / farthest must contain an option-scope word. Under this rule T2_closest #4 and T2_farthest #4 (\"Which of these objects …\") fail "
    "and were minimally changed to \"Which of these listed objects …\"; #8 \"Choose the option …\" passes because singular option counts.",
]

# Direction cue words: must not appear in direction questions (they would leak or change the answer); new cues such as "near/next to/window" must not appear in any template
CUE_WORDS = ["left", "right", "front", "back", "behind", "ahead", "beside"]
GLOBAL_FORBIDDEN = ["near the", "next to", "window", "door", "wall", "corner", "approximately", "roughly", "about"]
REQUIRED = {
    "T1": [["center"], ["meter"], ["distance", "far", "separate", "long"]],
    "T2_pair": [["closer"]],
    "T2_closest": [["closest"]],
    "T2_farthest": [["farthest"]],
    "T3": [["stand", "stood"], ["fac"], ["relative to you", "point of view", "relative to yourself"]],
    "T4": [["start"], ["direction", "facing", "faced", "looking"]],
    "T5": [["stand", "stood"], ["fac"], ["left"], ["closest"]],
}
FORBIDDEN = {
    "T1": [],
    "T2_pair": ["farther", "further", "farthest", "closest"],
    "T2_closest": ["farthest", "furthest", "farther"],
    "T2_farthest": ["closest", "nearest", "closer"],
    "T3": CUE_WORDS,
    "T4": CUE_WORDS,
    "T5": [w for w in CUE_WORDS if w != "left"] + ["farthest"],
}


def template_key(q) -> str:
    return f"T2_{q['variant']}" if q["template"] == "T2" else q["template"]


def load_bank():
    return yaml.safe_load(open(BANK))


def bank_hash() -> str:
    return hashlib.sha256(BANK.read_bytes()).hexdigest()


def object_words(cfg) -> list[str]:
    """Object names that must not appear outside placeholders: vocabulary categories allowed in questions, plus size-reference words."""
    cats = pd.read_csv(path_of(cfg, "vocab") / "categories.tsv", sep="\t", dtype=str, keep_default_na=False)
    words = [c.lower() for c, a in zip(cats.category, cats.allowed) if a.strip().lower() == "true"]
    return words + ["larger", "smaller", "largest", "smallest"]


def validate_one(key, original, para, obj_words) -> list[str]:
    errs = []
    ph_orig = re.findall(r"\{OBJ\d\}", original)
    ph = re.findall(r"\{OBJ\d\}", para)
    if sorted(ph) != sorted(ph_orig) or any(ph.count(p) != 1 for p in ph_orig):
        errs.append(f"placeholder mismatch: {ph} (expected {ph_orig}, once each)")
    for p in ph_orig:
        if not re.search(r"\bthe " + re.escape(p), para, flags=re.I):
            errs.append(f"no 'the ' before {p}")
    if re.search(r"\d", re.sub(r"\{OBJ\d\}", "", para)):
        errs.append("contains a digit")
    low = re.sub(r"\{OBJ\d\}", " ", para).lower()
    for w in obj_words:
        if re.search(r"\b" + re.escape(w) + r"s?\b", low):
            errs.append(f"introduces an object name or reference word: {w}")
    for w in GLOBAL_FORBIDDEN + FORBIDDEN[key]:
        if re.search(r"\b" + re.escape(w), low):
            errs.append(f"contains a forbidden word: {w}")
    for alts in REQUIRED[key]:
        if not any(a in low for a in alts):
            errs.append(f"missing key semantic word: {' / '.join(alts)}")
    # T2 closest / farthest must be restricted to the options; otherwise they could be read as the closest / farthest object in the whole scene
    if key in ("T2_closest", "T2_farthest") and not re.search(r"\b(following|below|listed|options?|choices)\b", low):
        errs.append("missing option-scope word (following / below / listed / options / choices)")
    if para.strip() == original.strip():
        errs.append("identical to the original template sentence")
    return errs


def validate(cfg):
    bank = load_bank()
    obj_words = object_words(cfg)
    res = {}
    for key, entry in bank.items():
        paras = entry["paraphrases"]
        rows = []
        for i, p in enumerate(paras):
            e = validate_one(key, entry["template"], p, obj_words)
            if paras.index(p) != i:
                e.append("duplicates another sentence of the same template")
            rows.append({"id": i, "text": p, "errors": e})
        res[key] = rows
    return res


def fill(text, slots, names):
    return text.format(**{k: names[v] for k, v in slots.items()})


def apply(cfg):
    bank = load_bank()
    groups = [json.loads(l) for l in open(path_of(cfg, "work") / DATASET / "_step6" / "test_groups.jsonl")]
    names, rngs = {}, {}
    for grp in groups:
        sid = grp["scene_id"]
        if sid not in names:
            objs = read_json(work_dir(cfg, DATASET, sid.split("_", 1)[1]) / "objects.json")
            names[sid] = {o["obj_id"]: o.get("ref_name") or o["category"] for o in objs}
            rngs[sid] = scene_rng(cfg, sid, "step7_rewrite")
        for q in grp["questions"]:
            key = template_key(q)
            entry = bank[key]
            assert q["text_template"] == entry["template"], (q["qid"], q["text_template"])
            i = int(rngs[sid].integers(len(entry["paraphrases"])))
            q["paraphrase_key"], q["paraphrase_id"] = key, i
            q["text"] = fill(entry["paraphrases"][i], q["slots"], names[sid])
    out = path_of(cfg, "work") / DATASET / "_step7"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "test_groups.jsonl", "w") as f:
        for grp in groups:
            f.write(json.dumps(grp, ensure_ascii=False) + "\n")
    info_path = path_of(cfg, "build_info") if "build_info" in cfg["paths"] else ROOT / "build_info.json"  # what-if sandboxes do not write to the repository
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info["step7_rewrite"] = {
        "method": "template-level paraphrase bank",
        "author_model": AUTHOR_MODEL,
        "external_api": False,
        "paraphrase_file": "configs/paraphrases.yaml",
        "paraphrase_file_sha256": bank_hash(),
        "n_templates": len(bank),
        "paraphrases_per_template": {k: len(v["paraphrases"]) for k, v in bank.items()},
        "selection": "one paraphrase per question, drawn uniformly with the scene-level seed scene_rng(seed, scene_id, 'step7_rewrite')",
        "temperature": None,
        "prompt": None,
        "seed": cfg["seed"],
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n")
    print(f"rewrote {sum(len(g_['questions']) for g_ in groups)} questions -> {out / 'test_groups.jsonl'}")
    return groups


def chi_square(groups, n_perm=5000):
    """Per template: chi-square test on the contingency table of paraphrase ID × correct answer (correct option letter for multiple choice; T3/T4 also test the quadrant; T1 uses answer quartiles)."""
    from scipy.stats import chi2_contingency
    by = defaultdict(list)
    for grp in groups:
        for q in grp["questions"]:
            by[q["paraphrase_key"]].append(q)
    rows = []
    for key in sorted(by):
        qs = by[key]
        tests = []
        if qs[0]["answer_type"] == "numeric":
            v = np.array([q["answer_raw"] for q in qs])
            bins = np.quantile(v, [0.25, 0.5, 0.75])
            tests.append(("answer quartile", [int(np.searchsorted(bins, x)) for x in v]))
        else:
            tests.append(("correct option letter", [q["answer"] for q in qs]))
            if key in ("T3", "T4"):
                tests.append(("quadrant", [q["option_values"]["ABCD".index(q["answer"])] for q in qs]))
        pid = np.array([q["paraphrase_id"] for q in qs])
        for name, ans in tests:
            ans = np.array(ans)
            tab = pd.crosstab(pd.Series(pid), pd.Series(ans))
            chi2, p, dof, _ = chi2_contingency(tab.values)
            # Permutation test: the chi-square approximation is unreliable for sparse tables, so shuffle paraphrase IDs n_perm times for an exact permutation p-value
            rng = np.random.default_rng(0)
            perm = [chi2_contingency(pd.crosstab(pd.Series(rng.permutation(pid)), pd.Series(ans)).values)[0] for _ in range(n_perm)]
            p_perm = (1 + sum(c >= chi2 for c in perm)) / (n_perm + 1)
            rows.append((key, name, len(qs), tab.shape, chi2, dof, p, p_perm))
    return rows


def report(cfg, groups, val, initial_fail=None):
    bank = load_bank()
    n_all = sum(len(v) for v in val.values())
    n_ok = sum(not r["errors"] for v in val.values() for r in v)
    use = Counter((q["paraphrase_key"], q["paraphrase_id"]) for g_ in groups for q in g_["questions"])
    per_key = Counter(q["paraphrase_key"] for g_ in groups for q in g_["questions"])
    L = ["# Step 7: question rewriting (template-level paraphrase bank)", "",
         "> No external API is called: Claude (Opus 5.5) wrote 12 paraphrases for each template directly; "
         "each question gets one paraphrase of its template, chosen uniformly at random with a scene-level seed; object names are filled in and written to `text`, and the original template sentence is kept in `text_template`. "
         f"Paraphrase file `configs/paraphrases.yaml`, SHA-256 `{bank_hash()}` (also recorded in `build_info.json`).", "",
         "## 1. Templates (from arkit_test_groups_preview.jsonl)", "",
         "| Template key | Original template sentence | Questions |", "|---|---|---|",
         *[f"| {k} | {v['template']} | {per_key[k]} |" for k, v in bank.items()], "",
         "Spelling variants such as larger / smaller / largest / smallest / armchair appear only in the object names filled into placeholders and do not form separate templates.", "",
         "## 2. Automatic validation", "",
         "Each sentence is checked for: placeholders identical to the original template, each appearing once and preceded by \"the \"; no digits; no vocabulary object names or size-reference words outside placeholders; "
         "no new cue words (near the / next to / window / door / wall / corner / approximately / roughly / about); "
         "direction questions (T3/T4) contain no left / right / front / back / behind / ahead / beside, and T5 allows only left; "
         "unchanged comparison direction (T2 pair must contain closer and no farther / closest etc.; closest / farthest are mutually exclusive); "
         "required semantic words: T1 center + meter + a distance word; T3 / T5 stand + face; T3 \"relative to you / point of view\"; "
         "T4 start + a direction word; T5 left + closest; T2 closest / farthest must contain an option-scope word (following / below / listed / option(s) / choices); "
         "not identical to the original template sentence and no duplicates within a template.", "",
         f"**Final pass rate: {n_ok} / {n_all} ({n_ok / n_all:.0%})**. " + (f"In this round (after review changes and the new scope rule), {initial_fail} sentences failed the first check and were rewritten until all passed." if initial_fail else "All sentences passed the first check; no rewriting needed."), "",
         "### Revisions after review", "", *[f"- {r}" for r in REVISIONS], "",
         "## 3. Full paraphrase bank (for sentence-by-sentence review)", ""]
    for k, v in bank.items():
        L += [f"### {k}", "", f"Original template: {v['template']}", "", "| # | Paraphrase | Validation | Uses |", "|---|---|---|---|"]
        for r in val[k]:
            L.append(f"| {r['id']} | {r['text']} | {'pass' if not r['errors'] else '; '.join(r['errors'])} | {use[(k, r['id'])]} |")
        L.append("")
    rng = np.random.default_rng([cfg["seed"], 7])
    qs = [(g_["scene_id"], q) for g_ in groups for q in g_["questions"]]
    pick = sorted(rng.choice(len(qs), min(50, len(qs)), replace=False).tolist())
    L += ["## 4. Before/after rewriting for 50 random questions", "", "| # | qid | Original template sentence (object names filled in) | Rewritten |", "|---|---|---|---|"]
    names = {}
    for n, i in enumerate(pick, 1):
        sid, q = qs[i]
        if sid not in names:
            objs = read_json(work_dir(cfg, DATASET, sid.split("_", 1)[1]) / "objects.json")
            names[sid] = {o["obj_id"]: o.get("ref_name") or o["category"] for o in objs}
        L.append(f"| {n} | {q['qid']} | {fill(q['text_template'], q['slots'], names[sid])} | {q['text']} |")
    L += ["", "## 5. Association between paraphrase ID and correct answer (chi-square test)", "",
          "For each template, a chi-square test of independence on the \"paraphrase ID × correct answer\" contingency table; a large p-value means no association. The tables are sparse (12 rows, most cells with expected count < 5), "
          "so permutation-test p-values (paraphrase IDs shuffled 5000 times) are also given. Paraphrase selection uses only scene-level random numbers and is independent of the answer; with 9 tests, "
          "the multiple-testing corrected significance threshold is about 0.05 / 9 ≈ 0.0056.", "",
          "| Template key | Answer variable | Questions | Table | χ² | dof | Chi-square p | Permutation p |", "|---|---|---|---|---|---|---|---|",
          *[f"| {k} | {name} | {n} | {shape[0]}×{shape[1]} | {chi2:.1f} | {dof} | {p:.3f} | {pp:.3f} |"
            for k, name, n, shape, chi2, dof, p, pp in chi_square(groups)]]
    (path_of(cfg, "reports") / "step7_rewrite.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-12:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "run"])
    ap.add_argument("--initial-fail", type=int, default=0, help="number of sentences that failed the first validation round (written to the report)")
    args = ap.parse_args()
    cfg = load_config()
    val = validate(cfg)
    bad = [(k, r["id"], r["text"], r["errors"]) for k, v in val.items() for r in v if r["errors"]]
    if args.cmd == "validate":
        print(f"{sum(len(v) for v in val.values()) - len(bad)} / {sum(len(v) for v in val.values())} passed")
        for b in bad:
            print(b)
        return
    if bad:
        raise SystemExit(f"{len(bad)} paraphrases fail validation; fix configs/paraphrases.yaml first")
    groups = apply(cfg)
    report(cfg, groups, val, args.initial_fail)


if __name__ == "__main__":
    main()
