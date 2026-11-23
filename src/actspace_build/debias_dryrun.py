"""Step 8 debiasing dry run: a feature-only blind test on the current ARKit questions. Reports only; removes no questions.

Method (Step 8 in docs/data_construction.md): each question uses only text features -- template/variant, the referring name of
each slot, the option texts (referring names of object options or quadrant names) and, for numeric questions, the category
combination; a random forest with scene-grouped GroupKFold (5 folds) gives out-of-fold predictions.
- Multiple choice: one model per template, the label is the text of the correct option; for each question the predicted
  probabilities are renormalised over its own options, bias score = probability of the correct option,
  blind accuracy = whether the most probable option is correct; chance level = mean of 1 / number of options.
- Numeric (T1): the distance is regressed and scored with MRA (mean hit rate of the relative error over thresholds
  0.50, 0.55, ..., 0.95); the reference baseline is the mean of the same category combination in the training folds
  (the overall training-fold mean when that combination is absent).
Unanswerable questions expect the answer "cannot be determined", which is not among the options, so they are excluded.
Usage: python -m actspace_build.debias_dryrun
Output: reports/arkit_debias_dryrun.md
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold

from .common import load_config, path_of, read_json, work_dir

DATASET = "arkitscenes"
THRESHOLDS = np.arange(0.50, 0.951, 0.05)
LETTERS = "ABCD"


def mra(pred, gt):
    rel = np.abs(np.asarray(pred) - np.asarray(gt)) / np.asarray(gt)
    return float(np.mean([(rel < 1 - t).mean() for t in THRESHOLDS]))


def load_rows(cfg):
    groups = [json.loads(l) for l in open(path_of(cfg, "work") / DATASET / "_step6" / "test_groups.jsonl")]
    names = {}
    rows = []
    for grp in groups:
        sid = grp["scene_id"]
        if sid not in names:
            objs = read_json(work_dir(cfg, DATASET, sid.split("_", 1)[1]) / "objects.json")
            names[sid] = {o["obj_id"]: o.get("ref_name") or o["category"] for o in objs}
        nm = names[sid]
        for q in grp["questions"]:
            r = {"scene": sid, "qid": q["qid"], "template": q["template"], "variant": q.get("variant") or "-",
                 "label": q["evidence_label"], "slots": {k: nm[v] for k, v in q["slots"].items()}}
            if q["answer_type"] == "numeric":
                r.update(numeric=True, answer=q["answer_raw"])
            else:
                opts = [nm[o] for o in q["option_objs"]] if "option_objs" in q else list(q["option_values"])
                r.update(numeric=False, options=opts, answer=opts[LETTERS.index(q["answer"])])
            rows.append(r)
    return rows


def featurize(rows, with_options):
    """One-hot / multi-hot text features: variant, name of each slot, set of option names."""
    vocab = {}

    def idx(k):
        return vocab.setdefault(k, len(vocab))
    feats = []
    for r in rows:
        f = [idx(f"variant={r['variant']}")]
        f += [idx(f"{k}={v}") for k, v in r["slots"].items()]
        f += [idx(f"name={v}") for v in r["slots"].values()]
        if with_options:
            f += [idx(f"opt={o}") for o in r["options"]]
            f.append(idx(f"n_opts={len(r['options'])}"))
        feats.append(f)
    X = np.zeros((len(rows), len(vocab)), np.float32)
    for i, f in enumerate(feats):
        X[i, f] = 1.0
    return X


def blind_mc(rows, seed, folds):
    X = featurize(rows, True)
    y = np.array([r["answer"] for r in rows])
    groups = np.array([r["scene"] for r in rows])
    rng = np.random.default_rng(seed)
    correct, bias, chance = np.zeros(len(rows)), np.zeros(len(rows)), np.array([1 / len(r["options"]) for r in rows])
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
        P = clf.predict_proba(X[te])
        cls = {c: k for k, c in enumerate(clf.classes_)}
        for row_i, p in zip(te, P):
            opts = rows[row_i]["options"]
            s = np.array([p[cls[o]] if o in cls else 0.0 for o in opts])
            s = s / s.sum() if s.sum() > 0 else np.full(len(opts), 1 / len(opts))
            s_tb = s + rng.uniform(0, 1e-9, len(s))  # break ties at random
            ans = opts.index(rows[row_i]["answer"])
            correct[row_i] = float(np.argmax(s_tb) == ans)
            bias[row_i] = s[ans]
    return correct, bias, chance


def blind_numeric(rows, seed, folds):
    X = featurize(rows, False)
    y = np.array([r["answer"] for r in rows], float)
    groups = np.array([r["scene"] for r in rows])
    combo = np.array(["|".join(sorted(r["slots"].values())) for r in rows])
    pred, base = np.zeros(len(rows)), np.zeros(len(rows))
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        pred[te] = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1).fit(X[tr], y[tr]).predict(X[te])
        means = {c: y[tr][combo[tr] == c].mean() for c in set(combo[tr])}
        base[te] = [means.get(c, y[tr].mean()) for c in combo[te]]
    return pred, base, y


def main():
    cfg = load_config()
    folds = cfg["debias"]["n_folds"]
    margin = cfg["debias"]["mc_margin_over_chance_pp"] / 100
    rows = [r for r in load_rows(cfg) if r["label"] != "unanswerable"]
    n_scenes = len({r["scene"] for r in rows})
    out, worst = [], None
    for t in sorted({r["template"] for r in rows}):
        rs = [r for r in rows if r["template"] == t]
        if rs[0]["numeric"]:
            pred, base, y = blind_numeric(rs, cfg["seed"], folds)
            m_rf, m_base = mra(pred, y), mra(base, y)
            gap = m_rf - m_base
            out.append(f"| {t} | numeric | {len(rs)} | MRA {m_rf:.3f} | category-combination mean baseline MRA {m_base:.3f} | {gap * 100:+.1f} pp | "
                       f"{'yes' if gap > margin else 'no'} |")
            score = gap
        else:
            correct, bias, chance = blind_mc(rs, cfg["seed"], folds)
            acc, ch = correct.mean(), chance.mean()
            gap = acc - ch
            by_var = Counter(r["variant"] for r in rs)
            extra = ""
            if len(by_var) > 1:
                parts = []
                for v in sorted(by_var):
                    m = np.array([r["variant"] == v for r in rs])
                    parts.append(f"{v}: {correct[m].mean():.1%} / chance {chance[m].mean():.1%} ({m.sum()} questions)")
                extra = "; " + "; ".join(parts)
            out.append(f"| {t} | multiple choice | {len(rs)} | blind accuracy {acc:.1%} | chance {ch:.1%} | {gap * 100:+.1f} pp | "
                       f"{'yes' if gap > margin else 'no'}{extra} |")
            score = gap
        if worst is None or score > worst[1]:
            worst = (t, score)
    L = ["# Step 8 debiasing dry run: ARKit feature-only blind test (report only, no questions removed)", "",
         f"Scope: the {len(rows)} answerable questions among the current ARKit questions ({n_scenes} scenes; unanswerable questions expect \"cannot be determined\", which is not among the options, so they are excluded).", "",
         f"Method: text features of the question only (template/variant, referring name of each slot, option texts, category combination for numeric questions), random forest (300 trees), "
         f"scene-grouped GroupKFold with {folds} folds, out-of-fold predictions. Multiple-choice models are trained per template and probabilities are renormalised over each question's own options; "
         "numeric questions are scored with MRA (thresholds 0.50-0.95) against a baseline of \"the mean of the same category combination in the training folds\".", "",
         f"Criteria (Step 8 in docs/data_construction.md): multiple-choice blind accuracy <= chance + {cfg['debias']['mc_margin_over_chance_pp']} pp; "
         f"numeric blind MRA <= category-combination mean baseline + {cfg['debias']['mc_margin_over_chance_pp']} pp.", "",
         "| Template | Type | Questions | Blind score | Reference | Difference | Exceeds +5 pp |", "|---|---|---|---|---|---|---|", *out, "",
         f"**Most biased template: {worst[0]} ({worst[1] * 100:+.1f} pp above the reference).**", "",
         "Note: this is a dry run and no questions were removed; the actual debiasing (iterative pruning in Step 8) runs on the full dataset after ScanNet++ is merged. "
         "The zero-shot LLM blind test of Step 8 was not run here."]
    (path_of(cfg, "reports") / "arkit_debias_dryrun.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
