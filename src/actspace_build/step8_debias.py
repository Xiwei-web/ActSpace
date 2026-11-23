"""Step 8: blind tests and debiasing (ARKit only; after ScanNet++ is added, rerun on the merged data from the pre-pruning state).

Input: work/arkitscenes/_step7/test_groups.jsonl (complete pre-pruning question set, left unchanged)
Output: work/arkitscenes/_step8/<debias.output_version>/test_groups.jsonl + manifest.json;
      report reports/step8_debias_arkit.md (v0.9) / reports/step8_debias_<version>.md (from v0.10 on; does not overwrite the v0.9 report)

Step 8 (see docs/data_construction.md):
1. Feature blind test: each question uses only text features (template/variant, paraphrase ID, referring name of each slot, option texts [by position and as a set], number of options),
   random forest, GroupKFold by scene with 5 folds, giving out-of-fold predictions; bias score of multiple-choice questions = out-of-fold probability of the correct option (normalized over the question's options),
   bias score of numeric questions = MRA score of the out-of-fold prediction (mean hit rate over thresholds 0.50–0.95).
   Two controls: a baseline that always guesses the most common answer of the training folds; rerunning the blind test with option order shuffled.
2. LLM zero-shot blind test: not done in this round (no external API calls).
3. Iterative pruning per template: each round removes the 5% with the highest bias scores, then rebalances the answer distribution (T3/T4 quadrants 25% ± 5%,
   T4 front / back 50% ± 5% each, option letters reassigned uniformly per (template, number of options)), until
   multiple-choice blind accuracy ≤ chance + 5 pp and numeric blind MRA ≤ category-pair mean baseline + 5 pp.
4. Stop and report if a single template has had more than 40% removed in total and still fails.
Unanswerable questions expect the answer "cannot be determined"; they are excluded from blind tests and never pruned.
When a question in a constraint is pruned, the constraint becomes invalid; a question group needs at least 1 constraint, 3 questions and 2 question types, otherwise the whole group is dropped (GROUP_INCOMPLETE).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold

from .common import DropLog, assert_not_frozen, load_config, path_of, read_json, work_dir, write_json

DATASET = "arkitscenes"
THRESHOLDS = np.arange(0.50, 0.951, 0.05)
LETTERS = "ABCD"
FRONT = ("front-left", "front-right")
QUADS = ("front-left", "front-right", "back-left", "back-right")


# ---------------------------------------------------------------- data

def load(cfg):
    src = path_of(cfg, "work") / DATASET / "_step7" / "test_groups.jsonl"
    groups = [json.loads(l) for l in open(src)]
    names = {}
    for grp in groups:
        sid = grp["scene_id"]
        if sid not in names:
            objs = read_json(work_dir(cfg, DATASET, sid.split("_", 1)[1]) / "objects.json")
            names[sid] = {o["obj_id"]: o.get("ref_name") or o["category"] for o in objs}
    return groups, names, hashlib.sha256(src.read_bytes()).hexdigest()


def option_texts(q):
    return [o.split(". ", 1)[1] for o in q["options"]]


def answer_text(q):
    return option_texts(q)[LETTERS.index(q["answer"])]


def features(qs, names, sids, shuffle_rng=None):
    """Text-only features (multi-hot). If shuffle_rng is given, each question's option order is shuffled first (control)."""
    vocab, rows = {}, []

    def idx(k):
        return vocab.setdefault(k, len(vocab))
    for q, sid in zip(qs, sids):
        nm = names[sid]
        f = [idx(f"variant={q.get('variant') or '-'}"), idx(f"para={q.get('paraphrase_key')}:{q.get('paraphrase_id')}")]
        f += [idx(f"{k}={nm[v]}") for k, v in q["slots"].items()] + [idx(f"name={nm[v]}") for v in q["slots"].values()]
        if q["answer_type"] != "numeric":
            opts = option_texts(q)
            if shuffle_rng is not None:
                opts = [opts[i] for i in shuffle_rng.permutation(len(opts))]
            f += [idx(f"pos{k}={o}") for k, o in enumerate(opts)] + [idx(f"opt={o}") for o in opts] + [idx(f"n_opts={len(opts)}")]
        rows.append(f)
    X = np.zeros((len(rows), len(vocab)), np.float32)
    for i, f in enumerate(rows):
        X[i, f] = 1.0
    return X


def mra_each(pred, gt):
    rel = np.abs(np.asarray(pred) - np.asarray(gt)) / np.asarray(gt)
    return np.mean([(rel < 1 - t) for t in THRESHOLDS], axis=0)


# ---------------------------------------------------------------- blind test

FOLD_FALLBACKS = Counter()  # (number of scenes, actual folds) → count, written to the report


def effective_folds(n_scenes, folds):
    """With fewer scenes than n_folds: 3 folds if there are ≥ 3 scenes, otherwise leave-one-scene-out (one scene per fold)."""
    if n_scenes >= folds:
        return folds
    if n_scenes < 2:
        raise ValueError(f"only {n_scenes} scene(s): cannot split by scene")
    return 3 if n_scenes >= 3 else n_scenes


def fold_splits(groups, folds, split_seed=None):
    """K folds grouped by scene. With split_seed None uses GroupKFold (deterministic); otherwise assigns scenes to folds at random (for stability checks).
    With fewer scenes than folds, falls back according to effective_folds and records it in FOLD_FALLBACKS."""
    k = effective_folds(len(set(groups)), folds)
    if k != folds:
        FOLD_FALLBACKS[(len(set(groups)), k)] += 1
        folds = k
    if split_seed is None:
        return list(GroupKFold(n_splits=folds).split(groups, groups=groups))
    uniq = sorted(set(groups))
    perm = np.random.default_rng(split_seed).permutation(len(uniq))
    fold_of = {u: int(perm[i]) % folds for i, u in enumerate(uniq)}
    f = np.array([fold_of[g_] for g_ in groups])
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(folds)]


def blind_mc(qs, sids, names, cfg, shuffle=False, split_seed=None, rf_seed=None):
    seed, folds = cfg["seed"], cfg["debias"]["n_folds"]
    X = features(qs, names, sids, np.random.default_rng([seed, 81]) if shuffle else None)
    y = np.array([answer_text(q) for q in qs])
    groups = np.array(sids)
    tie = np.random.default_rng([seed, 82])
    bias, correct, major = np.zeros(len(qs)), np.zeros(len(qs)), np.zeros(len(qs))
    for tr, te in fold_splits(groups, folds, split_seed):
        clf = RandomForestClassifier(n_estimators=300, random_state=seed if rf_seed is None else rf_seed, n_jobs=-1).fit(X[tr], y[tr])
        P = clf.predict_proba(X[te])
        cls = {c: k for k, c in enumerate(clf.classes_)}
        freq = Counter(y[tr])
        for i, p in zip(te, P):
            opts = option_texts(qs[i])
            s = np.array([p[cls[o]] if o in cls else 0.0 for o in opts])
            s = s / s.sum() if s.sum() > 0 else np.full(len(opts), 1 / len(opts))
            a = opts.index(y[i])
            bias[i] = s[a]
            correct[i] = float(np.argmax(s + tie.uniform(0, 1e-9, len(s))) == a)
            m = np.array([freq.get(o, 0) for o in opts], float)  # control: only guess the most common answer of the training folds
            major[i] = float(np.argmax(m + tie.uniform(0, 1e-9, len(m))) == a)
    chance = np.array([1 / len(q["options"]) for q in qs])
    return {"bias": bias, "acc": correct.mean(), "chance": chance.mean(), "majority": major.mean()}


def blind_numeric(qs, sids, names, cfg, split_seed=None, rf_seed=None):
    seed, folds = cfg["seed"], cfg["debias"]["n_folds"]
    X = features(qs, names, sids)
    y = np.array([q["answer_raw"] for q in qs], float)
    groups = np.array(sids)
    combo = np.array(["|".join(sorted(names[s][v] for v in q["slots"].values())) for q, s in zip(qs, sids)])
    pred, base, major = np.zeros(len(qs)), np.zeros(len(qs)), np.zeros(len(qs))
    for tr, te in fold_splits(groups, folds, split_seed):
        pred[te] = RandomForestRegressor(n_estimators=300, random_state=seed if rf_seed is None else rf_seed, n_jobs=-1).fit(X[tr], y[tr]).predict(X[te])
        means = {c: y[tr][combo[tr] == c].mean() for c in set(combo[tr])}
        base[te] = [means.get(c, y[tr].mean()) for c in combo[te]]
        vals, cnt = np.unique(np.round(y[tr], 1), return_counts=True)  # control: most common answer of the training folds (rounded to 0.1 m)
        major[te] = vals[np.argmax(cnt)]
    bias = mra_each(pred, y)
    return {"bias": bias, "mra": bias.mean(), "baseline": mra_each(base, y).mean(), "majority": mra_each(major, y).mean()}


def evaluate(groups, names, cfg, controls=False):
    """Run the blind test for each template and return {template: result}; items in a result are [(group index, question index)], aligned with bias."""
    by = defaultdict(list)
    for gi, grp in enumerate(groups):
        for qi, q in enumerate(grp["questions"]):
            if q["evidence_label"] != "unanswerable":
                by[q["template"]].append((gi, qi))
    margin = cfg["debias"]["mc_margin_over_chance_pp"] / 100
    out = {}
    for t, items in sorted(by.items()):
        qs = [groups[gi]["questions"][qi] for gi, qi in items]
        sids = [groups[gi]["scene_id"] for gi, _ in items]
        if qs[0]["answer_type"] == "numeric":
            r = blind_numeric(qs, sids, names, cfg)
            r["ok"] = r["mra"] <= r["baseline"] + margin
            r["gap"] = r["mra"] - r["baseline"]
        else:
            r = blind_mc(qs, sids, names, cfg)
            r["ok"] = r["acc"] <= r["chance"] + margin
            r["gap"] = r["acc"] - r["chance"]
            if controls:
                r["shuffled_acc"] = blind_mc(qs, sids, names, cfg, shuffle=True)["acc"]
            by_var = Counter(q.get("variant") or "-" for q in qs)
            if controls and len(by_var) > 1:
                r["by_variant"] = {}
                for v in sorted(by_var):
                    m = np.array([(q.get("variant") or "-") == v for q in qs])
                    sub = blind_mc([q for q, k in zip(qs, m) if k], [s for s, k in zip(sids, m) if k], names, cfg) if m.sum() >= 20 else None
                    r["by_variant"][v] = None if sub is None else (sub["acc"], sub["chance"], int(m.sum()))
        r["items"], r["n"] = items, len(items)
        out[t] = r
    return out


def stability(groups, names, cfg, n_rep=5):
    """Stability check: change the random scene-to-fold split and the random forest seed n_rep times; report mean ± std of each template's blind score minus control."""
    by = defaultdict(list)
    for gi, grp in enumerate(groups):
        for qi, q in enumerate(grp["questions"]):
            if q["evidence_label"] != "unanswerable":
                by[q["template"]].append((gi, qi))
    out = {}
    for t, items in sorted(by.items()):
        qs = [groups[gi]["questions"][qi] for gi, qi in items]
        sids = [groups[gi]["scene_id"] for gi, _ in items]
        gaps = []
        for k in range(n_rep):
            if qs[0]["answer_type"] == "numeric":
                r = blind_numeric(qs, sids, names, cfg, split_seed=[cfg["seed"], 800 + k], rf_seed=k)
                gaps.append(r["mra"] - r["baseline"])
            else:
                r = blind_mc(qs, sids, names, cfg, split_seed=[cfg["seed"], 800 + k], rf_seed=k)
                gaps.append(r["acc"] - r["chance"])
        out[t] = (float(np.mean(gaps)), float(np.std(gaps, ddof=1)), [round(float(x) * 100, 1) for x in gaps])
    return out


# ---------------------------------------------------------------- pruning and rebalancing

def remove_questions(groups, kill, reasons, drops, gcfg, cfg_keep=False):
    """kill: {(gi, qi)}. A deleted question in a constraint invalidates the constraint; groups that no longer have "≥ 1 constraint, ≥ 3 questions, ≥ 2 question types" are dropped entirely."""
    lo = gcfg["questions_per_group"][0]
    removed = []
    for gi, grp in enumerate(groups):
        ks = {qi for g_, qi in kill if g_ == gi}
        if not ks:
            continue
        qids_killed = {grp["questions"][qi]["qid"] for qi in ks}
        for qi in sorted(ks):
            q = grp["questions"][qi]
            drops[grp["scene_id"]].add("question", q["qid"], reasons[(gi, qi)], template=q["template"])
            removed.append((q["qid"], q["template"], reasons[(gi, qi)]))
        had = bool(grp["constraints"])
        grp["constraints"] = [c for c in grp["constraints"] if not set(c["qids"]) & qids_killed]
        grp["questions"] = [q for qi, q in enumerate(grp["questions"]) if qi not in ks]
        # Drop the whole group if all its original constraints are invalid; with debias.keep_group_on_constraint_loss (read-only what-if analysis) true, only the constraints become invalid.
        # Groups generated without constraints (groups.require_constraint: false) are not dropped for "having no constraint".
        lost = had and not grp["constraints"] and not cfg_keep
        if lost or len(grp["questions"]) < lo or len({q["template"] for q in grp["questions"]}) < 2:
            for q in grp["questions"]:
                removed.append((q["qid"], q["template"], "GROUP_INCOMPLETE"))
            drops[grp["scene_id"]].add("group", grp["group_id"], "GROUP_INCOMPLETE",
                                       detail="constraint / size / template diversity broken by debias pruning")
            grp["questions"] = []
    groups[:] = [g_ for g_ in groups if g_["questions"]]
    return removed


def quadrant_violation(groups, cfg):
    """Return the (template, quadrants) that need downsampling, or None if balanced. T3/T4 quadrants 25% ± tol; T4 front / back 50% ± tol each (answerable questions only)."""
    tol, fb_tol = cfg["questions"]["quadrant_balance_tol"], cfg["debias"]["t4_front_back_tol"]
    for t in ("T3", "T4"):
        qs = [q for g_ in groups for q in g_["questions"] if q["template"] == t]
        c = Counter(answer_text(q) for q in qs)
        n = sum(c.values())
        if not n:
            continue
        over = max(QUADS, key=lambda k: c[k])
        if any(abs(c[k] / n - 0.25) > tol for k in QUADS):
            return t, (over,)
        if t == "T4":
            ans = [q for q in qs if q["evidence_label"] != "unanswerable"]
            front = sum(answer_text(q) in FRONT for q in ans) / max(len(ans), 1)
            if abs(front - 0.5) > fb_tol:
                return t, FRONT if front > 0.5 else tuple(k for k in QUADS if k not in FRONT)
    return None


def rebalance_letters(groups, seed):
    """Reassign option letters exactly uniformly per (template, number of options) (reorders options, content unchanged)."""
    rng = np.random.default_rng([seed, 88])
    buckets = defaultdict(list)
    for g_ in groups:
        for q in g_["questions"]:
            if q["answer_type"] != "numeric":
                buckets[(q["template"], len(q["options"]))].append(q)
    for (t, n), qs in sorted(buckets.items()):
        pos = np.tile(np.arange(n), -(-len(qs) // n))[:len(qs)]
        rng.shuffle(pos)
        for q, p in zip(qs, pos):
            vals = q.get("option_objs") or q["option_values"]
            texts = option_texts(q)
            a = LETTERS.index(q["answer"])
            others = [k for k in range(n) if k != a]
            rng.shuffle(others)
            order = others[:p] + [a] + others[p:]
            key = "option_objs" if "option_objs" in q else "option_values"
            q[key] = [vals[k] for k in order]
            q["options"] = [f"{LETTERS[i]}. {texts[k]}" for i, k in enumerate(order)]
            q["answer"] = LETTERS[int(p)]


def run(cfg):
    dc = cfg["debias"]
    assert_not_frozen(path_of(cfg, "work") / DATASET / "_step8" / dc["output_version"])
    groups, names, src_hash = load(cfg)
    n0 = Counter(q["template"] for g_ in groups for q in g_["questions"])
    drops = {sid: DropLog(work_dir(cfg, DATASET, sid.split("_", 1)[1]), "step8") for sid in {g_["scene_id"] for g_ in groups}}
    removed_all = []
    history = []
    before = evaluate(groups, names, cfg, controls=True)
    status = "ok"
    for rnd in range(1, 100):
        res = before if rnd == 1 else evaluate(groups, names, cfg)
        history.append({t: (r.get("acc", r.get("mra")), r.get("chance", r.get("baseline")), r["n"]) for t, r in res.items()})
        bad = [t for t, r in res.items() if not r["ok"]]
        cum = Counter(t for _, t, _ in removed_all)
        over = [t for t in bad if cum[t] / n0[t] > dc["max_prune_frac_per_template"]]
        if over:
            status = f"stopped: {over} exceeded {dc['max_prune_frac_per_template']:.0%} cumulative removal without meeting the target"
            break
        if not bad and quadrant_violation(groups, cfg) is None:
            break
        kill, reasons = set(), {}
        for t in bad:
            r = res[t]
            k = max(1, math.ceil(dc["prune_step_frac"] * r["n"]))
            for j in np.argsort(-r["bias"], kind="stable")[:k]:
                kill.add(r["items"][j])
                reasons[r["items"][j]] = "BIAS_PRUNED"
        if kill:
            removed_all += remove_questions(groups, kill, reasons, drops, cfg["groups"], cfg["debias"].get("keep_group_on_constraint_loss", False))
        # Rebalance: when quadrants / T4 front-back are out of tolerance, delete the non-constraint question with the highest bias score from the over-represented quadrants (RATIO_BALANCE)
        while True:
            v = quadrant_violation(groups, cfg)
            if v is None:
                break
            t, quads = v
            ev = evaluate(groups, names, cfg)[t]
            cand = [(b, it) for b, it in zip(ev["bias"], ev["items"])
                    if answer_text(groups[it[0]]["questions"][it[1]]) in quads
                    and not any(groups[it[0]]["questions"][it[1]]["qid"] in c["qids"] for c in groups[it[0]]["constraints"])]
            if not cand:
                status = f"rebalance infeasible for {t}"
                break
            it = max(cand)[1]
            removed_all += remove_questions(groups, {it}, {it: "RATIO_BALANCE"}, drops, cfg["groups"], cfg["debias"].get("keep_group_on_constraint_loss", False))
        rebalance_letters(groups, cfg["seed"])
    if removed_all:
        rebalance_letters(groups, cfg["seed"])
    after = evaluate(groups, names, cfg, controls=True)
    stab_before = stability(*load(cfg)[:2], cfg)
    stab_after = stability(groups, names, cfg)
    for t, r in after.items():  # write the final out-of-fold bias scores
        for b, (gi, qi) in zip(r["bias"], r["items"]):
            groups[gi]["questions"][qi]["bias_score"] = round(float(b), 4)
    out = path_of(cfg, "work") / DATASET / "_step8" / dc["output_version"]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "test_groups.jsonl", "w") as f:
        for g_ in groups:
            f.write(json.dumps(g_, ensure_ascii=False) + "\n")
    manifest = {"version": dc["output_version"], "source": "work/arkitscenes/_step7/test_groups.jsonl", "source_sha256": src_hash,
                "status": status, "groups": len(groups), "questions": sum(len(g_["questions"]) for g_ in groups),
                "removed": [{"qid": q, "template": t, "reason": r} for q, t, r in removed_all], "rounds": len(history)}
    write_json(out / "manifest.json", manifest)
    report(cfg, before, after, history, removed_all, n0, groups, status, manifest, stab_before, stab_after)


def _fmt(r):
    if "mra" in r:
        return f"MRA {r['mra']:.3f}", f"baseline {r['baseline']:.3f}", f"{r['gap'] * 100:+.1f} pp", f"majority-answer MRA {r['majority']:.3f}", "—"
    return (f"{r['acc']:.1%}", f"chance {r['chance']:.1%}", f"{r['gap'] * 100:+.1f} pp", f"{r['majority']:.1%}",
            f"{r['shuffled_acc']:.1%}" if "shuffled_acc" in r else "—")


def report(cfg, before, after, history, removed, n0, groups, status, manifest, stab_before, stab_after):
    dc = cfg["debias"]
    rm = Counter((t, r) for _, t, r in removed)
    qs = [q for g_ in groups for q in g_["questions"]]
    n1 = Counter(q["template"] for q in qs)
    L = [f"# Step 8: blind tests and debiasing (ARKit only → {dc['output_version']})", "",
         "> Step 8 is run and pruning applied to the ARKit questions first, without waiting for ScanNet++; the complete pre-pruning question set "
         "(`work/arkitscenes/_step7/test_groups.jsonl`) is left unchanged, and after ScanNet++ is added Step 8 is rerun on the merged data from the pre-pruning state.", "",
         f"Result: **{manifest['status']}**; after pruning {manifest['groups']} question groups, {manifest['questions']} questions"
         f" ({sum(n0.values())} before pruning), {len(history)} rounds. Output `work/arkitscenes/_step8/{dc['output_version']}/test_groups.jsonl` and `manifest.json`.", "",
         "## Method", "",
         f"- Feature blind test: text features only (template/variant, paraphrase ID, referring name of each slot, option texts [by position and as a set], number of options), random forest with 300 trees, GroupKFold by scene with {dc['n_folds']} folds"
         + ("" if not FOLD_FALLBACKS else " (**fallback when there are too few scenes**: " + "; ".join(f"{n} scenes use {'leave-one-out' if k == n else f'{k} folds'} ({c} times)" for (n, k), c in sorted(FOLD_FALLBACKS.items())) + ")")
         + "; "
         "bias score of multiple-choice questions = out-of-fold probability of the correct option (normalized over the question's options); bias score of numeric questions = MRA score of the out-of-fold prediction. Unanswerable questions are excluded from blind tests and never pruned.",
         "- Control 1: each fold only guesses the most common answer of the training folds (multiple choice: the option text most frequent in the training folds; numeric: the most common distance in the training folds).",
         "- Control 2: rerun the blind test with each question's option order shuffled; results close to the original order mean the model does not exploit option positions.",
         "- LLM zero-shot blind test (optional): not done; no external API calls.",
         f"- Pruning: per template, each round removes the {dc['prune_step_frac']:.0%} with the highest bias scores (`BIAS_PRUNED`); a deleted question in a constraint invalidates the constraint, "
         "and groups that no longer have \"≥ 1 constraint, ≥ 3 questions, ≥ 2 question types\" are dropped entirely (`GROUP_INCOMPLETE`). Rebalancing after each round: T3/T4 quadrants 25% ± 5%, "
         "T4 (answerable) front / back 50% ± 5% each; when out of tolerance, the non-constraint question with the highest bias score is deleted from the over-represented quadrants (`RATIO_BALANCE`); option letters are reassigned uniformly per (template, number of options).",
         f"- Target: multiple-choice blind accuracy ≤ chance + {dc['mc_margin_over_chance_pp']} pp; numeric blind MRA ≤ category-pair mean baseline + {dc['mc_margin_over_chance_pp']} pp. "
         f"Stop and report if a single template has had more than {dc['max_prune_frac_per_template']:.0%} removed in total and still fails.", "",
         "## Blind-test scores per template before and after debiasing", "",
         "| Template | Stage | Questions (answerable) | Blind score | Control (chance / baseline) | Gap | Majority answer only | Options shuffled | Pass |", "|---|---|---|---|---|---|---|---|---|"]
    for t in sorted(before):
        for stage, r in (("before debiasing", before[t]), ("after debiasing", after.get(t))):
            if r is None:
                continue
            a, b, c, d, e = _fmt(r)
            L.append(f"| {t} | {stage} | {r['n']} | {a} | {b} | {c} | {d} | {e} | {'yes' if r['ok'] else '**no**'} |")
    var_lines = []
    for t, r in before.items():
        for v, x in (r.get("by_variant") or {}).items():
            if x:
                y = (after.get(t) or {}).get("by_variant", {}).get(v)
                var_lines.append(f"| {t}:{v} | {x[2]} | {x[0]:.1%} (chance {x[1]:.1%}) | "
                                 + (f"{y[2]} | {y[0]:.1%} (chance {y[1]:.1%})" if y else "— | —") + " |")
    if var_lines:
        L += ["", "Per-variant breakdown (trained separately, for reference only; pruning is decided per template as a whole):", "", "| Variant | Questions before | Blind before | Questions after | Blind after |", "|---|---|---|---|---|", *var_lines]
    L += ["", "## Stability check", "",
          "Blind scores depend on the fold split and the random forest seed. The table below assigns scenes to 5 folds at random and changes the random forest seed, repeated 5 times, and gives the mean ± std (pp) of \"blind score − control\". "
          "In the debiasing dry run (`arkit_debias_dryrun.md`, features without option positions and paraphrase IDs) T4 was +8.3 pp; this table and the one above show that a single estimate fluctuates by several percentage points.", "",
          "| Template | Before: mean ± std | Runs | After: mean ± std | Runs |", "|---|---|---|---|---|",
          *[f"| {t} | {stab_before[t][0] * 100:+.1f} ± {stab_before[t][1] * 100:.1f} | {stab_before[t][2]} | "
            + (f"{stab_after[t][0] * 100:+.1f} ± {stab_after[t][1] * 100:.1f} | {stab_after[t][2]} |" if t in stab_after else "— | — |")
            for t in sorted(stab_before)],
          "", "## Per-round log", "", "| Round | " + " | ".join(sorted(before)) + " |", "|---|" + "---|" * len(before)]
    for k, h in enumerate(history, 1):
        L.append(f"| {k} | " + " | ".join(f"{h[t][0]:.3f} / {h[t][1]:.3f} ({h[t][2]})" if t in h else "—" for t in sorted(before)) + " |")
    L += ["", "Each cell is \"blind score / control (chance or baseline) (answerable questions)\".", "",
          "## Removal statistics", "", "| Template | Before pruning | BIAS_PRUNED | RATIO_BALANCE | GROUP_INCOMPLETE (collateral) | After pruning | Cumulative removal |", "|---|---|---|---|---|---|---|",
          *[f"| {t} | {n0[t]} | {rm[(t, 'BIAS_PRUNED')]} | {rm[(t, 'RATIO_BALANCE')]} | {rm[(t, 'GROUP_INCOMPLETE')]} | {n1[t]} | "
            f"{1 - n1[t] / n0[t]:.1%} |" for t in sorted(n0)], ""]
    # Category distribution of removed questions
    cat = Counter()
    rem_q = {q for q, _, r in removed if r in ("BIAS_PRUNED", "RATIO_BALANCE")}
    src = [json.loads(l) for l in open(path_of(cfg, "work") / DATASET / "_step7" / "test_groups.jsonl")]
    for g_ in src:
        nm = {o["obj_id"]: o.get("ref_name") or o["category"]
              for o in read_json(work_dir(cfg, DATASET, g_["scene_id"].split("_", 1)[1]) / "objects.json")}
        for q in g_["questions"]:
            if q["qid"] in rem_q:
                cat.update(nm[v] for v in q["slots"].values())
    L += ["Object names in removed questions (BIAS_PRUNED + RATIO_BALANCE): " + (", ".join(f"{k} {v}" for k, v in cat.most_common()) or "none"), "",
          "## Distribution after pruning", ""]
    c4 = Counter(answer_text(q) for q in qs if q["template"] == "T4")
    a4 = [q for q in qs if q["template"] == "T4" and q["evidence_label"] != "unanswerable"]
    front = sum(answer_text(q) in FRONT for q in a4) / max(len(a4), 1)
    c3 = Counter(answer_text(q) for q in qs if q["template"] == "T3")
    L += [f"- T4 quadrants (all): " + ", ".join(f"{k} {c4[k] / max(sum(c4.values()), 1):.1%}" for k in QUADS)
          + f"; answerable T4 front {front:.1%} / back {1 - front:.1%}",
          "- T3 quadrants: " + ", ".join(f"{k} {c3[k] / max(sum(c3.values()), 1):.1%}" for k in QUADS),
          f"- Evidence labels: {dict(Counter(q['evidence_label'] for q in qs))}; unanswerable share {sum(q['evidence_label'] == 'unanswerable' for q in qs) / max(len(qs), 1):.1%}"]
    name = "step8_debias_arkit.md" if dc["output_version"] == "arkit_only_v0.9" else f"step8_debias_{dc['output_version']}.md"
    (path_of(cfg, "reports") / name).write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))


def main():
    run(load_config())


if __name__ == "__main__":
    main()
