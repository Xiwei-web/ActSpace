"""Scoring and aggregation (docs/evaluation_protocol.md).

Input: summary.json produced by run.py (per_question, per_episode) and the matching trajectories.jsonl (answers, used for consistency).
- Accuracy: mean of per-question scores (answers.score_question); also reported separately: multiple-choice accuracy (answerable T2–T5), T1 MRA and unanswerable accuracy;
  broken down by template, difficulty and masking (masked unanswerable question / group containing a mask).
- Answering: answer rate = share of answerable questions with an answer (neither cannot_determine nor missing); selective score = mean score over answered answerable questions only;
  correct abstention rate = share of unanswerable questions answered cannot_determine (same as unanswerable accuracy, listed separately to contrast with the answer rate).
- Efficiency (Active): steps used; efficiency = group_route_steps / steps used (capped at 1), computed only for groups where all evidence was seen; evidence hit rate;
  secondary reference: steps used / max oracle_steps in the group.
- Evidence hit (diagnostic): share of answerable questions "correct with evidence seen" and "correct without evidence seen" (correct = right option; T1 gets MRA split by evidence seen).
- Consistency (diagnostic): for groups with constraints, whether the constraint holds on the agent's answers (missing answer, cannot_determine or equal numbers → "undeterminable").
- Derived: exploration gain = Active − Passive-k, evidence gap = Oracle − Active (same agent, same budget).
- Statistics: scene-level bootstrap (resample scenes with replacement, 2000 draws, fixed seed), 95% intervals; derived quantities are paired on the same draws.
- For each budget, the number of groups whose group-level shortest route fits in the budget, plus scores on those groups only.
- Excluded questions: degenerate questions listed in configs/eval_exclude.yaml are dropped from the main tables (exclude_questions); an appendix table gives unfiltered results;
  constraints involving excluded questions are not counted in consistency.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from actspace_build.common import ROOT
from actspace_build.geometry import INVERSE_QUADRANT

from .answers import CANNOT
from .data import FrozenData
from .meta import feasible_groups, group_meta
from .trajectory import read_episodes

N_BOOT = 2000
LETTERS = "ABCDEFGH"


def load_run(root: Path, name: str) -> dict:
    s = json.loads((root / name / "summary.json").read_text())
    answers = {}
    for ep in read_episodes(root / name / "trajectories.jsonl"):
        if ep[-1]["type"] != "episode_end":     # while a run is unfinished (tiny draft) the last episode may be half written
            continue
        qmap = ep[0]["question_ids"]
        answers.update({qmap[k]: v for k, v in ep[-1]["answers"].items()})
    s["answers_by_qid"] = answers
    return s


def load_exclusions(path: Path | None = None) -> dict:
    """{qid: reason}: questions excluded from the main report tables (configs/eval_exclude.yaml)."""
    d = yaml.safe_load(open(path or ROOT / "configs" / "eval_exclude.yaml"))
    return {x["qid"]: x["reason"] for x in d["questions"]}


MIN_HORIZONTAL_SEP_M = 0.5   # question-generation rule for the next data version (open question Q28), "standing point only"


def horizontal_sep_violations(data, sep: float = MIN_HORIZONTAL_SEP_M) -> dict:
    """{qid: (min horizontal distance, description)}: direction questions violating the minimum horizontal distance rule.
    T3 / T5: horizontal distance between the standing object OBJ1 and the facing object, target object and every option object (option_objs); T4: between the object and the START camera.
    The list in configs/eval_exclude.yaml should match the result on v0.11 exactly."""
    out = {}
    for g in data.groups.values():
        sc = data.scene(g["scene_id"])
        objs = {o["obj_id"]: o for o in json.load(open(sc.wd / "objects.json"))}
        xy = lambda o: np.asarray(objs[o]["center"][:2], float)
        for q in g["questions"]:
            if q["template"] == "T4":
                cam = np.asarray(sc.pose[g["init_frame"]][0][:2], float)
                d = [(float(np.linalg.norm(xy(q["slots"]["OBJ1"]) - cam)), "OBJ1–START camera")]
            elif q["template"] in ("T3", "T5"):
                s1 = q["slots"]["OBJ1"]
                others = dict.fromkeys(list(q["slots"].values()) + list(q.get("option_objs") or []))
                d = [(float(np.linalg.norm(xy(s1) - xy(o))), f"OBJ1–{objs[o]['ref_name']}#{o}") for o in others if o != s1]
            else:
                continue
            m = min(d)
            if m[0] < sep:
                out[q["qid"]] = m
    return out


def exclude_questions(run: dict, qids) -> dict:
    """Drop per-question records of the given questions (per-group records unchanged); remember them so consistency skips constraints involving them."""
    qids = set(qids)
    return {**run, "per_question": [x for x in run["per_question"] if x["qid"] not in qids],
            "excluded_qids": sorted(qids | set(run.get("excluded_qids", ())))}


# ---------------------------------------------------------------- consistency

def _choice(q, letter):
    if letter in (None, CANNOT) or letter not in LETTERS[:len(q["options"])]:
        return None
    k = LETTERS.index(letter)
    return q["option_objs"][k] if "option_objs" in q else q["option_values"][k]


def constraint_holds(con, qby, ans):
    """Whether a constraint holds on the agent's answers: True / False / None (undeterminable). Same semantics as the check in integrity.py."""
    qs = [qby[i] for i in con["qids"]]
    if con["type"] == "numeric_vs_comparison":
        q1, q2, q3 = qs
        a1, a2, c = ans.get(q1["qid"]), ans.get(q2["qid"]), _choice(q3, ans.get(q3["qid"]))
        if not all(isinstance(x, (int, float)) for x in (a1, a2)) or c is None or a1 == a2:
            return None
        return c == (q1["slots"]["OBJ2"] if a1 < a2 else q2["slots"]["OBJ2"])
    if con["type"] == "direction_inverse":
        c1, c2 = (_choice(q, ans.get(q["qid"])) for q in qs)
        if c1 is None or c2 is None:
            return None
        return INVERSE_QUADRANT[c1] == c2
    closer = {}  # distance_transitivity
    for q in qs:
        w = _choice(q, ans.get(q["qid"]))
        if w is None:
            return None
        o1, o2 = q["option_objs"]
        closer[(w, o2 if w == o1 else o1)] = True
    objs = {o for q in qs for o in q["option_objs"]}
    return not any(closer.get((x, y)) and closer.get((y, z)) and closer.get((z, x)) for x, y, z in itertools.permutations(objs, 3))


def consistency(data, run):
    res = []
    gids = {e["group_id"] for e in run["per_episode"]}
    ex = set(run.get("excluded_qids", ()))
    for gid in gids:
        g = data.groups[gid]
        qby = {q["qid"]: q for q in g["questions"]}
        for con in g["constraints"]:
            if ex & set(con["qids"]):
                continue
            res.append((g["scene_id"], con["type"], constraint_holds(con, qby, run["answers_by_qid"])))
    return res


# ---------------------------------------------------------------- bootstrap

class Boot:
    """Scene-level bootstrap: all metrics share the same scene draws (fixed seed), so derived quantities are paired."""

    def __init__(self, scenes, n=N_BOOT, seed=20261102):
        self.scenes = sorted(scenes)
        self.idx = {s: i for i, s in enumerate(self.scenes)}
        rng = np.random.default_rng(seed)
        # number of times each scene is drawn in each resample (n × number of scenes)
        draws = rng.integers(len(self.scenes), size=(n, len(self.scenes)))
        self.w = np.stack([np.bincount(d, minlength=len(self.scenes)) for d in draws])

    def agg(self, items):
        """items: [(scene, value)] → (sum per scene, count per scene)."""
        s, c = np.zeros(len(self.scenes)), np.zeros(len(self.scenes))
        for sc, v in items:
            s[self.idx[sc]] += v
            c[self.idx[sc]] += 1
        return s, c

    def mean_ci(self, items):
        if not items:
            return None
        s, c = self.agg(items)
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = (self.w @ s) / (self.w @ c)
        bs = bs[np.isfinite(bs)]
        return {"mean": float(s.sum() / c.sum()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n": int(c.sum())}

    def diff_ci(self, items_a, items_b):
        """mean(a) − mean(b), paired resampling."""
        if not items_a or not items_b:
            return None
        sa, ca = self.agg(items_a)
        sb, cb = self.agg(items_b)
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = (self.w @ sa) / (self.w @ ca) - (self.w @ sb) / (self.w @ cb)
        bs = bs[np.isfinite(bs)]
        return {"mean": float(sa.sum() / ca.sum() - sb.sum() / cb.sum()), "lo": float(np.percentile(bs, 2.5)),
                "hi": float(np.percentile(bs, 97.5)), "n": int(ca.sum())}


# ---------------------------------------------------------------- metrics

class RepBoot:
    """Combined "scene bootstrap + repeats" interval (docs/evaluation_protocol.md). Each resample has two steps:
    1. Scenes: resample scenes with replacement within each dataset (pooled scores are stratified by dataset, keeping each dataset's scene count); all settings share the same scene draws, so differences are paired;
    2. Repeats: settings with R runs (Active and random_walk, 3 each) independently draw R runs with replacement from their R runs (one independent random stream per setting,
       derived from the seed and the setting name, reproducible); settings with a single run are not resampled, and their interval excludes model sampling randomness.
    Point estimate: mean over all runs pooled (one item per question per run).
    scope: only scenes of one dataset (None = pooled)."""

    def __init__(self, scenes_by_dataset: dict, n=N_BOOT, seed=20261103, scope=None):
        self.n, self.seed = n, seed
        ds_list = [scope] if scope else sorted(scenes_by_dataset)
        self.scenes = [s for ds in ds_list for s in sorted(scenes_by_dataset[ds])]
        self.idx = {s: i for i, s in enumerate(self.scenes)}
        rng = np.random.default_rng([seed, 1])
        w = np.zeros((n, len(self.scenes)))
        off = 0
        for ds in ds_list:
            k = len(scenes_by_dataset[ds])
            draws = rng.integers(k, size=(n, k)) + off
            for b in range(n):
                w[b] += np.bincount(draws[b], minlength=len(self.scenes))
            off += k
        self.w = w

    def _agg(self, items):
        s, c = np.zeros(len(self.scenes)), np.zeros(len(self.scenes))
        for sc, v in items:
            i = self.idx.get(sc)
            if i is not None:
                s[i] += v
                c[i] += 1
        return s, c

    def series(self, runs_items: list, key: str):
        """runs_items: [[(scene, value)] per run]. Returns (point estimate, value per resample, number of questions)."""
        S, C = zip(*(self._agg(items) for items in runs_items))
        S, C = np.stack(S), np.stack(C)              # R × scenes
        R = len(runs_items)
        if C.sum() == 0:
            return None, None, 0
        point = float(S.sum() / C.sum())
        if R == 1:
            cnt = np.ones((self.n, 1))
        else:
            h = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
            draws = np.random.default_rng([self.seed, 2, h]).integers(R, size=(self.n, R))
            cnt = np.stack([np.bincount(d, minlength=R) for d in draws])   # n × R
        num = np.einsum("br,bs,rs->b", cnt, self.w, S)
        den = np.einsum("br,bs,rs->b", cnt, self.w, C)
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = num / den
        return point, bs, int(C.sum() / R)

    @staticmethod
    def _ci(point, bs, n):
        if point is None:
            return None
        bs = bs[np.isfinite(bs)]
        return {"mean": point, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n": n}

    def mean_ci(self, runs_items, key):
        return self._ci(*self.series(runs_items, key))

    def diff_ci(self, runs_a, key_a, runs_b, key_b):
        pa, ba, na = self.series(runs_a, key_a)
        pb, bb, _ = self.series(runs_b, key_b)
        if pa is None or pb is None:
            return None
        return self._ci(pa - pb, ba - bb, na)


def question_items(run, sel=lambda x: True, groups=None):
    return [(x["scene_id"], x["score"]) for x in run["per_question"] if sel(x) and (groups is None or x["group_id"] in groups)]


SELECTORS = {
    "all questions": lambda x: True,
    "MC accuracy (T2–T5, answerable)": lambda x: x["answerable"] and x["template"] != "T1",
    "T1 MRA (answerable)": lambda x: x["answerable"] and x["template"] == "T1",
    "unanswerable accuracy": lambda x: not x["answerable"],
}


def answered(x) -> bool:
    """An answer was given: not cannot_determine and not missing."""
    return x["answer"] is not None and x["answer"] != CANNOT


def answer_metrics(run, boot, groups=None) -> dict:
    """Answer rate, selective score, correct abstention rate."""
    sel = lambda f: [x for x in run["per_question"] if f(x) and (groups is None or x["group_id"] in groups)]
    ans, unans = sel(lambda x: x["answerable"]), sel(lambda x: not x["answerable"])
    return {"answer rate (answerable)": boot.mean_ci([(x["scene_id"], float(answered(x))) for x in ans]),
            "selective score (answered answerable)": boot.mean_ci([(x["scene_id"], x["score"]) for x in ans if answered(x)]),
            "correct abstention (unanswerable)": boot.mean_ci([(x["scene_id"], float(x["answer"] == CANNOT)) for x in unans])}


def breakdowns(run, boot, groups=None):
    out = {}
    for t in ("T1", "T2", "T3", "T4", "T5"):
        for ans in (True, False):
            items = question_items(run, lambda x: x["template"] == t and x["answerable"] == ans, groups)
            if items:
                out[f"{t} {'answerable' if ans else 'unanswerable'}"] = boot.mean_ci(items)
    for d in ("easy", "medium", "hard"):
        out[f"difficulty {d}"] = boot.mean_ci(question_items(run, lambda x: x["difficulty"] == d and x["answerable"], groups))
    out["masked questions (unanswerable)"] = boot.mean_ci(question_items(run, lambda x: not x["answerable"], groups))
    out["answerable questions in masked groups"] = boot.mean_ci(question_items(run, lambda x: x["masked_group"] and x["answerable"], groups))
    out["questions in unmasked groups"] = boot.mean_ci(question_items(run, lambda x: not x["masked_group"], groups))
    return out


def evidence_diag(run, groups=None):
    qs = [x for x in run["per_question"] if x["answerable"] and (groups is None or x["group_id"] in groups)]
    mc = [x for x in qs if x["template"] != "T1"]
    t1 = [x for x in qs if x["template"] == "T1"]
    f = lambda xs: float(np.mean(xs)) if xs else None
    return {"evidence_hit_questions": f([x["evidence_hit"] for x in qs]),
            "mc_correct_and_seen": f([x["score"] == 1 and x["evidence_hit"] for x in mc]),
            "mc_correct_not_seen": f([x["score"] == 1 and not x["evidence_hit"] for x in mc]),
            "mc_acc_given_seen": f([x["score"] for x in mc if x["evidence_hit"]]),
            "mc_acc_given_not_seen": f([x["score"] for x in mc if not x["evidence_hit"]]),
            "t1_mra_seen": f([x["score"] for x in t1 if x["evidence_hit"]]),
            "t1_mra_not_seen": f([x["score"] for x in t1 if not x["evidence_hit"]]), "n_mc": len(mc), "n_t1": len(t1)}


def efficiency_stats(run, boot, groups=None):
    eps = [e for e in run["per_episode"] if groups is None or e["group_id"] in groups]
    eff = [(e["scene_id"], e["efficiency"]) for e in eps if e["efficiency"] is not None]
    return {"mean_steps": float(np.mean([e["steps_used"] for e in eps])),
            "groups_all_evidence_seen": sum(e["evidence_hit_all"] for e in eps), "groups": len(eps),
            "group_evidence_hit_rate": boot.mean_ci([(e["scene_id"], float(e["evidence_hit_all"])) for e in eps]),
            "efficiency": boot.mean_ci(eff),
            "steps_over_oracle_steps_max": float(np.median([e["steps_used"] / max(e["oracle_steps_max"], 1) for e in eps]))}


def run_metrics(data, run, boot, groups=None):
    m = {k: boot.mean_ci(question_items(run, f, groups)) for k, f in SELECTORS.items()}
    m.update(answer_metrics(run, boot, groups))
    m["breakdown"] = breakdowns(run, boot, groups)
    m["evidence"] = evidence_diag(run, groups)
    cons = consistency(data, run)  # consistency is computed over all groups only
    det = [c for c in cons if c[2] is not None]
    m["consistency"] = {"constraints": len(cons), "determinable": len(det),
                        "holds_frac": float(np.mean([c[2] for c in det])) if det else None,
                        "by_type": {t: {"n": sum(c[1] == t for c in det), "holds": sum(c[1] == t and c[2] for c in det)}
                                    for t in sorted({c[1] for c in cons})}}
    if run["mode"].startswith("active"):
        m["efficiency"] = efficiency_stats(run, boot, groups)
    return m


# ---------------------------------------------------------------- stage-2 suite

def stage2(root: Path, agents, budgets=(10, 20, 40), prefix="s2"):
    data = FrozenData()
    meta = group_meta(data)
    boot = Boot({g["scene_id"] for g in data.groups.values()})
    runs, res = {}, {"budgets": list(budgets), "agents": list(agents), "n_boot": N_BOOT,
                     "feasible": {b: len(feasible_groups(meta, b)) for b in budgets}, "metrics": {}, "derived": {}}
    for a in agents:
        names = {("blind", None): f"{prefix}_{a}_blind", ("oracle", None): f"{prefix}_{a}_oracle"}
        for b in budgets:
            names[("passive", b)] = f"{prefix}_{a}_passive_b{b}"
            names[("active", b)] = f"{prefix}_{a}_active_b{b}"
        for k, n in names.items():
            runs[(a, *k)] = load_run(root, n)
            res["metrics"][f"{a}|{k[0]}|{k[1]}"] = {"all": run_metrics(data, runs[(a, *k)], boot)}
        for b in budgets:
            feas = feasible_groups(meta, b)
            for k in (("passive", b), ("active", b), ("blind", None), ("oracle", None)):
                res["metrics"][f"{a}|{k[0]}|{k[1]}"].setdefault("feasible", {})[b] = {
                    "all questions": boot.mean_ci(question_items(runs[(a, *k)], lambda x: True, feas))}
            act, pas, orc = runs[(a, "active", b)], runs[(a, "passive", b)], runs[(a, "oracle", None)]
            d = {}
            for label, groups in (("all", None), ("feasible", feas)):
                d[label] = {"exploration_gain": boot.diff_ci(question_items(act, groups=groups), question_items(pas, groups=groups)),
                            "evidence_gap": boot.diff_ci(question_items(orc, groups=groups), question_items(act, groups=groups))}
            res["derived"][f"{a}|{b}"] = d
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=["random", "scripted_oracle", "seen_oracle"])
    ap.add_argument("--prefix", default="s2")
    ap.add_argument("--out", required=True, help="result JSON")
    args = ap.parse_args()
    data_root = FrozenData(verify=False).ecfg["paths"]["eval_runs"]
    res = stage2(Path(data_root), args.agents, prefix=args.prefix)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "metrics"}, ensure_ascii=False, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
