"""Consistency check: compares, item by item, the results of the same model and run name in two places (a reference results
snapshot and the results directory on a new machine).

    python scripts/compare_runs.py --ref <package dir>/results_snapshot --new <results dir> --model RoboBrain2.0-7B --run smoke1_r0

Per setting:
- Input: whether the request content hashes (request messages in calls_digest.jsonl / calls.jsonl: text + SHA-256 of each image)
  agree for the same group and turn (group_id, t). Later Active turns depend on the model's earlier actions, so only turns reached by
  both sides with identical earlier actions are compared.
- Output: per-question answer agreement, per-question score agreement, difference of the mean scores.
Verdict (RoboBrain2.0-7B, temperature 1e-6, nearly greedy): inputs must be 100% identical (a difference means the data, photos or
prompts differ and must be investigated first); answer agreement >= 80% for static settings (blind, passive, oracle, random_walk) and
>= 60% for Active, with a mean score difference within ±0.15, is normal (floating-point operation order differs across GPUs and batch
compositions, so even greedy decoding diverges on individual tokens; the smoke test has only 3 groups, about 12–14 questions, so one
question is 7–8 percentage points).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

SETTINGS = ["blind", "passive_8", "passive_20", "passive_40", "random_walk_40", "oracle_min", "oracle_route", "active_v2"]


def requests(d: Path) -> dict:
    p = d / "calls.jsonl" if (d / "calls.jsonl").exists() else d / "calls_digest.jsonl"
    out = {}
    if not p.exists():
        return out
    for line in open(p):
        r = json.loads(line)
        if r.get("type") != "call":
            continue
        h = r.get("request_sha256")
        if h is None and isinstance(r.get("request"), dict):
            h = hashlib.sha256(json.dumps(r["request"]["messages"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        out.setdefault((r.get("group_id"), (r.get("tag") or {}).get("t")), h)      # retries: only the first attempt counts
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--model", default="RoboBrain2.0-7B")
    ap.add_argument("--run", default="smoke1_r0")
    a = ap.parse_args()
    rows, ok_all = [], True
    print("| Setting | Identical inputs (turns) | Same answer | Same score | Mean score ref / new | Diff | Verdict |\n|---|---|---|---|---|---|---|")
    for s in SETTINGS:
        A, B = Path(a.ref) / a.model / s / a.run, Path(a.new) / a.model / s / a.run
        if not (A / "summary.json").exists() or not (B / "summary.json").exists():
            print(f"| {s} | — | — | — | — | — | missing run ({'ref' if not (A / 'summary.json').exists() else 'new'}) |")
            ok_all = False
            continue
        qa = {q["qid"]: q for q in json.loads((A / "summary.json").read_text())["per_question"]}
        qb = {q["qid"]: q for q in json.loads((B / "summary.json").read_text())["per_question"]}
        common = sorted(set(qa) & set(qb))
        same_ans = sum(qa[k].get("answer") == qb[k].get("answer") for k in common)
        same_sc = sum(qa[k]["score"] == qb[k]["score"] for k in common)
        ma = sum(qa[k]["score"] for k in common) / len(common)
        mb = sum(qb[k]["score"] for k in common) / len(common)
        ra, rb = requests(A), requests(B)
        keys = sorted(set(ra) & set(rb), key=str)
        if s == "active_v2":      # compare only up to the first differing input (later turns depend on the model's actions)
            by_g = {}
            for g, t in keys:
                by_g.setdefault(g, []).append(t)
            keys = []
            for g, ts in by_g.items():
                for t in sorted(ts):
                    keys.append((g, t))
                    if ra[(g, t)] != rb[(g, t)]:
                        break
            in_same = sum(ra[k] == rb[k] for k in keys if k[1] == 0)
            in_n = sum(1 for k in keys if k[1] == 0)
        else:
            in_same, in_n = sum(ra[k] == rb[k] for k in keys), len(keys)
        rate = same_ans / len(common)
        good = in_same == in_n and in_n > 0 and rate >= (0.6 if s == "active_v2" else 0.8) and abs(ma - mb) <= 0.15
        ok_all &= good
        print(f"| {s} | {in_same}/{in_n} | {same_ans}/{len(common)} ({rate:.0%}) | {same_sc}/{len(common)} | {ma:.3f} / {mb:.3f} | {mb - ma:+.3f} | {'OK' if good else '**CHECK**'} |")
    print(f"\nOverall: {'OK' if ok_all else 'some settings are out of range, see the table above'} (for Active only turn-0 inputs are compared)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
