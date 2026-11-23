"""ScanNet++ category vocabulary draft (Step 3, item 2; GATE B review material).

Two-level mapping (every raw label gets a category at both levels; Step 3 picks one via scannetpp.vocab_granularity,
set to merged_unique_fine_name at GATE B):
- Fine level, category_fine: semantic_map_to from the official map_benchmark.csv (spelling, plurals, synonyms) -> this
  module's SYNONYMS (misspellings not covered officially, different names for the same object) -> ARKit naming (tv -> TV).
  Plural labels (one annotation covering several objects) are marked as group.
- Merged level, category_merged: on top of the fine level, subclasses are merged into parents (FAMILIES, e.g.
  office chair -> chair).

Base categories allowed in questions: the official instance benchmark classes (top100_instance.txt, 84 classes) plus the
remaining classes of the 17 reviewed ARKit classes, minus ambiguous ones (AMBIGUOUS). Parents that appear only after
merging (lamp) are also allowed; subclasses of an allowed parent inherit the permission (e.g. copier via printer) so
questions can name them by their fine name. Categories banned at GATE B (GATEB_BANNED, including their subclasses) are
not allowed. All other categories (structural classes, long-tail labels) are not used in questions but still count
towards uniqueness.
name_ok: categories named by their fine name in questions (NAME_FINE). By default questions use the parent (merged)
name; the fine name is used only when the parent name would be a misnomer.

Usage: python -m actspace_build.scannetpp_vocab build
Outputs: vocab/label_mapping_scannetpp.tsv, vocab/categories_scannetpp_draft.tsv, vocab/category_frequency_scannetpp.tsv
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

import pandas as pd

from .common import load_config, path_of
from .scannetpp import read_split, sp_cfg, sp_root

# Annotation-tool operation markers, not objects: excluded from object instances and uniqueness counts (only counted in reports)
ARTIFACT_LABELS = {"split", "remove"}

# Misspellings / alternative names for the same object not covered by the official map_benchmark (fine level). Keys are class names after the official mapping.
SYNONYMS = {
    "celing lamp": "ceiling lamp", "tubelight": "ceiling lamp", "tube lights": "ceiling lamp",
    "desk lamp": "table lamp", "bedside lamp": "table lamp",
    "garbage bin": "trash can", "bin": "trash can",
    "wall outlet": "socket", "power socker": "socket", "covered power socket": "socket", "power sockets": "socket",
    "cardbox": "box", "carton": "box",
    "plastic bottle": "bottle", "chemical bottle": "bottle", "bottles": "bottle", "bottle spray": "spray bottle",
    "computer desk": "table", "joined tables": "table",
    "book shelf": "bookshelf", "shelve": "shelf",
    "storage ecabinet": "storage cabinet",
    "faucet": "tap", "water tap": "tap",
    "wall clock": "clock", "phone": "telephone", "pc": "computer tower",
    "powerstrip": "power strip", "multiplug": "power strip",
    "chopping board": "cutting board", "cutti̇ng board": "cutting board",
    "smok alarm": "smoke detector", "hand soap dispenser": "soap dispenser",
    "door frame": "doorframe", "electric stove": "stove", "stove top": "stove",
    "water kettle": "kettle", "washbasin": "sink", "nightstand": "bedside table",
    "copy machine": "copier", "photocopy machine": "copier", "flat panel display": "monitor",
    "hairdryer": "hair dryer", "tooth paste": "toothpaste", "scissor": "scissors", "post it": "post-it",
    "doormat": "door mat", "headphone": "headphones", "mousepad": "mouse pad",
    "toilet paper roll": "toilet paper", "toilet paper rolls": "toilet paper", "light switches": "light switch",
    "arm chair": "armchair", "stools": "stool", "towels": "towel", "plates": "plate", "papers": "paper",
    "files": "file", "folders": "folder", "pens": "pen", "paper towels": "paper towel", "kitchen utensils": "kitchen utensil",
    "utensils": "kitchen utensil", "objects": "object", "fake ceiling": "ceiling", "false ceiling": "ceiling",
    "tv": "TV",
}

# Plural labels where one annotation covers several objects (in addition to those with official instance_map_to = None)
GROUP_LABELS = {"books", "bottles", "power sockets", "papers", "files", "folders", "towels", "stools", "plates", "pens",
                "utensils", "kitchen utensils", "toilet paper rolls", "tube lights", "paper towels", "joined tables",
                "light switches", "magnets", "postcards", "rods", "tools", "objects", "cables"}

# Merged level: parent -> subclasses (keys and values are fine category names)
FAMILIES = {
    "chair": ["office chair", "armchair", "dining chair", "folding chair", "rolling chair", "piano chair", "sofa chair", "ottoman chair"],
    "stool": ["bar stool", "footstool"],
    "cabinet": ["kitchen cabinet", "storage cabinet", "bathroom cabinet", "bath cabinet", "bedside cabinet", "first aid cabinet"],
    "shelf": ["bookshelf", "bathroom shelf", "recessed shelf"],
    "table": ["coffee table", "dining table", "bedside table"],
    "lamp": ["table lamp", "floor lamp", "standing lamp", "wall lamp", "mirror lamp"],  # ceiling lamp stays a separate class (GATE B)
    "TV": ["monitor"],
    "picture": ["painting", "poster", "photo", "photo frame"],
    "bottle": ["soap bottle", "shampoo bottle", "detergent bottle", "spray bottle", "beverage bottle"],
    "bag": ["backpack", "paper bag", "tote bag", "duffle bag", "delivery bag", "cosmetic bag"],
    "shoes": ["slippers", "boot", "flip flop"],
    "printer": ["copier"],
    "towel": ["kitchen towel"],
    "whiteboard": ["interactive whiteboard"],
    "speaker": ["bluetooth speaker", "portable speaker", "ceiling speaker"],
    "sofa": ["folding sofa"],
    "clock": ["alarm clock"],
    "pot": ["plant pot"],  # plant pot and pot share one parent for uniqueness (plant pot is still banned from questions)
}
# Look-alike pairs: categories that look like the same thing in images but where one is banned and the other allowed.
# Except for part-of relations (bed-mattress / bedframe, window-window frame), each pair shares one parent for uniqueness;
# the banned side stays banned from questions (it does not inherit the parent's permission).
# plastic case could belong to both box and crate; it can have only one parent, so it goes to box.
LOOKALIKE = {
    "box": ["cardboard", "plastic case", "tissue box", "tool box", "cereal box", "first aid box", "safe box", "container", "case", "egg carton"],
    "crate": ["bottle crate"],
    "basket": ["laundry basket"],
    "pot": ["bowl", "cooking pot", "electric pot", "instant pot", "moka pot"],
    "cabinet": ["wardrobe", "fitted wardrobe", "closet", "foldable closet", "kitchen drawer"],
    "table": ["counter", "kitchen counter", "bathroom counter", "bedside counter", "workbench", "laboratory bench"],
    "whiteboard": ["board", "wall board", "pinboard", "blackboard", "pegboard"],
    "lamp": ["light", "light fixture", "spotlight", "softbox light"],
    "shelf": ["storage rack", "paper rack", "magazine rack"],
    "coat hanger": ["coat rack", "clothes rack", "clothes hanger", "cloth hanger"],
    "curtain": ["shower curtain"],
    "blanket": ["duvet", "comforter"],
    "door": ["sliding door"],
    "picture": ["frame"],
    "towel": ["paper towel"],
}
LOOKALIKE_CHILDREN = {c for cs in LOOKALIKE.values() for c in cs}
for _p, _cs in LOOKALIKE.items():
    FAMILIES.setdefault(_p, [])
    FAMILIES[_p] += [c for c in _cs if c not in FAMILIES[_p]]
NEW_PARENTS = {"lamp"}  # parents that appear only after merging (not base categories); allowed in questions

ARKIT17 = ["bathtub", "bed", "cabinet", "chair", "dishwasher", "fireplace", "oven", "refrigerator", "shelf", "sink", "sofa",
           "stool", "stove", "table", "toilet", "TV", "washing machine"]
# Official instance classes too ambiguous to serve as referring expressions
# NOTE: the Chinese reason strings in AMBIGUOUS, GATEB_BANNED, NAME_FINE and disallowed_reason() are written verbatim to
# the `note` column of vocab/categories_scannetpp_draft.tsv, which is hashed by the frozen dataset; keep them byte-identical.
# English glosses are given in the comments.
AMBIGUOUS = {"container": "含义模糊（任何容器都可称 container）"}  # "ambiguous (any container can be called a container)"
# Categories banned from questions by the construction guide, plus structural / miscellaneous top100 classes that are not instance classes
BANNED = {"wall", "floor", "ceiling", "object", "stuff", "unknown", "misc"}
# Categories banned from questions at GATE B (still counted for uniqueness); when a parent is banned, its subclasses are
# banned too (subclasses of bottle, bag, shoes). Reason glosses: "too small"; "easily moved clutter / personal items";
# "almost the same position as plant"; ceiling lamp: "separate class and banned (light strips up to ~3 m long, centre hard
# to judge; present in almost every scene, merging into lamp would make every table lamp non-unique)"; blanket: "blankets
# lie on the bed, box almost coincides with the bed".
GATEB_BANNED = {
    **{c: "GATE B：太小" for c in ["light switch", "socket", "smoke detector", "mouse", "cup", "jar", "marker", "stapler",
                                   "toilet paper", "exhaust fan"]},
    **{c: "GATE B：容易被挪动的杂物 / 个人物品" for c in ["bottle", "bowl", "book", "whiteboard eraser", "headphones", "spray bottle",
                                                     "tissue box", "paper towel", "paper bag", "shoes", "slippers", "jacket", "bag",
                                                     "backpack", "binder"]},
    "plant pot": "GATE B：与 plant 位置几乎相同",
    "ceiling lamp": "GATE B：单独成类且禁止（灯带最长约 3 m，中心难判断；几乎每个场景都有，并入 lamp 会让台灯全部不唯一）",
    "blanket": "S30（负责人 2026-11-03）：毯子都铺在床上，框与床几乎重合",
}
# Questions name objects by the parent name by default (e.g. bookshelf -> the shelf); the fine name is used only when the
# parent name would be a misnomer. Current exceptions (notes: "calling it a TV is a misnomer", "calling it a printer is a misnomer"):
NAME_FINE = {"monitor": "称 TV 是误称", "copier": "称 printer 是误称"}
# Names in questions are always singular (Step 7 keeps is / are consistent). Among the names used, only blinds is plural
DISPLAY_NAME = {"blinds": "window blind"}


def norm(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())


def load_official(cfg: dict):
    md = sp_root(cfg) / "metadata" / "semantic_benchmark"
    top100 = [norm(x) for x in (md / "top100.txt").read_text().splitlines() if x.strip()]
    top_inst = [norm(x) for x in (md / "top100_instance.txt").read_text().splitlines() if x.strip()]
    m = pd.read_csv(md / "map_benchmark.csv", dtype=str, keep_default_na=False)
    sem = {norm(r["class"]): norm(r.semantic_map_to) for _, r in m.iterrows() if r.semantic_map_to}
    inst_none = {norm(r["class"]) for _, r in m.iterrows() if r.instance_map_to == "None"}
    return top100, top_inst, sem, inst_none


def map_label(label: str, official) -> dict:
    """Raw label -> {kind, benchmark_class, category_fine, category_merged, rule}."""
    top100, top_inst, sem, inst_none = official
    n = norm(label)
    if n in ARTIFACT_LABELS:
        return {"kind": "artifact", "benchmark_class": "", "category_fine": "", "category_merged": "", "rule": "annotation_artifact"}
    bench = sem.get(n, n)
    rules = ["official_map"] if n in sem else []
    fine = SYNONYMS.get(bench, SYNONYMS.get(n, bench))
    if fine != bench:
        rules.append("synonym")
    parent = {c: p for p, cs in FAMILIES.items() for c in cs}
    merged = parent.get(fine, fine)
    if merged != fine:
        rules.append("family")
    kind = "group" if (n in inst_none or n in GROUP_LABELS) else "object"
    return {"kind": kind, "benchmark_class": bench if bench in top100 else "", "category_fine": fine, "category_merged": merged,
            "rule": "+".join(rules) or "identity"}


def base_categories(official) -> dict[str, str]:
    """Base categories allowed in questions -> source."""
    _, top_inst, _, _ = official
    out = {}
    for c in top_inst:
        c2 = SYNONYMS.get(c, c)
        if c2 not in AMBIGUOUS:
            out[c2] = "scannetpp_top100_instance"
    for c in ARKIT17:
        out.setdefault(c, "arkit17")
        if out[c] != "arkit17":
            out[c] += "+arkit17"
    for c in NEW_PARENTS:
        out[c] = "merged_parent"
    return out


def allowed_categories(cats, base: dict) -> dict[str, str]:
    """Categories allowed in questions -> source: a base category, or inherited from an allowed parent (inherit:<parent>); excludes GATE B banned categories and subclasses of banned parents."""
    parents = {c: p for p, cs in FAMILIES.items() for c in cs}
    out = {}
    for c in cats:
        p = parents.get(c)
        if c in GATEB_BANNED or p in GATEB_BANNED or c in LOOKALIKE_CHILDREN:  # the banned side of a look-alike pair stays banned
            continue
        if c in base:
            out[c] = base[c]
        elif p in base:
            out[c] = f"inherit:{p}"
    return out


def disallowed_reason(cat: str, official) -> str:
    top100, top_inst, _, _ = official
    parent = {c: p for p, cs in FAMILIES.items() for c in cs}.get(cat)
    if cat in GATEB_BANNED:
        return GATEB_BANNED[cat]
    # Persisted note strings (see NOTE above); glosses: "(inherited from parent {parent})", "category banned by the guide",
    # "structural / misc class in official top100 (not in the instance benchmark)",
    # "not a base category (84 official instance benchmark classes + 17 ARKit classes)".
    if parent in GATEB_BANNED:
        return f"{GATEB_BANNED[parent]}（随父类 {parent}）"
    if cat in BANNED:
        return "指南禁止出题类别"
    if cat in AMBIGUOUS:
        return AMBIGUOUS[cat]
    if cat in top100 and cat not in top_inst:
        return "官方 top100 中的结构 / 杂项类别（不在实例基准中）"
    return "不在基础类别中（官方实例基准 84 类 + ARKit 17 类）"


def raw_label_stats(cfg: dict) -> tuple[list[str], dict[str, list[str]]]:
    """Read (read-only) segments_anno.json of every scene in the split: list of raw labels per scene."""
    scenes = read_split(cfg)
    labels = {}
    for sid in scenes:
        a = json.load(open(sp_root(cfg) / "data" / sid / "scans" / "segments_anno.json"))
        labels[sid] = [g["label"] for g in a["segGroups"]]
    return scenes, labels


def frequency(per_scene_cats: dict[str, list[tuple[str, str]]], pilots: set) -> pd.DataFrame:
    """per_scene_cats: scene -> [(category, kind)]. Unique = exactly 1 in the scene and not a group."""
    inst, present, uniq, pil = Counter(), Counter(), Counter(), Counter()
    for sid, items in per_scene_cats.items():
        c = Counter(cat for cat, _ in items)
        groups = {cat for cat, k in items if k == "group"}
        inst.update(cat for cat, _ in items)
        present.update(c.keys())
        uniq.update(cat for cat, n in c.items() if n == 1 and cat not in groups)
        if sid in pilots:
            pil.update(cat for cat, _ in items)
    return pd.DataFrame([{"category": k, "instances": inst[k], "scenes_present": present[k], "scenes_unique": uniq[k],
                          "pilot_instances": pil[k]} for k in sorted(inst, key=lambda k: (-inst[k], k))])


def build(cfg: dict) -> dict:
    official = load_official(cfg)
    scenes, labels = raw_label_stats(cfg)
    pilots = set(sp_cfg(cfg)["pilot_scenes"])
    lab_inst, lab_scenes, lab_pilot = Counter(), Counter(), Counter()
    for sid, labs in labels.items():
        lab_inst.update(labs)
        lab_scenes.update(set(labs))
        if sid in pilots:
            lab_pilot.update(labs)
    rows = []
    for lab in sorted(lab_inst, key=lambda k: (-lab_inst[k], k)):
        m = map_label(lab, official)
        rows.append({"source_label": lab, **m, "instances_50": lab_inst[lab], "scenes_50": lab_scenes[lab], "pilot_instances": lab_pilot[lab]})
    mp = pd.DataFrame(rows)
    vdir = path_of(cfg, "vocab")
    mp[["source_label", "kind", "benchmark_class", "category_fine", "category_merged", "rule", "instances_50", "scenes_50",
        "pilot_instances"]].to_csv(vdir / "label_mapping_scannetpp.tsv", sep="\t", index=False)

    # category table: fine categories + merged parents
    objs = mp[mp.kind != "artifact"]
    src = defaultdict(list)
    for r in objs.itertuples():
        src[r.category_fine].append(f"{r.source_label}({r.instances_50})")
    parents = {c: p for p, cs in FAMILIES.items() for c in cs}
    all_cats = set(objs.category_fine) | set(objs.category_merged)
    allow = allowed_categories(all_cats, base_categories(official))
    cats = []
    for cat in sorted(all_cats, key=lambda c: (-int(objs[objs.category_merged == c].instances_50.sum() + objs[objs.category_fine == c].instances_50.sum()), c)):
        in_fine = cat in set(objs.category_fine)
        allowed = cat in allow
        cats.append({"category": cat, "allowed": str(allowed).lower(), "name_ok": str(allowed and cat in NAME_FINE).lower(),
                     "display_name": DISPLAY_NAME.get(cat, cat),
                     "source": allow.get(cat, ""),
                     "level": "fine+merged" if in_fine and cat not in parents else ("fine" if in_fine else "merged"),
                     "merged_into": parents.get(cat, ""), "instances_50_fine": int(objs[objs.category_fine == cat].instances_50.sum()),
                     "instances_50_merged": int(objs[objs.category_merged == cat].instances_50.sum()),
                     "source_labels": "; ".join(src.get(cat, [])),
                     "note": NAME_FINE.get(cat, "") if allowed else disallowed_reason(cat, official)})
    cdf = pd.DataFrame(cats)
    cdf.to_csv(vdir / "categories_scannetpp_draft.tsv", sep="\t", index=False)

    # frequencies (both levels)
    m_of = {r.source_label: r for r in mp.itertuples()}
    fine = {sid: [(m_of[l].category_fine, m_of[l].kind) for l in labs if m_of[l].kind != "artifact"] for sid, labs in labels.items()}
    merged = {sid: [(m_of[l].category_merged, m_of[l].kind) for l in labs if m_of[l].kind != "artifact"] for sid, labs in labels.items()}
    allowed = set(allow)
    ff, fm = frequency(fine, pilots), frequency(merged, pilots)
    ff.insert(1, "level", "fine")
    fm.insert(1, "level", "merged")
    freq = pd.concat([ff, fm], ignore_index=True)
    freq["allowed"] = freq.category.isin(allowed)
    freq.to_csv(vdir / "category_frequency_scannetpp.tsv", sep="\t", index=False)
    return {"num_scenes": len(scenes), "num_labels": len(mp), "num_instances": int(mp.instances_50.sum()),
            "num_artifact": int(mp[mp.kind == "artifact"].instances_50.sum()), "num_categories": len(cdf),
            "num_allowed": int((cdf.allowed == "true").sum())}


def load_mapping(cfg: dict):
    """For Step 3: read the review TSVs (manual edits to the TSVs take effect directly). Returns (raw label -> row, set of categories allowed in questions)."""
    vdir = path_of(cfg, "vocab")
    mp = pd.read_csv(vdir / "label_mapping_scannetpp.tsv", sep="\t", dtype=str, keep_default_na=False)
    cats = pd.read_csv(vdir / "categories_scannetpp_draft.tsv", sep="\t", dtype=str, keep_default_na=False)
    return {r.source_label: r._asdict() for r in mp.itertuples(index=False)}, set(cats[cats.allowed == "true"].category)


def load_display_names(cfg: dict) -> dict:
    """Category -> name used in questions (display_name column of categories_scannetpp_draft.tsv; entries equal to the category are omitted)."""
    cats = pd.read_csv(path_of(cfg, "vocab") / "categories_scannetpp_draft.tsv", sep="\t", dtype=str, keep_default_na=False)
    return {r.category: r.display_name for r in cats.itertuples() if r.display_name and r.display_name != r.category}


def load_name_ok(cfg: dict) -> set:
    """Categories named by their fine name in questions (name_ok column of categories_scannetpp_draft.tsv, i.e. NAME_FINE); all others use the parent name."""
    cats = pd.read_csv(path_of(cfg, "vocab") / "categories_scannetpp_draft.tsv", sep="\t", dtype=str, keep_default_na=False)
    return set(cats[cats.name_ok == "true"].category)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    args = ap.parse_args()
    print(build(load_config()))


if __name__ == "__main__":
    main()
