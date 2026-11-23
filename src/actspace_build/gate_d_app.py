"""GATE D verification page (from round 2 on): a single-page app.

All sampled question groups live in one index.html, one view per group (#group_id). This keeps browser local storage in a
single origin, so "Export all annotations" works when the local file is opened in Chrome / Firefox / Safari (with one HTML file
per group, Firefox / Safari treat each file as a separate origin and the list page cannot read answers saved by other pages).

Two phases per group:
1. Blind: only the initial view and the raw oracle views (upright, no overlays), questions and options; answer and tick flags.
2. After the blind answers are submitted they are locked and each question shows its ground truth plus review materials:
   (a) each view with boxes and names of the question's objects overlaid; (b) a top-down map (object centers and names;
   direction questions add the standing point, heading arrow and quadrants; T4 adds the start position and heading).
   Mask-method questions are marked explicitly as masked. The annotator judges the ground truth "correct / incorrect / unsure"
   per question and may add a note.
Review materials are separate image files that are loaded only after the blind answers are submitted.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np

from . import arkit, geometry as g, topdown
from .common import path_of
from .gate_d import DATASET, gt_text

MASKED_HINT = ("This is a masked question: the object in the question was deliberately excluded from the given views, so it "
               "cannot be seen in the images, and the ground truth is 'Cannot be determined'.")  # embedded in a double-quoted JS string


def dataset_io(cfg, dataset):
    """(frame image root, function returning the scene mesh vertices). ARKit keeps the original behaviour; ScanNet++ reads
    this dataset's directory and mesh."""
    if dataset == "scannetpp":
        from .scannetpp import sp_root
        from .scannetpp_step3 import load_mesh
        return sp_root(cfg), lambda vid: load_mesh(cfg, vid)[1]
    return (path_of(cfg, "arkit_3dod_root").resolve().parents[1],
            lambda vid: arkit.load_mesh_legacy(arkit.scene_files_3dod(path_of(cfg, "arkit_3dod_root"), vid)["mesh"]).vertices)


PAGE_IMAGE_WIDTH = {"scannetpp": 1280}  # ScanNet++ images are 1920x1440, scaled to 1280 wide on the page (as in the visibility spot-check images); ARKit is not scaled


def fit_width(img, dataset):
    w = PAGE_IMAGE_WIDTH.get(dataset)
    if not w or img.shape[1] <= w:
        return img
    return cv2.resize(img, (w, int(round(img.shape[0] * w / img.shape[1]))), interpolation=cv2.INTER_AREA)


def scene_structure(cfg, dataset, vid, sc, V):
    """Structure layer of the top-down map: vertices of objects labelled wall (subsampled), OBBs of doors and windows
    (including non-referable instances)."""
    import numpy as np
    from .common import work_dir
    pts = np.load(work_dir(cfg, dataset, vid) / "object_points.npz")
    rng = np.random.default_rng(0)
    wall, doors, windows = [], [], []
    for o in sc.objs:
        c = o.get("category_fine", o["category"])
        if c == "wall":
            idx = pts[f"obj_{o['obj_id']}"]
            wall.append(V[idx if len(idx) <= 4000 else rng.choice(idx, 4000, replace=False), :2])
        elif c in ("door", "window"):
            (doors if c == "door" else windows).append(o["obb"])
    return {"wall_xy": np.concatenate(wall) if wall else np.zeros((0, 2)), "doors": doors, "windows": windows}


def build_group_assets(cfg, grp, out_dir, sc, names, bg, part=None, dataset=DATASET, structure=None):
    """Write the group's raw view images and per-question review images; return the data needed by the page."""
    from .reports import _draw_tile
    frames = {f["frame_id"]: f for f in sc.frames}
    root = dataset_io(cfg, dataset)[0]
    nodes, tags = [grp["init_frame"]], {}
    for i, q in enumerate(grp["questions"], 1):
        pv = q.get("path_views") or []
        for j, n in enumerate(pv, 1):  # T4 path views from the start to the target
            tags.setdefault(n, []).append(f"Q{i} path {j}/{len(pv)}")
        for n in (q.get("oracle_nodes") or []) + (q.get("link_views") or []) + pv:  # oracle, linking and path views shown together
            if n not in nodes:
                nodes.append(n)
    masked = set((grp.get("view_graph_mask") or {}).get("masked_nodes", []))
    assert not masked & set(nodes), "masked nodes must not appear on the page"
    gdir = out_dir / grp["group_id"]
    (gdir / "img").mkdir(parents=True, exist_ok=True)
    (gdir / "review").mkdir(parents=True, exist_ok=True)
    views = []
    for k, n in enumerate(nodes):
        f = frames[n]
        img = g.rotate_image_cw(cv2.imread(str(root / f["rel_path"])), f["rotation_cw_quarters"])
        cv2.imwrite(str(gdir / "img" / f"{n}.jpg"), fit_width(img, dataset), [cv2.IMWRITE_JPEG_QUALITY, 85])
        label = "Initial view (start; T4 questions use this position and heading)" if k == 0 else f"View {k}"
        if n in tags:
            label += " (" + "; ".join(tags[n]) + ")"
        views.append({"file": f"{grp['group_id']}/img/{n}.jpg", "start": k == 0, "label": label})
    p0 = frames[grp["init_frame"]]
    qs = []

    def tile(i, n, title, name):
        fn = f"{grp['group_id']}/review/q{i}_{name}.jpg"
        cv2.imwrite(str(out_dir / fn), fit_width(_draw_tile(cfg, sc, n, ev, title, image_root=root), dataset), [cv2.IMWRITE_JPEG_QUALITY, 80])
        return fn
    for i, q in enumerate(grp["questions"], 1):
        ev = list(dict.fromkeys(q["evidence_objs"]))
        overlays = []
        # T4: path views first, in order (with the target object's box overlaid)
        pv = q.get("path_views") or []
        for j, n in enumerate(pv, 1):
            overlays.append(tile(i, n, f"path {j}/{len(pv)} ({n})", f"path{j}"))
        # (a) view overlays: views in which at least one evidence object of the question is visible or partially visible
        for k, n in enumerate(nodes):
            j = sc.node_index[n]
            if n not in pv and any(sc.st(o, j) != 0 for o in ev):
                overlays.append(tile(i, n, f"view {k} ({'start' if k == 0 else n})", f"view{k}"))
        # (b) top-down map
        s_ = q["slots"]
        kw = {}
        if q["template"] in ("T3", "T5"):
            kw = {"stand": s_["OBJ1"], "face": s_["OBJ2"], "target": s_.get("OBJ3")}
        elif q["template"] == "T4":
            kw = {"start": (np.asarray(p0["position"])[:2], p0["yaw_deg"]), "target": s_["OBJ1"]}
        shown = {o: sc.objs[sc.row[o]] for o in set(sc.referable) | set(ev)}
        td = topdown.render(shown, names, ev, bg, title=f"top-down view, question {i} (x right, y up; 1 m scale bar)", structure=structure, **kw)
        fn_td = f"{grp['group_id']}/review/q{i}_topdown.jpg"
        cv2.imwrite(str(out_dir / fn_td), td, [cv2.IMWRITE_JPEG_QUALITY, 85])
        qs.append({"qid": q["qid"], "text": q["text"], "numeric": q["answer_type"] == "numeric",
                   "options": q["options"], "masked": q.get("unanswerable_method") == "mask",
                   "overlays": overlays, "n_path": len(pv), "topdown": fn_td})
    return {"group_id": grp["group_id"], "scene_id": grp["scene_id"], "part": part, "views": views, "questions": qs}


def build(version, parts, out_dir: Path, dataset=DATASET, draw_structure=False):
    """parts: [(cfg, list of question groups, part name or None)], placed on the same page in order. Each part may use a
    different config (e.g. the sandbox work directory of an experimental branch). Writes out_dir/index.html and per-group
    images. Group ids must be unique across parts. dataset defaults to ARKit (original behaviour).
    With draw_structure=True the review top-down maps also draw wall points, doors and windows (scene_structure); the default
    False leaves ARKit unchanged."""
    from .step56_questions import Scene
    out_dir.mkdir(parents=True, exist_ok=True)
    data, gt, order = [], {}, []
    for cfg, groups, part in parts:
        by_scene = {}
        for grp in groups:
            by_scene.setdefault(grp["scene_id"], []).append(grp)
            order.append(grp["group_id"])
        for sid in sorted(by_scene):
            vid = sid.split("_", 1)[1]
            sc = Scene(cfg, dataset, vid)
            names = {o["obj_id"]: o.get("ref_name") or o["category"] for o in sc.objs}
            V = dataset_io(cfg, dataset)[1](vid)
            bg = topdown.mesh_background(V)
            st = scene_structure(cfg, dataset, vid, sc, V) if draw_structure else None
            for grp in by_scene[sid]:
                data.append(build_group_assets(cfg, grp, out_dir, sc, names, bg, part, dataset, st))
                gt.update({q["qid"]: gt_text(q) for q in grp["questions"]})
    assert len(set(order)) == len(order), "duplicate question group ids across parts"
    rank = {gid: k for k, gid in enumerate(order)}
    data.sort(key=lambda d: rank[d["group_id"]])
    gt_b64 = base64.b64encode(json.dumps(gt, ensure_ascii=False).encode()).decode()
    (out_dir / "index.html").write_text(HTML.replace("__VERSION__", html.escape(version))
                                        .replace("__DATA__", json.dumps({"version": version, "groups": data}, ensure_ascii=False))
                                        .replace("__GT__", gt_b64).replace("__HINT__", MASKED_HINT))
    return data


HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ActSpace GATE D verification · __VERSION__</title>
<style>
:root { --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6b66; --card:#ffffff; --line:#e3e2dd; --accent:#2c6ecb; --warn:#b4541f; --ok:#2f7d32; }
@media (prefers-color-scheme: dark) { :root { --bg:#1a1a19; --fg:#ecebe7; --muted:#a3a29d; --card:#242422; --line:#3a3936; --accent:#7aa7e8; --warn:#e39a6d; --ok:#7cc47f; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system, "PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif; }
main { max-width:1180px; margin:0 auto; padding:16px 16px 60px; }
h1 { font-size:20px; margin:6px 0 4px; } h2 { font-size:16px; margin:24px 0 10px; }
.meta, .hint { color:var(--muted); font-size:13px; }
.bar { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 0; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
.bar input { padding:6px 8px; border:1px solid var(--line); border-radius:6px; background:var(--card); color:var(--fg); }
button { padding:7px 14px; border-radius:6px; border:1px solid var(--accent); background:var(--accent); color:#fff; cursor:pointer; font-size:14px; }
button.secondary { background:transparent; color:var(--accent); }
table { border-collapse:collapse; width:100%; } td, th { border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; }
.status-review { color:var(--ok); } .status-blind { color:var(--warn); } .status-none { color:var(--muted); }
.views { display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:10px; }
figure { margin:0; background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
figure img { width:100%; display:block; cursor:zoom-in; }
figcaption { padding:6px 8px; font-size:13px; color:var(--muted); }
figure.start { border:2px solid var(--accent); } figure.start figcaption { color:var(--accent); font-weight:600; }
.q { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px 16px; margin:12px 0; }
.q .text { font-size:16px; margin:2px 0 10px; }
.q label { display:block; margin:4px 0; cursor:pointer; } label.inline { display:inline-block; margin:0 14px 0 6px; }
.flags { margin-top:10px; padding-top:8px; border-top:1px dashed var(--line); } .flags label { color:var(--warn); }
textarea { width:100%; min-height:44px; margin-top:6px; padding:6px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--fg); font:inherit; }
input[type=number] { width:120px; padding:5px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--fg); }
.review { margin-top:12px; padding:10px 12px; border-radius:6px; background:var(--bg); border:1px solid var(--accent); }
.review .gt { margin-bottom:8px; } .masked { color:var(--warn); font-weight:600; margin-bottom:8px; }
.review .imgs { display:grid; grid-template-columns:repeat(auto-fill, minmax(240px, 1fr)); gap:8px; margin:8px 0; }
.review .imgs img { width:100%; border:1px solid var(--line); border-radius:6px; cursor:zoom-in; }
.part { display:inline-block; padding:1px 8px; border-radius:10px; border:1px solid var(--warn); color:var(--warn); font-size:13px; }
.part.big { font-size:15px; font-weight:600; margin:8px 0 2px; }
#zoom { display:none; position:fixed; inset:0; background:rgba(0,0,0,.85); z-index:10; align-items:center; justify-content:center; }
#zoom img { max-width:96vw; max-height:92vh; background:#fff; }
</style></head>
<body><main>
<div class="bar"><strong>ActSpace GATE D human verification</strong>
<label>Annotator ID <input id="annotator" placeholder="e.g. ann01" size="10"></label>
<span id="progress" class="meta"></span>
<span id="actions"></span></div>
<div id="app"></div>
<div id="zoom"><img alt="Zoomed image"></div>
</main>
<script>
const DATA = __DATA__;
const GT = JSON.parse(new TextDecoder().decode(Uint8Array.from(atob("__GT__"), c => c.charCodeAt(0))));
const HINT = "__HINT__";
const $ = id => document.getElementById(id);
const ann = () => $('annotator').value.trim();
const key = gid => `actspace-gated2-${DATA.version}-${ann() || 'anon'}-${gid}`;
const load = gid => { try { return JSON.parse(localStorage.getItem(key(gid)) || 'null'); } catch (e) { return null; } };
const store = (gid, d) => { try { localStorage.setItem(key(gid), JSON.stringify(d)); localStorage.setItem('actspace-gated2-annotator', ann()); } catch (e) {} };
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function download(name, text, type) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], {type})); a.download = name;
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}
function status(gid) { const d = load(gid); return d ? d.phase : 'none'; }

// ---------------- List
const STATUS = {review: 'Both phases done', blind: 'Blind phase in progress', none: 'Not started'};
function renderList() {
  const hasPart = DATA.groups.some(g => g.part);
  const rows = DATA.groups.map((g, i) => `<tr><td>${i + 1}</td>${hasPart ? `<td><span class="part">${esc(g.part || '')}</span></td>` : ''}<td><a href="#${g.group_id}">${g.group_id}</a></td><td>${esc(g.scene_id)}</td><td>${g.questions.length}</td><td id="st-${g.group_id}"></td></tr>`).join('');
  $('actions').innerHTML = `<button onclick="exportAll()">Export all annotations</button>`;
  $('app').innerHTML = `<h1>Question groups · ${esc(DATA.version)}</h1>
    <p class="meta">${DATA.groups.length} question groups. Click an id to open a group; answers are saved in this browser automatically. When everything is done, click "Export all annotations" at the top right and return the single file you get.</p>
    <table id="grouptable"><tr><th>#</th>${hasPart ? '<th>Part</th>' : ''}<th>Question group</th><th>Scene</th><th>Questions</th><th>Status</th></tr>${rows}</table>`;
  refreshList();
}
// Update only the status cells instead of rebuilding the list, so that when a link is clicked right after entering the
// annotator ID, the change event fired on blur does not replace the link being clicked
function refreshList() {
  for (const g of DATA.groups) { const s = status(g.group_id), td = $(`st-${g.group_id}`); td.className = `status-${s}`; td.textContent = STATUS[s]; }
  const done = DATA.groups.filter(g => status(g.group_id) === 'review').length;
  $('progress').textContent = `${done} / ${DATA.groups.length} groups done`;
}
function exportAll() {
  if (!ann()) { alert('Please enter your annotator ID first'); return; }
  const recs = DATA.groups.map(g => load(g.group_id)).filter(Boolean);
  if (!recs.length) { alert('There are no annotations yet'); return; }
  const unfinished = recs.filter(r => r.phase !== 'review').length + (DATA.groups.length - recs.length);
  if (unfinished && !confirm(`${unfinished} groups have not completed both phases. Export anyway?`)) return;
  download(`annotations_${ann()}_${DATA.version}.jsonl`, recs.map(r => JSON.stringify(r)).join('\n') + '\n', 'application/x-ndjson');
}

// ---------------- Single group
function groupById(gid) { return DATA.groups.find(g => g.group_id === gid); }
function renderGroup(g) {
  const i = DATA.groups.indexOf(g);
  const figs = g.views.map(v => `<figure class="${v.start ? 'start' : ''}"><img src="${v.file}" alt=""><figcaption>${esc(v.label)}</figcaption></figure>`).join('');
  const qs = g.questions.map((q, k) => {
    const opts = q.numeric
      ? `<label><input type="radio" name="${q.qid}-ans" value="numeric"> Numeric answer: <input type="number" step="0.1" min="0" id="${q.qid}-num" placeholder="meters" oninput="this.closest('label').querySelector('input[type=radio]').checked = true"> m</label>`
      : q.options.map(o => `<label><input type="radio" name="${q.qid}-ans" value="${o.split('. ')[0]}"> ${esc(o)}</label>`).join('');
    return `<div class="q"><div class="meta">Question ${k + 1}</div><div class="text">${esc(q.text)}</div><div class="blind">${opts}
      <label><input type="radio" name="${q.qid}-ans" value="cannot_determine"> Cannot be determined from these views</label>
      <div class="flags"><label><input type="checkbox" id="${q.qid}-amb"> The question is ambiguous</label>
      <label><input type="checkbox" id="${q.qid}-wrong"> Object reference or label is wrong (label wrong)</label>
      <label><input type="checkbox" id="${q.qid}-unann"> The scene contains an unannotated object of the same category (ambiguous reference)</label>
      <textarea id="${q.qid}-note" placeholder="Note (optional)"></textarea></div></div>
      <div class="review" id="${q.qid}-review" hidden></div></div>`;
  }).join('');
  const prev = i > 0 ? `<a href="#${DATA.groups[i - 1].group_id}"><button class="secondary" type="button">Previous group</button></a>` : '';
  const next = i < DATA.groups.length - 1 ? `<a href="#${DATA.groups[i + 1].group_id}"><button class="secondary" type="button">Next group</button></a>` : '';
  $('actions').innerHTML = `<button id="submit-blind" onclick="submitBlind('${g.group_id}')">Submit blind answers (cannot be changed afterwards)</button>
    <button onclick="downloadGroup('${g.group_id}')">Download this group's annotations</button> ${prev} ${next}
    <a href="#"><button class="secondary" type="button">Back to list</button></a>`;
  $('app').innerHTML = `${g.part ? `<div class="part big">${esc(g.part)}</div>` : ''}<h1>Question group ${g.group_id} (${i + 1} / ${DATA.groups.length})</h1>
    <div class="meta">Scene ${esc(g.scene_id)} · ${g.questions.length} questions · ${g.views.length} views · version ${esc(DATA.version)}</div>
    <p class="hint">Answer only from the images below. You start from the "Initial view" and can see all listed views. If the views are not enough to answer, choose "Cannot be determined".
    Distance means the straight-line distance between object centers (meters); <strong>estimates are fine for distance questions, just enter one number</strong>; <strong>if you cannot tell, choose "Cannot be determined" instead of entering 0</strong>; <strong>for direction questions, imagine standing at the position and facing the direction described in the question</strong>, then judge front / back / left / right. Click an image to enlarge it.</p>
    <p class="hint" id="phase-hint">Phase 1: blind answers. When all questions are answered, click "Submit blind answers"; the ground truth and review materials of each question are then shown (phase 2), and the blind answers can no longer be changed.</p>
    <h2>Views (${g.views.length})</h2><div class="views">${figs}</div><h2>Questions</h2>${qs}`;
  restore(g); window.scrollTo(0, 0);
}
function blindAnswers(g) {
  const out = {};
  for (const q of g.questions) {
    const sel = document.querySelector(`input[name="${q.qid}-ans"]:checked`);
    const num = $(`${q.qid}-num`);
    let a = sel ? sel.value : null;
    if (a === 'numeric' || (!sel && num && num.value !== '')) a = num && num.value !== '' ? parseFloat(num.value) : null;
    out[q.qid] = {answer: a, ambiguous: $(`${q.qid}-amb`).checked, label_wrong: $(`${q.qid}-wrong`).checked,
                  unannotated_same_category_object: $(`${q.qid}-unann`).checked, note: $(`${q.qid}-note`).value};
  }
  return out;
}
function reviewAnswers(g) {
  const out = {};
  for (const q of g.questions) {
    const sel = document.querySelector(`input[name="${q.qid}-verdict"]:checked`);
    out[q.qid] = {verdict: sel ? sel.value : null, note: $(`${q.qid}-rnote`) ? $(`${q.qid}-rnote`).value : ''};
  }
  return out;
}
function collect(g) {
  const old = load(g.group_id) || {};
  const phase = old.phase === 'review' ? 'review' : 'blind';
  return {group_id: g.group_id, scene_id: g.scene_id, version: DATA.version, annotator: ann(), saved_at: new Date().toISOString(),
          phase, blind_submitted_at: old.blind_submitted_at || null,
          blind: phase === 'review' ? old.blind : blindAnswers(g), review: phase === 'review' ? reviewAnswers(g) : null};
}
function save() {
  const gid = location.hash.slice(1); const g = groupById(gid);
  if (!g || !ann()) return;
  const d = collect(g); store(gid, d);
  const n = g.questions.length;
  $('progress').textContent = d.phase === 'blind'
    ? `Phase 1 · blind: ${Object.values(d.blind).filter(a => a.answer !== null).length} / ${n} answered`
    : `Phase 2 · review: ${Object.values(d.review).filter(a => a.verdict !== null).length} / ${n} reviewed`;
}
function showReview(g) {
  document.querySelectorAll('.blind input, .blind textarea').forEach(el => el.disabled = true);
  $('submit-blind').style.display = 'none';
  for (const q of g.questions) {
    const box = $(`${q.qid}-review`); box.hidden = false;
    const imgs = q.overlays.map(f => `<img src="${f}" alt="">`).join('') + `<img src="${q.topdown}" alt="">`;
    box.innerHTML = (q.masked ? `<div class="masked">${esc(HINT)}</div>` : '') +
      `<div class="gt">Ground truth: <strong>${esc(GT[q.qid])}</strong></div>
       <div class="meta">Review materials: ${q.n_path ? `the first ${q.n_path} images are path views from the start to the target (in order, path 1/${q.n_path}…), followed by ` : ''}views with boxes and names of this question's objects overlaid (green = visible in that view, orange = partially visible, red = not visible); the last image is the top-down map.</div>
       <div class="imgs">${imgs}</div>
       <div>Is the ground truth correct? <label class="inline"><input type="radio" name="${q.qid}-verdict" value="correct"> Correct</label>
       <label class="inline"><input type="radio" name="${q.qid}-verdict" value="incorrect"> Incorrect</label>
       <label class="inline"><input type="radio" name="${q.qid}-verdict" value="unsure"> Unsure</label></div>
       <textarea id="${q.qid}-rnote" placeholder="Review note (optional; please give the reason if you choose Incorrect)"></textarea>`;
  }
  $('phase-hint').textContent = 'Phase 2: the blind answers are submitted and can no longer be changed. Using the review materials, judge for each question whether the ground truth is correct; when done, go back to the list or click "Download this group\'s annotations".';
}
function restore(g) {
  const d = load(g.group_id);
  if (d) {
    for (const [qid, a] of Object.entries(d.blind || {})) {
      if (a.answer !== null) {
        if (typeof a.answer === 'number') { document.querySelector(`input[name="${qid}-ans"][value="numeric"]`).checked = true; $(`${qid}-num`).value = a.answer; }
        else { const r = document.querySelector(`input[name="${qid}-ans"][value="${a.answer}"]`); if (r) r.checked = true; }
      }
      $(`${qid}-amb`).checked = a.ambiguous; $(`${qid}-wrong`).checked = a.label_wrong;
      $(`${qid}-unann`).checked = a.unannotated_same_category_object; $(`${qid}-note`).value = a.note || '';
    }
    if (d.phase === 'review') {
      showReview(g);
      for (const [qid, r] of Object.entries(d.review || {})) {
        if (r.verdict) { const el = document.querySelector(`input[name="${qid}-verdict"][value="${r.verdict}"]`); if (el) el.checked = true; }
        $(`${qid}-rnote`).value = r.note || '';
      }
    }
  }
  save();
}
function submitBlind(gid) {
  const g = groupById(gid);
  if (!ann()) { alert('Please enter your annotator ID first'); return; }
  const b = blindAnswers(g);
  const missing = g.questions.filter(q => b[q.qid].answer === null).length;
  if (missing) { alert(`${missing} questions are not answered yet (choose "Cannot be determined" if you cannot tell)`); return; }
  if (!confirm('After submission the blind answers can no longer be changed, and the ground truth and review materials will be shown. Submit?')) return;
  store(gid, {group_id: gid, scene_id: g.scene_id, version: DATA.version, annotator: ann(), saved_at: new Date().toISOString(),
              phase: 'review', blind_submitted_at: new Date().toISOString(), blind: b, review: null});
  showReview(g); save();
}
function downloadGroup(gid) {
  const d = load(gid);
  if (!ann()) { alert('Please enter your annotator ID first'); return; }
  if (!d || d.phase !== 'review') { alert('Please complete and submit the blind answers (phase 1) and review the ground truth (phase 2) before downloading'); return; }
  download(`${ann()}__${gid}.json`, JSON.stringify(d, null, 1), 'application/json');
}
function route() {
  const g = groupById(location.hash.slice(1));
  if (g) renderGroup(g); else renderList();
}
document.addEventListener('input', e => { if (e.target.id !== 'annotator') save(); });
document.addEventListener('change', e => {
  if (e.target.id !== 'annotator') { save(); return; }
  try { localStorage.setItem('actspace-gated2-annotator', ann()); } catch (x) {}
  if ($('grouptable')) refreshList(); else save();  // ID changed on a group page: the current answers are stored under the new ID
});
document.addEventListener('click', e => {
  if (e.target.matches('figure img, .review .imgs img')) { const z = $('zoom'); z.querySelector('img').src = e.target.src; z.style.display = 'flex'; }
  else if (e.target.closest('#zoom')) $('zoom').style.display = 'none';
});
window.addEventListener('hashchange', route);
window.addEventListener('DOMContentLoaded', () => {
  try { const a = localStorage.getItem('actspace-gated2-annotator'); if (a) $('annotator').value = a; } catch (e) {}
  route();
});
</script>
</body></html>
"""
