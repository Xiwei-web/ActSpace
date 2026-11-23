"""Paper tables (report_paper): marking within groups, difference format, comparison with hand-entered numbers."""
from actspace_eval import report_paper as P


def row(group, model, scope="Merged", **kw):
    r = {"group": group, "model": model, "scope": scope, "condition": ""}
    r.update({c: None for c, _ in P.MAIN_COLS + P.APPX_COLS})
    r.update(kw)
    return r


def test_mark_within_group_and_lower_better():
    rows = [row("Proprietary API", "a", active=0.70, d_o=0.10), row("Proprietary API", "b", active=0.60, d_o=0.20), row("Proprietary API", "c", active=0.50, d_o=0.30),
            row("Open-source", "x", active=0.40, d_o=0.05), row("Open-source", "y", active=0.30, d_o=0.40)]
    P.mark(rows, ["active", "d_o"])
    by = {r["model"]: r for r in rows}
    assert "active" in by["a"]["best"] and "active" in by["b"]["second"]
    assert "active" in by["x"]["best"] and "active" in by["y"]["second"]     # the open-source group is ranked separately
    assert "d_o" in by["a"]["best"] and "d_o" in by["x"]["best"]             # lower ΔO is better
    assert "d_o" in by["c"]["second"] or "d_o" in by["b"]["second"]
    assert "d_o" in by["b"]["second"]


def test_mark_ties_and_missing():
    rows = [row("Open-source", "x", active=0.5), row("Open-source", "y", active=0.5), row("Open-source", "z", active=None)]
    P.mark(rows, ["active"])
    assert all("active" in r["best"] for r in rows[:2]) and not rows[2]["best"]


def test_num_and_cell():
    assert P.num(0.0687) == "6.9" and P.num(-0.063, "d_rw") == "−6.3" and P.num(0.069, "d_rw") == "+6.9" and P.num(None) == P.DASH
    r = row("Open-source", "x", active=0.5)
    r["best"], r["second"] = {"active"}, set()
    assert P.cell(r, "active") == "**50.0**"


def test_compare_reports_mismatches():
    rows = [row("Open-source", "x", active=0.5, d_rw=-0.063, T1=0.294)]
    ref = [("Merged", "x", "active", "50.0"), ("Merged", "x", "d_rw", "-6.3"), ("Merged", "x", "T1", "29.5"), ("Merged", "y", "active", "1.0"), ("Merged", "x", "T2", "–")]
    bad, n = P.compare(rows, ref, "paper")
    assert n == 4
    assert [(b[2], b[3]) for b in bad] == [("x", "T1"), ("y", "active")]


def test_delta_is_difference_of_displayed_values():
    """ΔRW / ΔO = difference of the two displayed (rounded) values; the unrounded difference is kept separately."""
    d, raw = P.display_diff(0.50049, 0.43751)          # displayed 50.0 and 43.8 -> 6.2 (unrounded 6.298)
    assert P.num(d, "d_rw") == "+6.2" and abs(raw - 0.06298) < 1e-12
    assert P.display_diff(None, 0.4) == (None, None)
    for a in (0.12345, 0.5, 0.33333, 0.66667, 0.0049):
        for b in (0.1, 0.25555, 0.9999, 0.00051):
            d, _ = P.display_diff(a, b)
            assert float(P.num(d, "d_rw").replace("−", "-")) == float(P.num(a)) - float(P.num(b)) or \
                abs(float(P.num(d, "d_rw").replace("−", "-")) - (float(P.num(a)) - float(P.num(b)))) < 1e-9


def test_tiny_subset_rows_are_flagged_and_never_marked():
    """A tiny-subset model (astra) is shown with a dagger, never marked, even when its value is the highest."""
    rows = [row("Proprietary API", "a", active=0.60), row("Proprietary API", "b", active=0.50),
            {**row("Proprietary API", "astra", active=0.90), "subset": "tiny"}]
    rows[0]["subset"] = rows[1]["subset"] = "full"
    P.mark(rows, ["active"])
    by = {r["model"]: r for r in rows}
    assert "active" in by["a"]["best"] and "active" in by["b"]["second"] and not by["astra"]["best"] and not by["astra"]["second"]
    assert P.label(by["astra"]) == "astra †" and P.label(by["a"]) == "a"
    assert ("astra", "tiny_core") in P.TINY_ONLY and "32-group tiny subset only" in P.TINY_NOTE


def test_latex_table_has_all_rows_marks_and_footnote():
    rows = [{**row("Proprietary API", "a", scope=s, active=0.6, d_rw=0.051), "subset": "full"} for s in ("Merged", "ARKitScenes", "ScanNet++")]
    rows += [{**row("Proprietary API", "astra", scope=s, active=0.9), "subset": "tiny"} for s in ("Merged", "ARKitScenes", "ScanNet++")]
    rows += [{**row("Open-source", "InternVL3_5-8B", scope=s, active=0.3, d_rw=-0.063), "subset": "full"} for s in ("Merged", "ARKitScenes", "ScanNet++")]
    P.mark(rows, ["active"])
    tex = P.latex_tables(rows)
    assert tex.count("\\begin{tabular}") == 3 == tex.count("\\end{tabular}") and tex.count("\\toprule") == 3
    assert "astra$^\\dagger$" in tex and "32-group tiny subset only" in tex and "InternVL3\\_5-8B" in tex and "random\\_walk\\_40" in tex
    assert "\\textbf{60.0}" in tex and "\\textbf{90.0}" not in tex and "$-$6.3" in tex and "$+$5.1" in tex
    body = "\n".join(l for l in tex.splitlines() if not l.startswith("%")).replace("\\_", "").replace("$\\Delta_{", "")
    assert "_" not in body                                               # every underscore outside comments is escaped
    for line in tex.splitlines():
        if line.endswith("\\\\") and not line.startswith("\\multicolumn"):
            assert line.count("&") == len(P.MAIN_COLS)                     # every row has all columns
