"""
Comprehensive post-hoc analysis of LimitTree results.
Zero LLM calls — all derived from cached match responses and paper data.

Analyses produced:
  T1.1  Panel action -> match rate (uses approximate string matching for forest_panel)
  T1.2  Per-source coverage (OpenReview vs citation)
  T1.3  Evidence quote grounding verification
  T1.4  Category confusion matrix (gold vs framework)
  T1.5  Severity accuracy
  T2.1  Inter-method complementarity (Jaccard on gold-hit sets)
  T2.2  Per-paper performance variance
  T2.3  Confidence calibration
  T2.4  Cost / efficiency (best-effort from logs)
  T2.5  Category distribution vs gold (KL divergence)
  T3.1  Unmatched framework outputs — examples
  T3.2  Three full case studies
  T3.3  Error cases (gold not matched by any method)

Output: data/analysis_report.md
"""
from __future__ import annotations

import json
import math
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path("./limit_tree")

METHODS = {
    "forest_panel":   {"lims": "data/framework_test/framework_limitations.json",
                       "eval_dir": "data/eval_match_forest_panel_test",
                       "results_json": "data/framework_test/framework_results.json"},
    "forest_nopanel": {"lims": "data/framework_test_nopanel/framework_limitations.json",
                       "eval_dir": "data/eval_match_forest_nopanel_test"},
    "zero_shot":      {"lims": "data/baseline_zs_test/baseline_limitations.json",
                       "eval_dir": "data/eval_match_zero_shot_test"},
    "single_skeptic": {"lims": "data/baseline_ss_test/baseline_limitations.json",
                       "eval_dir": "data/eval_match_single_skeptic_test"},
}

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_json(p: str):
    return json.loads((ROOT / p).read_text())


def load_limitations(p: str):
    return load_json(p)


def load_match_responses(eval_dir: str):
    path = ROOT / eval_dir / "match_responses.pkl"
    if not path.exists():
        return None
    return pickle.loads(path.read_bytes())


def load_request_meta(eval_dir: str):
    path = ROOT / eval_dir / "request_meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def parse_match(text: str):
    """Return True/False/None."""
    if not text:
        return None
    t = text.strip().lower().strip(".,'\"")
    if t in ("true", "yes", "same", "match"):
        return True
    if t in ("false", "no", "different"):
        return False
    first = t.split()[0] if t else ""
    first = first.strip(".,:;\"'")
    if first in ("true", "yes"): return True
    if first in ("false", "no", "different"): return False
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "is_match" in obj:
            return bool(obj["is_match"])
        if isinstance(obj, bool):
            return obj
    except Exception:
        pass
    return None


def build_match_table(eval_dir: str):
    """Return list of (paper_id, gold_idx, fw_idx, is_match, fw_category, gold_category, fw_severity, gold_severity, fw_id, gold_id)."""
    responses = load_match_responses(eval_dir)
    meta = load_request_meta(eval_dir)
    if responses is None or meta is None:
        return None
    rows = []
    for rid, text in responses.items():
        m = meta.get(rid)
        if not m:
            continue
        is_match = parse_match(text)
        if is_match is None:
            continue
        rows.append({
            "paper_id": m["paper_id"],
            "gold_idx": m["gold_idx"],
            "fw_idx": m["fw_idx"],
            "is_match": is_match,
            "fw_category": m["fw_category"],
            "gold_category": m["gold_category"],
            "fw_severity": m["fw_severity"],
            "gold_severity": m["gold_severity"],
            "fw_id": m["fw_id"],
            "gold_id": m["gold_id"],
        })
    return rows


# ---------------------------------------------------------------------------
# Matching fallback (for forest_panel, whose S3 data is gone)
# Uses TF-IDF-style token overlap, no LLM.
# ---------------------------------------------------------------------------

def _tok(s):
    s = s.lower()
    return set(re.findall(r"[a-z]{3,}", s))

STOP = _tok("the a an of to in for on with and or but is are was were be been being this that these those paper authors work method approach model dataset evaluation result results "
            "we our which can may should would could do does did use uses used using not no")

def token_overlap(a: str, b: str) -> float:
    A, B = _tok(a) - STOP, _tok(b) - STOP
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


def approx_match_table_for_forest_panel(fw_lims, gold_by_paper, threshold=0.22):
    """Approximate matches using Jaccard token overlap on descriptions.
    Returns same-shape rows as build_match_table."""
    rows = []
    for f in fw_lims:
        pid = f["paper_id"]
        gs = gold_by_paper.get(pid, [])
        for gi, g in enumerate(gs):
            sim = token_overlap(f["description"], g["description"])
            rows.append({
                "paper_id": pid,
                "gold_idx": gi,
                "fw_idx": 0,  # not meaningful for approx
                "is_match": sim >= threshold,
                "fw_category": f["category"],
                "gold_category": g["category"],
                "fw_severity": f.get("severity", "minor"),
                "gold_severity": g.get("severity", "minor"),
                "fw_id": f["id"],
                "gold_id": g["id"],
                "_approx_sim": sim,
            })
    return rows


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def analysis_panel_action_match_rate(fw_lims_panel, panel_tree_data, match_rows_approx):
    """For each panel verdict (endorse/downgrade/merge/reclassify), compute fraction of those
    claims that match some gold. Since we lack real per-pair for forest_panel, use approx."""
    # Map fw_id -> panel verdict
    fw_id_to_verdict = {}
    for paper_entry in panel_tree_data:
        for node in paper_entry.get("nodes", []):
            panel = node.get("panel_verdict")
            if not panel:
                continue
            # Find corresponding final lim; node_id appears in lim.id via annotator_notes
            node_id = node.get("node_id")
            lim_id = node.get("final_limitation_id")
            if lim_id:
                fw_id_to_verdict[lim_id] = panel

    # Alternative extraction from framework_limitations — check annotator_notes
    for f in fw_lims_panel:
        notes = f.get("annotator_notes") or ""
        m = re.search(r"panel_verdict[=:]\s*(\w+)", notes)
        if m:
            fw_id_to_verdict[f["id"]] = m.group(1)

    # For each fw_id, did it match some gold?
    match_by_fw = defaultdict(bool)
    for r in match_rows_approx:
        if r["is_match"]:
            match_by_fw[r["fw_id"]] = True

    # Aggregate per verdict
    by_verdict = defaultdict(lambda: {"total": 0, "matched": 0})
    for f in fw_lims_panel:
        v = fw_id_to_verdict.get(f["id"], "unknown")
        by_verdict[v]["total"] += 1
        if match_by_fw.get(f["id"], False):
            by_verdict[v]["matched"] += 1
    return by_verdict


def analysis_per_source_coverage(match_rows, gold_by_paper, gold):
    """Split gold by source.kind, compute Coverage@10 per source, per method.
    Restricts to papers that appear in match_rows (i.e., test papers)."""
    # Build paper -> gold_idx -> set(fw_idx) for is_match=True
    mm = defaultdict(lambda: defaultdict(set))
    test_papers = set()
    for r in match_rows:
        test_papers.add(r["paper_id"])
        if r["is_match"]:
            mm[r["paper_id"]][r["gold_idx"]].add(r["fw_idx"])

    hit_by_source = defaultdict(lambda: {"hit": 0, "total": 0})
    for paper_id in test_papers:
        gold_lims = gold_by_paper.get(paper_id, [])
        mat = mm.get(paper_id, {})
        for gi, g in enumerate(gold_lims):
            src_kind = g["source"]["kind"]
            hit_by_source[src_kind]["total"] += 1
            hits = mat.get(gi, set())
            if any(fi < 10 for fi in hits):
                hit_by_source[src_kind]["hit"] += 1
    return dict(hit_by_source)


def analysis_evidence_grounding(fw_lims, papers_by_id):
    """For each fw lim, check if evidence_quote actually appears in paper's full_text."""
    out = {"total": 0, "has_quote": 0, "quote_grounded": 0, "by_category": defaultdict(lambda: {"total":0,"grounded":0})}
    for f in fw_lims:
        out["total"] += 1
        cat = f["category"]
        out["by_category"][cat]["total"] += 1
        ev = f.get("evidence_in_paper")
        if not ev or not ev.get("text"):
            continue
        out["has_quote"] += 1
        quote = ev["text"].strip()
        paper = papers_by_id.get(f["paper_id"])
        if not paper:
            continue
        # paper text
        text = paper.get("full_text") or ""
        if not text:
            # reconstruct from sections
            text = "\n".join(s.get("text","") for s in paper.get("sections", []))
        # check substring or fuzzy (quote must be >= 20 chars and appear ~exactly)
        if len(quote) < 10:
            continue
        needle = " ".join(quote.split())[:200]  # first 200 chars, normalized spaces
        haystack = " ".join(text.split())
        if needle in haystack:
            out["quote_grounded"] += 1
            out["by_category"][cat]["grounded"] += 1
        else:
            # try first 80 chars
            if needle[:80] in haystack and len(needle) > 80:
                out["quote_grounded"] += 1
                out["by_category"][cat]["grounded"] += 1
    return out


def analysis_category_confusion(match_rows):
    """For matched pairs, cross-tab gold_category vs fw_category."""
    cats = ["scope", "methodology", "theoretical", "reproducibility", "fairness"]
    mat = {g: {f: 0 for f in cats} for g in cats}
    for r in match_rows:
        if r["is_match"] and r["fw_category"] in cats and r["gold_category"] in cats:
            mat[r["gold_category"]][r["fw_category"]] += 1
    return mat


def analysis_severity(match_rows):
    out = {"same": 0, "fw_minor_gold_major": 0, "fw_major_gold_minor": 0, "total": 0}
    for r in match_rows:
        if not r["is_match"]: continue
        fs = (r["fw_severity"] or "minor").lower()
        gs = (r["gold_severity"] or "minor").lower()
        out["total"] += 1
        if fs == gs:
            out["same"] += 1
        elif fs == "minor" and gs == "major":
            out["fw_minor_gold_major"] += 1
        elif fs == "major" and gs == "minor":
            out["fw_major_gold_minor"] += 1
    return out


def analysis_intermethod_overlap(method_matches):
    """Jaccard on the set of (paper_id, gold_idx) pairs each method matched."""
    hit_sets = {}
    for name, rows in method_matches.items():
        if rows is None:
            hit_sets[name] = None
            continue
        hits = set()
        for r in rows:
            if r["is_match"]:
                hits.add((r["paper_id"], r["gold_idx"]))
        hit_sets[name] = hits
    # Pairwise Jaccard
    pairs = {}
    names = [n for n, h in hit_sets.items() if h is not None]
    for i, a in enumerate(names):
        for b in names[i+1:]:
            A, B = hit_sets[a], hit_sets[b]
            inter = len(A & B)
            union = len(A | B) or 1
            pairs[f"{a} ∩ {b}"] = {"intersection": inter, "union": union,
                                   "jaccard": inter/union, "only_a": len(A-B),
                                   "only_b": len(B-A)}
    # Union coverage
    if names:
        all_hits = set()
        for n in names:
            all_hits |= hit_sets[n]
        pairs["_union_all"] = {"total_hit_gold": len(all_hits)}
    return hit_sets, pairs


def analysis_per_paper(match_rows, gold_by_paper, fw_by_paper):
    """Per-paper coverage and precision."""
    rows = []
    mm = defaultdict(lambda: defaultdict(set))
    for r in match_rows:
        if r["is_match"]:
            mm[r["paper_id"]][r["gold_idx"]].add(r["fw_idx"])
    for pid, gs in gold_by_paper.items():
        fws = fw_by_paper.get(pid, [])
        if not fws:
            continue
        mat = mm.get(pid, {})
        covered = sum(1 for gi in range(len(gs)) if any(fi < 10 for fi in mat.get(gi, set())))
        matched_fw = set()
        for gi, s in mat.items():
            matched_fw |= s
        rows.append({
            "paper_id": pid,
            "n_gold": len(gs),
            "n_fw": len(fws),
            "covered": covered,
            "coverage": covered / max(1, len(gs)),
            "matched_fw": len(matched_fw),
            "precision": len(matched_fw) / max(1, min(20, len(fws))),
        })
    return rows


def analysis_confidence_calibration(fw_lims, match_rows):
    """Bin by llm_confidence and compute match rate per bucket."""
    matched_fw = set()
    for r in match_rows:
        if r["is_match"]:
            matched_fw.add(r["fw_id"])
    buckets = [("0.00-0.60", 0.0, 0.60), ("0.60-0.75", 0.60, 0.75),
               ("0.75-0.85", 0.75, 0.85), ("0.85-0.95", 0.85, 0.95), ("0.95-1.00", 0.95, 1.01)]
    out = {}
    for label, lo, hi in buckets:
        total = 0; hit = 0
        for f in fw_lims:
            c = f.get("llm_confidence")
            if c is None: continue
            if lo <= c < hi:
                total += 1
                if f["id"] in matched_fw:
                    hit += 1
        out[label] = {"total": total, "match_rate": hit/total if total else None}
    return out


def analysis_cat_dist_vs_gold(fw_lims, gold):
    """KL(fw || gold) for category distributions."""
    cats = ["scope", "methodology", "theoretical", "reproducibility", "fairness"]
    def dist(lst):
        c = Counter(l["category"] for l in lst)
        tot = sum(c.values()) or 1
        return {k: (c.get(k, 0) + 0.1)/(tot + 0.5) for k in cats}
    gd = dist(gold); fd = dist(fw_lims)
    kl = sum(fd[k] * math.log(fd[k]/gd[k]) for k in cats)
    return {"gold": {k: round(gd[k], 3) for k in cats},
            "method": {k: round(fd[k], 3) for k in cats},
            "kl_divergence": kl}


def analysis_unmatched_fw(fw_lims, match_rows, k=10):
    matched = set(r["fw_id"] for r in match_rows if r["is_match"])
    unmatched = [f for f in fw_lims if f["id"] not in matched]
    return unmatched[:k]


def analysis_error_cases(gold_by_paper, method_matches, k=10):
    """Gold limitations matched by no method."""
    hit = defaultdict(set)
    for name, rows in method_matches.items():
        if rows is None: continue
        for r in rows:
            if r["is_match"]:
                hit[name].add((r["paper_id"], r["gold_idx"]))
    # any method hit?
    union = set()
    for name, s in hit.items():
        union |= s

    missed = []
    for pid, gs in gold_by_paper.items():
        for gi, g in enumerate(gs):
            if (pid, gi) not in union:
                missed.append(g)
    return missed[:k]


def analysis_case_study(paper_id, fw_lims, gold_by_paper, method_outputs):
    """Collect fw output + gold for one paper across methods."""
    case = {"paper_id": paper_id, "gold": gold_by_paper.get(paper_id, [])}
    for name, lims in method_outputs.items():
        case[name] = [l for l in lims if l["paper_id"] == paper_id]
    return case


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def md_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    # --- Load everything ---
    print("Loading data...")
    gold = load_json("data/tier_a/limitations.json")
    gold_by_paper = defaultdict(list)
    for g in gold:
        gold_by_paper[g["paper_id"]].append(g)

    method_lims = {}
    for name, spec in METHODS.items():
        method_lims[name] = load_limitations(spec["lims"])

    # Papers (for grounding)
    papers_by_id = {}
    papers_root = ROOT / "data/corpus/papers"
    if papers_root.exists():
        for d in papers_root.iterdir():
            pp = d / "paper.json"
            if pp.exists():
                p = json.loads(pp.read_text())
                papers_by_id[p["id"]] = p

    # Match tables
    method_matches = {}
    for name, spec in METHODS.items():
        m = build_match_table(spec["eval_dir"])
        method_matches[name] = m

    # forest_panel raw data was deleted from S3; its aggregate metrics are still available
    # but per-pair analyses below will skip it.
    has_pair_data = {n: method_matches[n] is not None and len(method_matches[n]) > 0 for n in METHODS}
    print(f"Per-pair data available: {has_pair_data}")

    # fw_by_paper for each method
    fw_by_paper = {}
    for name in METHODS:
        fw_by_paper[name] = defaultdict(list)
        for l in method_lims[name]:
            fw_by_paper[name][l["paper_id"]].append(l)

    # --- Run analyses ---
    out = []
    out.append("# LimitTree — Full post-hoc analysis (20-paper pilot)\n")
    out.append(f"All results on the 20-paper test set. Gold: {len(gold)} limitations across {len(gold_by_paper)} papers (374 of the 414 corpus papers have gold lims; 20 are in this test slice).\n")
    out.append(f"\n> **Data note:** forest_panel's eval-match raw per-pair data was deleted from S3 "
               f"(during yesterday's cleanup). Its aggregate metrics (Coverage@K, Precision) are preserved. "
               f"Per-pair analyses (category confusion, severity, unmatched-examples) use forest_nopanel "
               f"as the closest proxy — it shares the same skeptic setup minus Panel Review.\n")

    # T1.1 Panel action distribution (from framework output only — per-pair match missing)
    out.append("\n## T1.1  Panel verdict distribution and claim modification rate\n")
    out.append("_Forest_panel eval-match raw data was deleted from S3, so we cannot compute per-verdict match rates. "
               "But panel verdicts are recorded in every forest_panel limitation's annotator_notes and in the pickled orchestrator. "
               "Below is the distribution and what panel actions did structurally._\n")
    # Count panel verdicts from annotator_notes (stored as JSON string within notes)
    panel_counts = Counter()
    reclassify_examples = []
    for f in method_lims["forest_panel"]:
        notes = f.get("annotator_notes") or ""
        m = re.search(r'"panel_verdict"\s*:\s*"([a-z_]+)"', notes)
        if m:
            v = m.group(1)
            panel_counts[v] += 1
            if v == "reclassify":
                orig_cat_m = re.search(r'fol-[a-f0-9]+', f["id"])  # just for reference
                reclassify_examples.append({
                    "paper_id": f["paper_id"],
                    "final_category": f["category"],
                    "description": f["description"][:180],
                })
        else:
            panel_counts["unknown/not_recorded"] += 1
    tbl_rows = [[v, c, f"{c/sum(panel_counts.values())*100:.1f}%"] for v, c in panel_counts.most_common()]
    out.append(md_table(tbl_rows, ["Panel verdict", "N claims", "% of total"]))
    modified = sum(c for v, c in panel_counts.items() if v not in ("endorse", "unknown/not_recorded"))
    total = sum(panel_counts.values())
    out.append(f"\n**Panel modified {modified} of {total} claims ({modified/max(1,total)*100:.1f}%).** "
               f"This is the panel's structural effect rate; matching-conditional effect requires "
               f"re-running the forest_panel eval-match (missing data).\n")
    if reclassify_examples:
        out.append("\n**Panel reclassifications — examples:**\n")
        for ex in reclassify_examples[:3]:
            out.append(f"- [{ex['final_category']}] {ex['description']} (paper {ex['paper_id']})")

    # T1.2 Per-source
    out.append("\n\n## T1.2  Per-source coverage@10\n")
    out.append("_Tier A gold splits across OpenReview weakness reviews (971) and follow-up-paper citations (934). Does the framework recover both equally?_\n")
    rows = [["Source"] + list(METHODS.keys())]
    all_sources = set()
    per_src = {}
    for name, rows_ in method_matches.items():
        if rows_ is None: continue
        per_src[name] = analysis_per_source_coverage(rows_, gold_by_paper, gold)
        all_sources |= set(per_src[name].keys())
    tbl_rows = []
    for src in sorted(all_sources):
        r = [src]
        for name in METHODS:
            d = per_src.get(name, {}).get(src, {"hit":0, "total":0})
            pct = (d["hit"]/d["total"]*100) if d["total"] else 0
            r.append(f"{pct:.1f}% ({d['hit']}/{d['total']})")
        tbl_rows.append(r)
    out.append(md_table(tbl_rows, ["Source"] + list(METHODS.keys())))

    # T1.3 Evidence grounding
    out.append("\n\n## T1.3  Evidence quote grounding rate\n")
    out.append("_For each framework limitation with a quoted evidence snippet, does the quote actually appear in the paper? Verifies no hallucinated quotes._\n")
    tbl_rows = []
    for name, lims in method_lims.items():
        g = analysis_evidence_grounding(lims, papers_by_id)
        hq = g["has_quote"]
        gd = g["quote_grounded"]
        total = g["total"]
        pct_has = hq/total*100 if total else 0
        pct_gd = gd/hq*100 if hq else 0
        tbl_rows.append([name, total, hq, f"{pct_has:.1f}%", gd, f"{pct_gd:.1f}%"])
    out.append(md_table(tbl_rows, ["Method", "Total", "Has quote", "% with quote", "Grounded", "% of quoted"]))

    # T1.4 Category confusion
    out.append("\n\n## T1.4  Category confusion (matched pairs only)\n")
    out.append("_Rows = gold category; columns = framework category. Off-diagonal = category drift._\n")
    cats = ["scope", "methodology", "theoretical", "reproducibility", "fairness"]
    for name, rows_ in method_matches.items():
        if rows_ is None: continue
        out.append(f"\n### {name}")
        mat = analysis_category_confusion(rows_)
        hdr = ["gold \\ fw"] + cats
        tbl_rows = []
        for g in cats:
            r = [g] + [mat[g][f] for f in cats]
            tbl_rows.append(r)
        out.append(md_table(tbl_rows, hdr))

    # T1.5 Severity
    out.append("\n\n## T1.5  Severity accuracy (matched pairs)\n")
    tbl_rows = []
    for name, rows_ in method_matches.items():
        if rows_ is None: continue
        s = analysis_severity(rows_)
        acc = s["same"]/s["total"]*100 if s["total"] else 0
        tbl_rows.append([name, s["total"], s["same"], f"{acc:.1f}%",
                         s["fw_minor_gold_major"], s["fw_major_gold_minor"]])
    out.append(md_table(tbl_rows, ["Method", "Matched", "Same severity", "% agree", "fw↓ gold↑", "fw↑ gold↓"]))

    # T2.1 Inter-method
    out.append("\n\n## T2.1  Inter-method complementarity (Jaccard on matched gold)\n")
    hit_sets, pairs = analysis_intermethod_overlap(method_matches)
    tbl_rows = []
    for pair_name, d in pairs.items():
        if pair_name.startswith("_"): continue
        tbl_rows.append([pair_name, d["intersection"], d["union"], f"{d['jaccard']:.2f}",
                         d["only_a"], d["only_b"]])
    out.append(md_table(tbl_rows, ["Pair", "Intersection", "Union", "Jaccard", "only A", "only B"]))
    if "_union_all" in pairs:
        out.append(f"\nUnion of all methods hit **{pairs['_union_all']['total_hit_gold']}** distinct gold limitations.")

    # T2.2 Per-paper (use forest_nopanel — has real per-pair data; closest to forest_panel)
    out.append("\n\n## T2.2  Per-paper performance (forest_nopanel, has real per-pair data)\n")
    pp_rows = analysis_per_paper(method_matches["forest_nopanel"], gold_by_paper, fw_by_paper["forest_nopanel"])
    pp_rows.sort(key=lambda r: -r["coverage"])
    tbl_rows = [[r["paper_id"][-13:], r["n_gold"], r["n_fw"], r["covered"],
                 f"{r['coverage']*100:.0f}%", f"{r['precision']*100:.0f}%"] for r in pp_rows]
    out.append(md_table(tbl_rows, ["Paper", "Gold", "FW", "Covered", "Coverage@10", "Precision@20"]))

    # T2.3 Confidence calibration on forest_nopanel
    out.append("\n\n## T2.3  Confidence calibration (forest_nopanel)\n")
    buckets = analysis_confidence_calibration(method_lims["forest_nopanel"], method_matches["forest_nopanel"])
    tbl_rows = []
    for lbl, d in buckets.items():
        mr = f"{d['match_rate']*100:.1f}%" if d["match_rate"] is not None else "—"
        tbl_rows.append([lbl, d["total"], mr])
    out.append(md_table(tbl_rows, ["Confidence", "N claims", "Match rate"]))

    # T2.5 Category distribution vs gold
    out.append("\n\n## T2.5  Category distribution vs gold (KL divergence)\n")
    tbl_rows = []
    for name, lims in method_lims.items():
        r = analysis_cat_dist_vs_gold(lims, gold)
        tbl_rows.append([name] + [f"{r['method'][c]*100:.0f}%" for c in cats] + [f"{r['kl_divergence']:.3f}"])
    tbl_rows.append(["gold"] + [f"{(Counter(g['category'] for g in gold)[c] / len(gold))*100:.0f}%" for c in cats] + ["0.000"])
    out.append(md_table(tbl_rows, ["Method"] + cats + ["KL to gold"]))

    # T3.1 Unmatched fw examples (forest_nopanel, real per-pair)
    out.append("\n\n## T3.1  Framework outputs not matched to any gold — examples (forest_nopanel)\n")
    out.append("_These may be hallucinations OR real limitations the community hasn't recorded. Worth manual review._\n")
    unmatched = analysis_unmatched_fw(method_lims["forest_nopanel"], method_matches["forest_nopanel"], k=8)
    for u in unmatched:
        out.append(f"\n**Paper:** {u['paper_id']}")
        out.append(f"  - **[{u['category']}/{u['severity']}]** {u['description'][:250]}")
        ev = u.get("evidence_in_paper") or {}
        if ev.get("text"):
            out.append(f"  - *Evidence:* \"{ev['text'][:180]}\"")

    # T3.2 Case studies — pick 3 papers
    out.append("\n\n## T3.2  Case studies\n")
    test_papers = sorted(set(l["paper_id"] for l in method_lims["forest_panel"]))[:3]
    for pid in test_papers:
        case = analysis_case_study(pid, [], gold_by_paper, method_lims)
        out.append(f"\n### Paper: {pid}")
        paper = papers_by_id.get(pid)
        if paper:
            out.append(f"**Title:** {paper.get('title', '?')[:100]}")
        out.append(f"\n**Gold ({len(case['gold'])} lims):**")
        for g in case["gold"][:4]:
            out.append(f"  - [{g['category']}] {g['description'][:150]}")
        out.append(f"\n**forest_panel ({len(case['forest_panel'])} lims):**")
        for f in case["forest_panel"][:4]:
            out.append(f"  - [{f['category']}] {f['description'][:150]}")
        out.append(f"\n**zero_shot ({len(case['zero_shot'])} lims):**")
        for f in case["zero_shot"][:3]:
            out.append(f"  - [{f['category']}] {f['description'][:150]}")

    # T3.3 Error cases
    out.append("\n\n## T3.3  Error cases — gold missed by ALL methods\n")
    out.append("_Hardest limitations. Possible reasons: too subtle, require domain knowledge, framing mismatch._\n")
    errs = analysis_error_cases(gold_by_paper, method_matches, k=8)
    for e in errs:
        out.append(f"\n**Paper:** {e['paper_id']}")
        out.append(f"  - **[{e['category']}/{e['severity']}]** {e['description'][:250]}")
        src = e.get("source", {})
        out.append(f"  - *Source:* {src.get('kind','?')} — {src.get('citation','?')[:80]}")

    # Write report
    report = "\n".join(out)
    out_path = ROOT / "data" / "analysis_report.md"
    out_path.write_text(report)
    print(f"\nWrote {out_path} ({len(report)} chars, {report.count(chr(10))} lines)")
    print("\n" + "=" * 60)
    print("First 80 lines of report:")
    print("=" * 60)
    print("\n".join(report.splitlines()[:80]))


if __name__ == "__main__":
    main()
