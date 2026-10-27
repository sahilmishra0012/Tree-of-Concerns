"""
End-to-end demo: run benchmark extraction AND forest-of-limitations on a real
arXiv paper, then save everything.

Usage:
    python scripts/run_demo.py --arxiv 2109.10500 --out data/2109.10500/

What happens:
  1. Fetch the paper (arxiv + ar5iv full text).
  2. Fetch up to K papers that cite it.
  3. Extract candidate limitations from citing-paper critique contexts (Tier A).
  4. Deduplicate candidates.
  5. Run the Forest-of-Limitations framework on the same paper.
  6. Save: paper.json, candidates.json, forest.json, summary.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make imports work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.extractor import deduplicate, extract_from_citing_papers
from benchmark.paper_fetch import fetch_arxiv_paper, fetch_citing_papers
from framework.forest import run_forest_of_limitations
from llm.client import LLMClient
from schema.models import Limitation, Paper


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "model_dump_json"):
        path.write_text(obj.model_dump_json(indent=2))
    elif isinstance(obj, list) and obj and hasattr(obj[0], "model_dump"):
        path.write_text(json.dumps([o.model_dump(mode="json") for o in obj], indent=2, default=str))
    else:
        path.write_text(json.dumps(obj, indent=2, default=str))


def summary_md(
    paper: Paper,
    candidates: list[Limitation],
    forest_result,
) -> str:
    lines = [
        f"# {paper.title}",
        "",
        f"arxiv: {paper.arxiv_id}  year: {paper.year}  sections: {len(paper.sections)}",
        "",
        "## Benchmark (Tier A candidates, LLM-extracted from citing papers)",
        "",
        f"{len(candidates)} candidate limitations:",
        "",
    ]
    for l in candidates:
        conf = f"{l.llm_confidence:.2f}" if l.llm_confidence is not None else "-"
        lines.append(
            f"- **[{l.category.value}/{l.severity.value}]** (conf={conf}) {l.description}"
        )
        if l.evidence_in_paper:
            lines.append(f"  - evidence: \"{l.evidence_in_paper.text[:200]}\"")
        lines.append(f"  - source: {l.source.short()}")
    lines += ["", "## Framework (Forest-of-Limitations)", ""]
    for tree in forest_result.trees:
        lines.append(f"### {tree.category.value.capitalize()}Skeptic")
        for leaf in tree.root.iter_leaves():
            if leaf.verdict == "valid":
                last_skeptic = next(
                    (t for t in reversed(leaf.turns) if t.role == "skeptic"), None
                )
                desc = last_skeptic.text if last_skeptic else leaf.description
                lines.append(f"- **[{leaf.severity.value if leaf.severity else '?'}]** {desc}")
                if leaf.evidence_in_paper:
                    lines.append(f"  - evidence: \"{leaf.evidence_in_paper.text[:200]}\"")
        lines.append("")
    lines.append("")
    lines.append(f"_Total framework limitations: {len(forest_result.limitations)}_")
    lines.append(f"_Framework meta: {forest_result.meta}_")
    return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--arxiv", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max_sources", type=int, default=10,
                    help="max citing papers to extract from")
    ap.add_argument("--skip_benchmark", action="store_true")
    ap.add_argument("--skip_framework", action="store_true")
    ap.add_argument("--max_depth", type=int, default=1)
    ap.add_argument("--max_children", type=int, default=2)
    ap.add_argument("--budget_per_category", type=int, default=3)
    ap.add_argument("--categories", nargs="+", default=None,
                    help="subset of categories to run, e.g. scope methodology")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    llm = LLMClient.from_env()
    print(f"LLM: {type(llm).__name__} model={llm.model}")

    # ---- Step 1: fetch paper ----
    print(f"\n[1/3] Fetching paper {args.arxiv}...")
    paper = fetch_arxiv_paper(args.arxiv, with_full_text=True)
    print(f"  title: {paper.title}")
    print(f"  sections: {len(paper.sections)}  full_text_chars: {len(paper.full_text or '')}")
    save_json(paper, out / "paper.json")

    # ---- Step 2: benchmark extraction (Tier A) ----
    candidates: list[Limitation] = []
    if not args.skip_benchmark:
        print(f"\n[2/3] Benchmark extraction...")
        citing = fetch_citing_papers(args.arxiv, max_results=args.max_sources)
        print(f"  found {len(citing)} citing papers")
        save_json(citing, out / "citing_papers.json")

        candidates = extract_from_citing_papers(llm, paper, citing, max_sources=args.max_sources)
        print(f"  extracted {len(candidates)} candidate limitations")
        if candidates:
            candidates = deduplicate(llm, candidates)
            print(f"  after dedup: {len(candidates)}")
        save_json(candidates, out / "candidates.json")

    # ---- Step 3: framework (Forest-of-Limitations) ----
    if not args.skip_framework:
        print(f"\n[3/3] Framework (Forest-of-Limitations)...")
        from schema.models import LimitationCategory as LC
        cats = None
        if args.categories:
            cats = [LC(c) for c in args.categories]
        forest = run_forest_of_limitations(
            paper, llm,
            categories=cats,
            max_depth=args.max_depth,
            max_children=args.max_children,
            total_expansion_budget_per_category=args.budget_per_category,
            verbose=True,
        )
        print(f"\n  framework found {len(forest.limitations)} limitations")
        save_json(forest, out / "forest.json")
    else:
        forest = None

    # Summary
    md = summary_md(paper, candidates, forest) if forest else "# summary\n(forest skipped)"
    (out / "summary.md").write_text(md)
    print(f"\nWrote outputs to: {out}")
    print(f"  - paper.json      ({(out/'paper.json').stat().st_size} bytes)")
    if not args.skip_benchmark:
        print(f"  - candidates.json  (~{len(candidates)} items)")
    if not args.skip_framework:
        print(f"  - forest.json")
    print(f"  - summary.md")


if __name__ == "__main__":
    main()
