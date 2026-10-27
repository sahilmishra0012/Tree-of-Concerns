#!/usr/bin/env python3
"""
Synchronous driver for the Tree-of-Concerns framework.

For each paper in --papers_root, runs:
  1. Forest of five specialized skeptic debate trees (sync).
  2. Per-claim Panel Review reconciliation.

All LLM calls are synchronous via BedrockSyncClient. Outputs are written
incrementally per paper to support resume.

Usage:
    python scripts/run_framework.py \\
        --papers_root data/corpus/papers \\
        --out_dir data/framework_run \\
        [--max_depth 1 --max_children 2 --no_panel]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framework.forest import run_forest_of_limitations
from llm.bedrock import BedrockSyncClient
from llm.prompts import PanelReview, get as get_prompt
from schema.models import Paper, Limitation

log = logging.getLogger("run_framework")

DEFAULT_MODEL = "global.anthropic.claude-opus-4-6-v1"
DEFAULT_REGION = "us-west-2"


def panel_review_one(claim: Limitation, paper: Paper, client: BedrockSyncClient) -> Limitation:
    """Run Panel Review on a single valid claim and apply the action."""
    prompt = get_prompt("panel_review_v1")
    msgs = prompt.build_messages(
        topic=claim.description[:120],
        category=claim.category.value,
        severity=claim.severity.value,
        description=claim.description,
        evidence_quote=(claim.evidence_in_paper.text if claim.evidence_in_paper else ""),
        paper_response="(see per-node debate transcript)",
        moderator_reasoning="(see per-node debate transcript)",
    )
    try:
        resp = client.complete_json(
            system=prompt.system,
            messages=msgs,
            schema=PanelReview,
            max_tokens=prompt.max_tokens,
            temperature=prompt.temperature,
        )
    except Exception as e:
        log.warning("Panel review failed: %s", e)
        return claim

    if resp.verdict == "reject":
        return None
    if resp.verdict in ("reclassify",) and resp.final_category:
        try:
            from schema.models import LimitationCategory
            claim.category = LimitationCategory(resp.final_category)
        except Exception:
            pass
    if resp.verdict == "downgrade" and resp.final_severity:
        try:
            from schema.models import Severity
            claim.severity = Severity(resp.final_severity)
        except Exception:
            pass
    if claim.annotator_notes is None:
        claim.annotator_notes = ""
    claim.annotator_notes += f"\n[panel] verdict={resp.verdict}; reasoning={resp.reasoning}"
    return claim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_depth", type=int, default=1)
    ap.add_argument("--max_children", type=int, default=2)
    ap.add_argument("--budget", type=int, default=3,
                    help="per-category expansion budget (defensive cap)")
    ap.add_argument("--no_panel", action="store_true",
                    help="run forest only, skip Panel Review")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--region", default=DEFAULT_REGION)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "framework_limitations.json"

    client = BedrockSyncClient(model_id=args.model, region=args.region)

    paper_dirs = sorted([d for d in args.papers_root.iterdir() if d.is_dir()])
    log.info("Running ToC framework on %d papers (panel=%s)",
             len(paper_dirs), not args.no_panel)

    results: dict[str, list[dict]] = {}
    if out_path.exists():
        try:
            results = json.load(open(out_path))
            log.info("Resuming with %d papers already done", len(results))
        except Exception:
            results = {}

    for i, pdir in enumerate(paper_dirs, 1):
        try:
            paper = Paper.model_validate(json.loads((pdir / "paper.json").read_text()))
        except Exception as e:
            log.warning("[%d/%d] %s: load failed: %s", i, len(paper_dirs), pdir.name, e)
            continue
        if paper.id in results:
            continue

        t0 = time.time()
        try:
            forest = run_forest_of_limitations(
                paper=paper,
                llm=client,
                max_depth=args.max_depth,
                max_children=args.max_children,
                total_expansion_budget_per_category=args.budget,
                verbose=False,
            )
        except Exception as e:
            log.warning("[%d/%d] %s: forest failed: %s", i, len(paper_dirs), paper.id, e)
            continue

        lims = forest.limitations
        if not args.no_panel:
            kept: list[Limitation] = []
            for lim in lims:
                lim2 = panel_review_one(lim, paper, client)
                if lim2 is not None:
                    kept.append(lim2)
            lims = kept

        results[paper.id] = [l.model_dump(mode="json") for l in lims]
        out_path.write_text(json.dumps(results, indent=2))
        log.info("[%d/%d] %s: %d limitations (%.1fs)",
                 i, len(paper_dirs), paper.id, len(lims), time.time() - t0)

    n_total = sum(len(v) for v in results.values())
    log.info("Done. %d papers, %d total limitations.", len(results), n_total)


if __name__ == "__main__":
    main()
