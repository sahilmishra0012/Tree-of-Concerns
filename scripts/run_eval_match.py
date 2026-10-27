#!/usr/bin/env python3
"""
Synchronous LLM-as-judge evaluation: compare system-generated limitations
against gold limitations and compute Coverage@K and Precision.

Usage:
    python scripts/run_eval_match.py \\
        --gold      data/tier_a/limitations.json \\
        --system    data/framework_run/framework_limitations.json \\
        --out_dir   data/eval_match
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.bedrock import BedrockSyncClient
from llm.prompts import MatchResult, get as get_prompt
from schema.models import Limitation

log = logging.getLogger("eval_match")

DEFAULT_MODEL = "global.anthropic.claude-opus-4-6-v1"
DEFAULT_REGION = "us-west-2"


def load_limitations(path: Path) -> list[Limitation]:
    """Load a limitations file. Accepts either a flat list or a {paper_id: [..]} dict."""
    data = json.load(open(path))
    if isinstance(data, dict):
        out = []
        for ls in data.values():
            for d in ls:
                out.append(Limitation.model_validate(d))
        return out
    return [Limitation.model_validate(d) for d in data]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--system", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--ks", default="1,3,5,10,20",
                    help="Cov@K values to compute, comma-separated")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--region", default=DEFAULT_REGION)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gold = load_limitations(args.gold)
    system = load_limitations(args.system)
    log.info("Loaded gold=%d, system=%d", len(gold), len(system))

    # Group by paper
    gold_by_pid = defaultdict(list)
    for l in gold:
        gold_by_pid[l.paper_id].append(l)
    sys_by_pid = defaultdict(list)
    for l in system:
        sys_by_pid[l.paper_id].append(l)

    common = sorted(set(gold_by_pid) & set(sys_by_pid))
    log.info("Papers in common: %d", len(common))

    client = BedrockSyncClient(model_id=args.model, region=args.region)
    prompt = get_prompt("match_limitations_v1")

    # match_responses[(pid, gold_idx, sys_idx)] = bool
    match_path = args.out_dir / "match_responses.json"
    matches = {}
    if match_path.exists():
        matches = {tuple(json.loads(k)): v for k, v in json.load(open(match_path)).items()}

    n_calls = 0
    for pid in common:
        for gi, g in enumerate(gold_by_pid[pid]):
            for si, s in enumerate(sys_by_pid[pid]):
                key = (pid, gi, si)
                if key in matches: continue
                msgs = prompt.build_messages(a=s.description, b=g.description)
                try:
                    obj = client.complete_json(
                        system=prompt.system, messages=msgs,
                        schema=MatchResult,
                        max_tokens=prompt.max_tokens,
                        temperature=prompt.temperature,
                    )
                    matches[key] = bool(obj.is_match)
                    n_calls += 1
                except Exception as e:
                    log.warning("match %s g=%d s=%d failed: %s", pid, gi, si, e)
                    matches[key] = False
        # persist
        json.dump({json.dumps(list(k)): v for k, v in matches.items()},
                  open(match_path, "w"), indent=2)
    log.info("Made %d match calls", n_calls)

    # Compute Cov@K and Precision per paper
    ks = [int(x) for x in args.ks.split(",")]
    per_paper = {}
    for pid in common:
        gold_ids = list(range(len(gold_by_pid[pid])))
        cov = {}
        for K in ks:
            matched_gold = set()
            for si in range(min(K, len(sys_by_pid[pid]))):
                for gi in gold_ids:
                    if matches.get((pid, gi, si)):
                        matched_gold.add(gi)
            cov[K] = len(matched_gold) / len(gold_ids) if gold_ids else 0.0
        # Precision via greedy bipartite
        claimed = set()
        n_matched = 0
        for si in range(len(sys_by_pid[pid])):
            for gi in gold_ids:
                if gi in claimed: continue
                if matches.get((pid, gi, si)):
                    n_matched += 1
                    claimed.add(gi)
                    break
        prec = n_matched / len(sys_by_pid[pid]) if sys_by_pid[pid] else 0.0
        per_paper[pid] = {"cov_at_k": cov, "precision": prec}

    # Aggregate
    n_papers = len(per_paper)
    agg_cov = {K: sum(p["cov_at_k"][K] for p in per_paper.values()) / max(n_papers, 1) for K in ks}
    agg_prec = sum(p["precision"] for p in per_paper.values()) / max(n_papers, 1)
    metrics = {
        "n_papers": n_papers,
        "n_match_calls": n_calls,
        "coverage_at_k_mean": agg_cov,
        "precision_mean": agg_prec,
    }
    json.dump(metrics, open(args.out_dir / "metrics.json", "w"), indent=2)
    json.dump(per_paper, open(args.out_dir / "per_paper.json", "w"), indent=2)
    log.info("Cov@K mean: %s", {K: f"{v:.3f}" for K, v in agg_cov.items()})
    log.info("Precision mean: %.3f", agg_prec)


if __name__ == "__main__":
    main()
