#!/usr/bin/env python3
"""
Synchronous baselines runner: Zero-shot, Single-skeptic CoT, and DIAGPaper.

Pick the baseline with --method:
    --method zero_shot       single-shot list extraction
    --method single_skeptic  CoT single agent across all five categories
    --method diagpaper       reviewer/author dialogue over 20 review dimensions

Usage:
    python scripts/run_baseline.py --papers_root data/corpus/papers \\
        --out_dir data/baseline_zs --method zero_shot
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.bedrock import BedrockSyncClient
from llm.prompts import LLMLimitationList, get as get_prompt
from schema.models import Paper, Source, SourceType, Limitation
from benchmark.extractor import _llm_to_limitation

log = logging.getLogger("run_baseline")

DEFAULT_MODEL = "global.anthropic.claude-opus-4-6-v1"
DEFAULT_REGION = "us-west-2"

DIAGPAPER_DIMENSIONS = [
    "Is the problem studied important and well-motivated?",
    "Has the author fully shown the problem importance convincingly?",
    "Any missing related works that need to be cited and discussed?",
    "Is the Related Work section well organized and does it indicate novelties?",
    "Do figures and tables well support the claims and align with descriptions?",
    "Any conflicting descriptions in this paper?",
    "Is the proposed approach novel?",
    "Any unclear or confusing part in the approach description?",
    "Any undiscussed limitations in the proposed approach?",
    "Any methodological flaws that could invalidate the results?",
    "If a new dataset is presented, is its construction clear and professional?",
    "Are the datasets representative enough for this target problem?",
    "Are all necessary experiments conducted thoroughly?",
    "Are the baselines representative enough? Any missing baselines?",
    "Do experimental analyses provide sufficient insight or just superficial description?",
    "Does the proposed approach show better performance than state-of-the-art?",
    "Are evaluation metrics appropriate for the tasks?",
    "Any grammar errors or severe writing issues?",
    "Is the dataset construction necessary and convincing?",
    "Are there any special data quality concerns?",
]


def parse_json_block(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None


def run_single_prompt(client, prompt_name, paper, paper_context_chars=12000):
    """Used for zero_shot and single_skeptic."""
    p = get_prompt(prompt_name)
    msgs = p.build_messages(paper_context=paper.as_context(max_chars=paper_context_chars))
    resp = client.complete(system=p.system, messages=msgs,
                           max_tokens=p.max_tokens, temperature=p.temperature)
    obj = parse_json_block(resp.text)
    if not obj: return []
    parsed = LLMLimitationList.model_validate(obj)
    src = Source(kind=SourceType.EXPERT_ANNOTATION, citation=f"{prompt_name} baseline")
    out = []
    for k, c in enumerate(parsed.limitations):
        lim = _llm_to_limitation(c, paper, src, k)
        if lim:
            out.append(lim)
    return out


DIAGPAPER_REVIEWER_SYSTEM = """\
You are a specialized paper reviewer focusing on the following review dimension: "{dim}".

Given a scientific paper, identify ONE specific weakness related to this dimension.
The weakness must be specific to THIS paper, grounded in concrete evidence, and
clearly related to the dimension above. If you cannot identify a valid weakness,
output empty fields.

Output ONLY a JSON object:
{{"weakness_text": "...", "category": "methodology|scope|theoretical|reproducibility|fairness",
  "location": "...", "evidence_quote": "..."}}
"""

DIAGPAPER_AUTHOR_SYSTEM = """\
You are the author of the paper below. A reviewer raised a weakness. Evaluate it
against the paper. Rate validity (fully_valid|partially_valid|invalid) and
evidence_strength (substantial|moderate|weak).

PAPER: {paper_context}

Output ONLY:
{{"validity": "...", "evidence_strength": "...", "rebuttal": "..."}}
"""


def run_diagpaper(client, paper, paper_context_chars=12000):
    paper_ctx = paper.as_context(max_chars=paper_context_chars)
    src = Source(kind=SourceType.EXPERT_ANNOTATION, citation="DIAGPaper baseline")
    out: list[Limitation] = []
    for dim in DIAGPAPER_DIMENSIONS:
        # Wave 1: reviewer proposes
        sys_msg = DIAGPAPER_REVIEWER_SYSTEM.format(dim=dim)
        try:
            r = client.complete(system=sys_msg,
                                messages=[{"role": "user",
                                           "content": f"Paper: {paper_ctx}\n\nIdentify a weakness for: {dim}"}],
                                max_tokens=500, temperature=0.3)
            obj = parse_json_block(r.text)
        except Exception as e:
            log.warning("  DIAGPaper rev '%s' failed: %s", dim[:30], e); continue
        if not obj or not obj.get("weakness_text"):
            continue
        weakness_json = obj
        # Wave 2: author validates
        try:
            r2 = client.complete(system=DIAGPAPER_AUTHOR_SYSTEM.format(paper_context=paper_ctx),
                                 messages=[{"role": "user",
                                            "content": f"Reviewer weakness: {json.dumps(weakness_json)}\n\nEvaluate."}],
                                 max_tokens=400, temperature=0.2)
            v = parse_json_block(r2.text)
        except Exception:
            v = None
        if not v or v.get("validity") == "invalid" or v.get("evidence_strength") == "weak":
            continue
        # Build a Limitation
        from schema.models import LimitationCategory, Severity, EvidenceSpan
        cat = (weakness_json.get("category") or "methodology").lower()
        if cat not in {"scope","methodology","theoretical","reproducibility","fairness"}:
            cat = "methodology"
        sev = "major" if v.get("validity") == "fully_valid" else "minor"
        evidence = None
        if weakness_json.get("evidence_quote"):
            evidence = EvidenceSpan(text=weakness_json["evidence_quote"], section=weakness_json.get("location"))
        import hashlib
        uid = hashlib.sha1(f"{paper.id}|diag|{weakness_json['weakness_text'][:80]}".encode()).hexdigest()[:8]
        out.append(Limitation(
            id=f"{paper.id}:diag-{uid}",
            paper_id=paper.id,
            description=weakness_json["weakness_text"],
            category=LimitationCategory(cat),
            severity=Severity(sev),
            evidence_in_paper=evidence,
            source=src,
            llm_confidence=None,
        ))
    return out


METHOD_TO_PROMPT = {
    "zero_shot": "baseline_zero_shot_v1",
    "single_skeptic": "baseline_single_skeptic_v1",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--method", choices=["zero_shot", "single_skeptic", "diagpaper"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--region", default=DEFAULT_REGION)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"baseline_{args.method}.json"

    client = BedrockSyncClient(model_id=args.model, region=args.region)

    paper_dirs = sorted([d for d in args.papers_root.iterdir() if d.is_dir()])
    log.info("Running baseline=%s on %d papers", args.method, len(paper_dirs))

    results: dict[str, list[dict]] = {}
    if out_path.exists():
        try:
            results = json.load(open(out_path))
        except Exception:
            results = {}

    for i, pdir in enumerate(paper_dirs, 1):
        try:
            paper = Paper.model_validate(json.loads((pdir / "paper.json").read_text()))
        except Exception:
            continue
        if paper.id in results:
            continue
        t0 = time.time()
        try:
            if args.method == "diagpaper":
                lims = run_diagpaper(client, paper)
            else:
                lims = run_single_prompt(client, METHOD_TO_PROMPT[args.method], paper)
        except Exception as e:
            log.warning("[%d/%d] %s: %s", i, len(paper_dirs), paper.id, e)
            continue
        results[paper.id] = [l.model_dump(mode="json") for l in lims]
        out_path.write_text(json.dumps(results, indent=2))
        log.info("[%d/%d] %s: %d limitations (%.1fs)",
                 i, len(paper_dirs), paper.id, len(lims), time.time() - t0)

    n_total = sum(len(v) for v in results.values())
    log.info("Done. %d papers, %d limitations.", len(results), n_total)


if __name__ == "__main__":
    main()
