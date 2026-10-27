#!/usr/bin/env python3
"""
Synchronous benchmark extraction (Tier A): Stage 1 candidate extraction +
Stage 2 dedup + unstated filter.

For each paper in the corpus that has either >=2 OpenReview reviewer-weakness
points or >=1 citing-paper critique context, this script produces the gold
limitations of ToC-Bench by running:

  Stage 1: For each (paper, source) pair, prompt Claude (or another LLM) to
           extract structured limitation candidates.
  Stage 2: For each paper with >=2 candidates, cluster near-duplicates;
           for each paper with a self-critical section, drop candidates that
           are already-acknowledged.

All LLM calls are synchronous via BedrockSyncClient (one HTTP request per
call, ~5-10 s each). Per-paper outputs are persisted incrementally so the
script can resume after interruption.

Usage:
    python scripts/run_tier_a.py --papers_root data/corpus/papers \\
                                  --out_dir data/tier_a
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

from benchmark.extractor import build_extraction_request_payload, _llm_to_limitation
from benchmark.paper_fetch import filter_critique_contexts
from llm.bedrock import BedrockSyncClient
from llm.prompts import LLMLimitationList, get as get_prompt
from schema.models import Paper, Source, SourceType, Limitation

log = logging.getLogger("tier_a")

DEFAULT_MODEL = "global.anthropic.claude-opus-4-6-v1"
DEFAULT_REGION = "us-west-2"


def safe_id(pid: str) -> str:
    return pid.replace("arxiv:", "arxiv_").replace("/", "_")


def parse_json_block(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def format_or_review(review: dict) -> str:
    lines = [f"OPENREVIEW REVIEW by {review.get('reviewer') or 'anonymous'}"]
    if review.get("rating"):
        lines.append(f"Rating: {review['rating']}")
    if review.get("summary"):
        lines.append(f"\nReviewer summary:\n{review['summary']}")
    if review.get("weaknesses"):
        lines.append(f"\nReviewer-stated weaknesses:\n{review['weaknesses']}")
    for i, wp in enumerate(review.get("weakness_points", []), 1):
        lines.append(f"  {i}. {wp.get('text', '').strip()}")
    return "\n".join(lines)


def format_silver_source(src: dict) -> str:
    contexts = src.get("critique_contexts") or []
    lines = [
        f"Title: {src.get('title') or '?'}",
        f"Year: {src.get('year') or '?'}",
        f"Abstract: {(src.get('abstract') or '')[:2000]}",
        "",
        "Citation contexts (the citing paper's sentences around its citation of P):",
    ]
    for c in contexts[:6]:
        lines.append(f"- {c}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 1: candidate extraction
# ---------------------------------------------------------------------------

def stage1(papers_root: Path, out_path: Path, client: BedrockSyncClient):
    raw: dict[str, list[dict]] = {}
    if out_path.exists():
        try:
            raw = json.load(open(out_path))
            log.info("Resuming with %d papers already in raw", len(raw))
        except Exception:
            raw = {}

    paper_dirs = sorted([d for d in papers_root.iterdir() if d.is_dir()])
    log.info("Stage 1: scanning %d corpus papers", len(paper_dirs))

    n_calls = 0
    for i, pdir in enumerate(paper_dirs, 1):
        pid = f"arxiv:{pdir.name.replace('arxiv_', '')}"
        if pid in raw:
            continue
        try:
            paper = Paper.model_validate(json.loads((pdir / "paper.json").read_text()))
        except Exception as e:
            log.warning("[%d/%d] %s: paper.json load failed: %s", i, len(paper_dirs), pid, e)
            continue

        sources: list[tuple[Source, str]] = []
        # OpenReview reviews with >= 2 weakness points
        or_path = pdir / "openreview.json"
        if or_path.exists():
            or_data = json.load(open(or_path))
            for review in (or_data.get("reviews") or []):
                if isinstance(review.get("weakness_points"), list) and len(review["weakness_points"]) >= 2:
                    src_obj = Source(
                        kind=SourceType.OPENREVIEW,
                        citation=f"OpenReview review by {review.get('reviewer','?')}",
                    )
                    sources.append((src_obj, format_or_review(review)))
        # Citation contexts
        cc_path = pdir / "critique_contexts.json"
        if cc_path.exists():
            cc = json.load(open(cc_path))
            for src in (cc or []):
                if not src.get("critique_contexts"):
                    continue
                src_obj = Source(
                    kind=SourceType.FOLLOWUP_PAPER,
                    citation=(src.get("title") or "")
                    + (f" ({src.get('year')})" if src.get("year") else ""),
                    url=src.get("arxiv_id") or "",
                )
                sources.append((src_obj, format_silver_source(src)))

        if not sources:
            raw[pid] = []
            continue

        paper_lims: list[Limitation] = []
        for j, (src_obj, src_text) in enumerate(sources, 1):
            sys_, msgs, mt, temp = build_extraction_request_payload(paper, src_obj, src_text)
            try:
                t0 = time.time()
                resp = client.complete(system=sys_, messages=msgs,
                                       max_tokens=mt or 2000,
                                       temperature=temp or 0.2)
                n_calls += 1
                obj = parse_json_block(resp.text)
                if not obj:
                    continue
                parsed = LLMLimitationList.model_validate(obj)
                for k, p in enumerate(parsed.limitations):
                    lim = _llm_to_limitation(p, paper, src_obj, len(paper_lims) + k)
                    if lim:
                        paper_lims.append(lim)
                log.info("[%d/%d] %s src=%d: %d candidates (%.1fs)",
                         i, len(paper_dirs), pid, j, len(parsed.limitations), time.time() - t0)
            except Exception as e:
                log.warning("  %s src=%d: %s", pid, j, e)

        raw[pid] = [l.model_dump(mode="json") for l in paper_lims]
        out_path.write_text(json.dumps(raw, indent=2))

    log.info("Stage 1 done. %d API calls, %d papers w/ candidates",
             n_calls, sum(1 for v in raw.values() if v))
    return raw


# ---------------------------------------------------------------------------
# Stage 2: dedup + unstated filter
# ---------------------------------------------------------------------------

NEG_FILTER_SYSTEM = """\
You are filtering candidate "unstated limitations" of a scientific paper P, using P's own
"Limitations" / "Future Work" / "Broader Impacts" section text. Your task: for each
candidate, decide whether it is ALREADY ACKNOWLEDGED by the paper authors in that section.

A candidate is "already_acknowledged: true" if the authors describe the same underlying
concern in their own section, even if worded differently. Be conservative — when in
doubt, mark false (keep the candidate).

Output ONLY a JSON object:
{
  "decisions": [
    {"candidate_id": "...", "already_acknowledged": <true|false>, "reasoning": "..."}
  ]
}
"""


def build_neg_filter_user(paper: Paper, stated: str, candidates: list[Limitation]) -> str:
    lines = [
        f"TARGET PAPER: {paper.title}\n",
        "PAPER'S OWN SELF-CRITICAL SECTION TEXT:",
        stated,
        "\n---\nCANDIDATE UNSTATED LIMITATIONS (id -> description):",
    ]
    for c in candidates:
        lines.append(f"\n  id: {c.id}")
        lines.append(f"  description: {c.description}")
    lines.append("\nFor each candidate, decide whether it is already_acknowledged. Output only the JSON.")
    return "\n".join(lines)


def load_stated(papers_root: Path, pid: str) -> str:
    p = papers_root / safe_id(pid) / "stated_limitations.json"
    if not p.exists():
        return ""
    sl = json.load(open(p))
    if isinstance(sl, list):
        return "\n\n".join(s.get("text", "") if isinstance(s, dict) else str(s) for s in sl)
    if isinstance(sl, dict):
        return sl.get("text", "") or json.dumps(sl)
    return str(sl)


def stage2(raw: dict, papers_root: Path, out_path: Path, client: BedrockSyncClient):
    cleaned: dict[str, list[dict]] = {}
    if out_path.exists():
        try:
            cleaned = json.load(open(out_path))
        except Exception:
            cleaned = {}

    papers_with = [pid for pid, v in raw.items() if v]
    log.info("Stage 2: %d papers with candidates", len(papers_with))

    for i, pid in enumerate(papers_with, 1):
        if pid in cleaned:
            continue
        lims = [Limitation.model_validate(d) for d in raw[pid]]

        # Dedup if >= 2
        if len(lims) >= 2:
            prompt = get_prompt("dedup_limitations_v1")
            input_listing = "\n".join(
                f"- id={l.id}\n  category={l.category.value}\n"
                f"  severity={l.severity.value}\n  description={l.description}"
                for l in lims
            )
            try:
                resp = client.complete(
                    system=prompt.system,
                    messages=prompt.build_messages(input_listing=input_listing),
                    max_tokens=prompt.max_tokens,
                    temperature=prompt.temperature,
                )
                obj = parse_json_block(resp.text)
                if obj and obj.get("clusters"):
                    keep = {c.get("canonical_id") for c in obj["clusters"] if c.get("canonical_id")}
                    if keep:
                        before = len(lims)
                        lims = [l for l in lims if l.id in keep]
                        log.info("[%d/%d] %s: dedup %d -> %d", i, len(papers_with), pid, before, len(lims))
            except Exception as e:
                log.warning("  %s dedup: %s", pid, e)

        # Negative filter
        stated = load_stated(papers_root, pid)
        if stated and lims:
            paper = Paper.model_validate(json.loads(
                (papers_root / safe_id(pid) / "paper.json").read_text()))
            user = build_neg_filter_user(paper, stated, lims)
            try:
                resp = client.complete(
                    system=NEG_FILTER_SYSTEM,
                    messages=[{"role": "user", "content": user}],
                    max_tokens=3000, temperature=0.0,
                )
                obj = parse_json_block(resp.text)
                if obj and obj.get("decisions"):
                    drop = {d.get("candidate_id") for d in obj["decisions"] if d.get("already_acknowledged")}
                    if drop:
                        before = len(lims)
                        lims = [l for l in lims if l.id not in drop]
                        log.info("[%d/%d] %s: neg-filter -%d kept %d", i, len(papers_with), pid, before-len(lims), len(lims))
            except Exception as e:
                log.warning("  %s neg-filter: %s", pid, e)

        cleaned[pid] = [l.model_dump(mode="json") for l in lims]
        out_path.write_text(json.dumps(cleaned, indent=2))

    n_lims = sum(len(v) for v in cleaned.values())
    log.info("Stage 2 done: %d papers, %d clean limitations",
             sum(1 for v in cleaned.values() if v), n_lims)
    return cleaned


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers_root", type=Path, default=Path("data/corpus/papers"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/tier_a"))
    ap.add_argument("--stage", choices=["1", "2", "both"], default="both")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--region", default=DEFAULT_REGION)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    client = BedrockSyncClient(model_id=args.model, region=args.region)
    raw_path = args.out_dir / "raw_limitations.json"
    clean_path = args.out_dir / "limitations.json"

    if args.stage in ("1", "both"):
        raw = stage1(args.papers_root, raw_path, client)
    else:
        raw = json.load(open(raw_path))
    if args.stage in ("2", "both"):
        stage2(raw, args.papers_root, clean_path, client)


if __name__ == "__main__":
    main()
