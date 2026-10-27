"""
Tier A: LLM-assisted extraction of candidate limitations.

Pipeline per target paper P:
  1. Fetch P (title, abstract, full text).
  2. Fetch papers that cite P via Semantic Scholar.
  3. For each citing paper F, filter citation contexts to critique-like
     sentences. Feed (P summary, F context) to the LLM and ask for structured
     limitation records.
  4. De-duplicate similar limitations across sources via LLM-assisted
     clustering.

Sync mode:
    extract_from_citing_papers(llm, paper, citing_papers) -> list[Limitation]

"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from llm.client import LLMClient
from llm.prompts import (
    DedupResult,
    LLMLimitation,
    LLMLimitationList,
    get as get_prompt,
)
from schema.models import (
    EvidenceSpan,
    Limitation,
    LimitationCategory,
    Paper,
    Severity,
    Source,
    SourceType,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt + request construction (shared between sync and batch paths)
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = "extract_limitation_v1"
DEDUP_PROMPT = "dedup_limitations_v1"


def build_extraction_request_payload(
    paper: Paper,
    source: Source,
    source_text: str,
    paper_context_chars: int = 12000,
) -> tuple[str, list[dict], Optional[int], Optional[float]]:
    """
    Build (system, messages, max_tokens, temperature) for one extraction call.
    This is shared between the sync client path and the batch path.
    """
    prompt = get_prompt(EXTRACT_PROMPT)
    system = prompt.system
    messages = prompt.build_messages(
        paper_context=paper.as_context(max_chars=paper_context_chars),
        source_label=source.short(),
        source_text=source_text,
    )
    return system, messages, prompt.max_tokens, prompt.temperature


# ---------------------------------------------------------------------------
# Validation / mapping
# ---------------------------------------------------------------------------

_VALID_CATEGORIES = {c.value for c in LimitationCategory}
_VALID_SEVERITIES = {s.value for s in Severity}


def _llm_to_limitation(
    proposal: LLMLimitation,
    paper: Paper,
    source: Source,
    idx: int,
) -> Optional[Limitation]:
    """Convert an LLM proposal into a Limitation; return None if invalid."""
    cat = proposal.category.strip().lower()
    sev = proposal.severity.strip().lower()
    if cat not in _VALID_CATEGORIES:
        log.warning("Invalid category %r, skipping", cat)
        return None
    if sev not in _VALID_SEVERITIES:
        log.warning("Invalid severity %r, coercing to minor", sev)
        sev = Severity.MINOR.value

    evidence = None
    if proposal.evidence_quote_from_paper.strip():
        evidence = EvidenceSpan(
            text=proposal.evidence_quote_from_paper.strip(),
            section=(proposal.evidence_section or None),
        )

    src = source.model_copy()
    if not src.quote and proposal.source_quote:
        src = src.model_copy(update={"quote": proposal.source_quote.strip()})

    uid = hashlib.sha1(
        f"{paper.id}|{idx}|{proposal.description[:80]}".encode()
    ).hexdigest()[:8]
    return Limitation(
        id=f"{paper.id}:lim-{uid}",
        paper_id=paper.id,
        description=proposal.description.strip(),
        category=LimitationCategory(cat),
        severity=Severity(sev),
        evidence_in_paper=evidence,
        source=src,
        llm_confidence=proposal.confidence,
    )


# ---------------------------------------------------------------------------
# Critique-context preparation
# ---------------------------------------------------------------------------

def _source_from_citing_paper(cp: dict) -> tuple[Source, str]:
    """Build (Source, source_text) from a Semantic Scholar citingPaper dict."""
    from benchmark.paper_fetch import filter_critique_contexts

    title = cp.get("title") or "?"
    ext_ids = cp.get("externalIds") or {}
    arxiv_id = ext_ids.get("ArXiv")
    citation = f"{title} ({cp.get('year')})" + (f" arXiv:{arxiv_id}" if arxiv_id else "")
    url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None
    source = Source(
        kind=SourceType.FOLLOWUP_PAPER,
        citation=citation,
        url=url,
    )
    contexts = cp.get("contexts") or []
    critique_ctx = filter_critique_contexts(contexts)
    source_text = (
        f"Title: {title}\nAbstract: {cp.get('abstract') or '(none)'}\n\n"
        f"Citation contexts (the citing paper's sentences around its citation of P):\n"
        + "\n".join(f"- {c}" for c in critique_ctx[:6])
    )
    return source, source_text, critique_ctx


# ---------------------------------------------------------------------------
# Sync entry points
# ---------------------------------------------------------------------------

def extract_from_source(
    llm: LLMClient,
    paper: Paper,
    source: Source,
    source_text: str,
    paper_context_chars: int = 12000,
) -> list[Limitation]:
    """
    Run a single (paper, source) extraction synchronously.
    """
    system, messages, max_tokens, temperature = build_extraction_request_payload(
        paper=paper, source=source, source_text=source_text,
        paper_context_chars=paper_context_chars,
    )
    try:
        result: LLMLimitationList = llm.complete_json(
            system=system,
            messages=messages,
            schema=LLMLimitationList,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        log.warning("LLM extraction failed for %s: %s", source.short(), e)
        return []

    out: list[Limitation] = []
    for i, p in enumerate(result.limitations):
        lim = _llm_to_limitation(p, paper, source, idx=i)
        if lim is not None:
            out.append(lim)
    return out


def extract_from_citing_papers(
    llm: LLMClient,
    paper: Paper,
    citing_papers: list[dict],
    max_sources: int = 20,
) -> list[Limitation]:
    """
    Sync path: iterate over each citing paper and extract limitations.
    """
    all_limits: list[Limitation] = []
    for cp in citing_papers[:max_sources]:
        source, source_text, critique_ctx = _source_from_citing_paper(cp)
        if not critique_ctx:
            continue
        limits = extract_from_source(llm, paper, source, source_text)
        log.info("Extracted %d candidate limitations from %s", len(limits), source.short())
        all_limits.extend(limits)
    return all_limits
