"""
Framework: debate tree node.

Each node runs (argue → respond → revise → adjudicate). Prompts come from
`llm.prompts`; this module only orchestrates.

Sync implementation. Batch-mode orchestration lives in
`framework/batch_orchestrator.py`.
"""
from __future__ import annotations

import hashlib
import logging

from framework.personas import PaperPersona, Skeptic
from llm.client import LLMClient
from llm.prompts import (
    ModeratorVerdict,
    PaperResponse,
    SkepticArgument,
    SkepticRevision,
    get as get_prompt,
)
from schema.models import (
    DebateNode,
    DebateTurn,
    EvidenceSpan,
    Severity,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node IDs
# ---------------------------------------------------------------------------

def _next_id(parent_id: str, seed: str) -> str:
    h = hashlib.sha1(f"{parent_id}|{seed}".encode()).hexdigest()[:6]
    return f"{parent_id}.{h}"


# ---------------------------------------------------------------------------
# Sync debate node
# ---------------------------------------------------------------------------

def run_debate_node(
    skeptic: Skeptic,
    paper_persona: PaperPersona,
    moderator: LLMClient,
    topic: str,
    topic_description: str,
    parent_id: str = "root",
    depth: int = 0,
    max_depth: int = 2,
    max_children: int = 3,
    max_expansions_remaining: int = 4,
) -> tuple[DebateNode, int]:
    """
    Run one sync debate node. Returns (node, remaining expansion budget).
    """
    node_id = _next_id(parent_id, topic[:40])
    node = DebateNode(
        node_id=node_id,
        topic=topic,
        description=topic_description,
        category=skeptic.category,
    )

    # --- 1. Skeptic argues ---
    argue_prompt = get_prompt("skeptic_argue_v1")
    try:
        arg: SkepticArgument = skeptic.llm.complete_json(
            system=skeptic.system_prompt(paper_persona.paper),
            messages=argue_prompt.build_messages(
                topic=topic, topic_description=topic_description,
            ),
            schema=SkepticArgument,
            max_tokens=argue_prompt.max_tokens,
            temperature=argue_prompt.temperature,
        )
    except Exception as e:
        log.warning("Skeptic argument failed at %s: %s", node_id, e)
        node.verdict = "unclear"
        return node, max_expansions_remaining

    node.turns.append(DebateTurn(
        role="skeptic",
        text=arg.description,
        meta={"topic": arg.topic, "evidence": arg.evidence_quote,
              "section": arg.evidence_section, "severity": arg.severity_guess},
    ))
    if arg.evidence_quote.strip():
        node.evidence_in_paper = EvidenceSpan(
            text=arg.evidence_quote.strip(),
            section=arg.evidence_section or None,
        )

    # --- 2. Paper responds ---
    respond_prompt = get_prompt("paper_respond_v1")
    try:
        resp: PaperResponse = paper_persona.llm.complete_json(
            system=paper_persona.system_prompt(),
            messages=respond_prompt.build_messages(
                topic=arg.topic,
                description=arg.description,
                evidence_quote=arg.evidence_quote,
                evidence_section=arg.evidence_section,
            ),
            schema=PaperResponse,
            max_tokens=respond_prompt.max_tokens,
            temperature=respond_prompt.temperature,
        )
    except Exception as e:
        log.warning("Paper response failed at %s: %s", node_id, e)
        node.verdict = "unclear"
        return node, max_expansions_remaining

    node.turns.append(DebateTurn(
        role="paper",
        text=resp.response,
        meta={"acknowledges": resp.acknowledges, "citation": resp.citation_quote},
    ))

    # --- 3. Skeptic revises ---
    revise_prompt = get_prompt("skeptic_revise_v1")
    try:
        rev: SkepticRevision = skeptic.llm.complete_json(
            system=skeptic.system_prompt(paper_persona.paper),
            messages=revise_prompt.build_messages(
                original_description=arg.description,
                evidence_quote=arg.evidence_quote,
                paper_response=resp.response,
                acknowledges=resp.acknowledges,
                citation_quote=resp.citation_quote,
            ),
            schema=SkepticRevision,
            max_tokens=revise_prompt.max_tokens,
            temperature=revise_prompt.temperature,
        )
    except Exception as e:
        log.warning("Skeptic revision failed at %s: %s", node_id, e)
        node.verdict = "unclear"
        return node, max_expansions_remaining

    node.turns.append(DebateTurn(
        role="skeptic",
        text=rev.revised_description,
        meta={"concedes": rev.concedes},
    ))

    # --- 4. Moderator verdict ---
    moderator_prompt = get_prompt("moderator_verdict_v1")
    try:
        verdict: ModeratorVerdict = moderator.complete_json(
            system=moderator_prompt.system,
            messages=moderator_prompt.build_messages(
                topic=topic,
                category=skeptic.category.value,
                argument_description=arg.description,
                evidence_quote=arg.evidence_quote,
                paper_response=resp.response,
                citation_quote=resp.citation_quote,
                acknowledges=resp.acknowledges,
                revised_description=rev.revised_description,
                concedes=rev.concedes,
                max_children=max_children,
            ),
            schema=ModeratorVerdict,
            max_tokens=moderator_prompt.max_tokens,
            temperature=moderator_prompt.temperature,
        )
    except Exception as e:
        log.warning("Moderator failed at %s: %s", node_id, e)
        node.verdict = "unclear"
        return node, max_expansions_remaining

    node.verdict = verdict.verdict.strip().lower()
    try:
        node.severity = Severity(verdict.severity.strip().lower())
    except ValueError:
        node.severity = Severity.MINOR
    node.turns.append(DebateTurn(
        role="moderator",
        text=verdict.reasoning,
        meta={"verdict": node.verdict,
              "severity": node.severity.value if node.severity else None,
              "should_expand": verdict.should_expand},
    ))

    if rev.concedes:
        node.verdict = "deflected"

    # --- 5. Expand into children ---
    if (
        node.verdict == "valid"
        and verdict.should_expand
        and depth < max_depth
        and max_expansions_remaining > 0
        and verdict.expansion_prompts
    ):
        remaining = max_expansions_remaining - 1
        for child_prompt_txt in verdict.expansion_prompts[:max_children]:
            if remaining <= 0:
                break
            child_topic = child_prompt_txt.strip()[:120]
            child_desc = f"Sub-concern drilling into: {child_topic}"
            child, remaining = run_debate_node(
                skeptic=skeptic,
                paper_persona=paper_persona,
                moderator=moderator,
                topic=child_topic,
                topic_description=child_desc,
                parent_id=node_id,
                depth=depth + 1,
                max_depth=max_depth,
                max_children=max_children,
                max_expansions_remaining=remaining,
            )
            node.children.append(child)
        max_expansions_remaining = remaining

    return node, max_expansions_remaining
