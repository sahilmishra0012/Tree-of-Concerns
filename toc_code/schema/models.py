"""
LimitTree schema.

Defines the core data types for the benchmark and framework:
  - Paper: a scientific paper with parsed sections
  - Limitation: a specific critique / unstated limitation of a paper
  - EvidenceSpan: text in the paper that the limitation refers to
  - Source: the external source (follow-up paper, review, etc.) that originated
      this limitation, or
      "expert-annotation" (Tier C).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LimitationCategory(str, Enum):
    SCOPE = "scope"                      # over-claimed generalization, untested conditions
    METHODOLOGY = "methodology"          # biased eval, missing baseline, confound, data leakage
    REPRODUCIBILITY = "reproducibility"  # missing hyperparams, unavailable data, seed sensitivity
    FAIRNESS = "fairness"                # subgroup disparities, demographic gaps, biased data
    THEORETICAL = "theoretical"          # unjustified assumption, circular reasoning, proof gap


class Severity(str, Enum):
    MINOR = "minor"
    MAJOR = "major"


class SourceType(str, Enum):
    FOLLOWUP_PAPER = "followup_paper"        # a later paper that explicitly critiques this one
    OPENREVIEW = "openreview"                # OpenReview weakness comment (unrebutted)
    CORRECTION = "correction"                # errata / retraction / correction notice
    EXPERT_ANNOTATION = "expert_annotation"  # manually produced by a human expert


class AnnotationStatus(str, Enum):
    CANDIDATE = "candidate"        # proposed by the LLM pipeline, not yet verified
    ACCEPTED = "accepted"          # human verified
    EDITED = "edited"              # human accepted with edits
    REJECTED = "rejected"          # human rejected


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

class EvidenceSpan(BaseModel):
    """A span of text in a paper, used as evidence for a limitation."""
    text: str                                    # exact quoted text from the paper
    section: Optional[str] = None                # section title ("Experiments", etc.)
    start_char: Optional[int] = None             # character offset in the full paper
    end_char: Optional[int] = None


class Source(BaseModel):
    """The external source that surfaced this limitation."""
    kind: SourceType
    citation: Optional[str] = None               # e.g., "Smith et al. 2024, arXiv:2403.XXXXX"
    quote: Optional[str] = None                  # verbatim text from the source
    url: Optional[str] = None                    # arXiv / OpenReview / DOI link

    def short(self) -> str:
        """Short human-readable tag."""
        return f"{self.kind.value}:{self.citation or self.url or '?'}"


class Limitation(BaseModel):
    """A single limitation of a target paper."""
    # Identity
    id: str                                      # unique, e.g., "arxiv:2301.12345:lim-003"
    paper_id: str                                # id of the target paper

    # Content
    description: str                             # 1-2 sentence limitation
    category: LimitationCategory
    severity: Severity

    # Grounding
    evidence_in_paper: Optional[EvidenceSpan] = None   # where in the paper is the concern?
    source: Source                                     # who raised it / how was it surfaced?

    # Meta
    status: AnnotationStatus = AnnotationStatus.CANDIDATE
    annotator_notes: Optional[str] = None              # free-form human annotator notes
    # If EDITED, the original LLM-proposed description is kept here:
    original_description: Optional[str] = None
    # Confidence of the LLM (0.0-1.0) if the extractor reports it:
    llm_confidence: Optional[float] = None


class Paper(BaseModel):
    """A scientific paper with parsed sections."""
    id: str                                      # canonical, e.g., "arxiv:2109.10500"
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    sections: list["PaperSection"] = Field(default_factory=list)
    full_text: Optional[str] = None              # raw concatenation of sections if available

    def as_context(self, max_chars: int = 30000) -> str:
        """Dump the paper as a string for LLM context, truncated to max_chars."""
        parts = [f"TITLE: {self.title}\n\nABSTRACT: {self.abstract}\n"]
        for s in self.sections:
            parts.append(f"\n## {s.title}\n\n{s.text}")
        joined = "".join(parts)
        return joined[:max_chars]


class PaperSection(BaseModel):
    title: str
    text: str
    order: int = 0


# Forward ref resolution
Paper.model_rebuild()


# ---------------------------------------------------------------------------
# Benchmark containers
# ---------------------------------------------------------------------------

class PaperAnnotation(BaseModel):
    """All limitations attached to a single paper (across tiers)."""
    paper: Paper
    limitations: list[Limitation] = Field(default_factory=list)
    tier: str = "A"                              # "A" | "B" | "C"

    def gold(self) -> list[Limitation]:
        """Limitations treated as gold (ACCEPTED or EDITED)."""
        return [
            l for l in self.limitations
            if l.status in (AnnotationStatus.ACCEPTED, AnnotationStatus.EDITED)
        ]


class Benchmark(BaseModel):
    """Top-level benchmark container."""
    name: str = "LimitTree"
    version: str = "0.1"
    tier_A: list[PaperAnnotation] = Field(default_factory=list)
    tier_C: list[PaperAnnotation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Framework output
# ---------------------------------------------------------------------------

class DebateTurn(BaseModel):
    role: str                    # "skeptic" | "paper" | "moderator"
    text: str
    meta: dict = Field(default_factory=dict)


class DebateNode(BaseModel):
    """A node in a skeptic's critique tree."""
    node_id: str
    topic: str                                   # e.g., "over-claimed zero-shot generalization"
    description: str
    category: LimitationCategory
    turns: list[DebateTurn] = Field(default_factory=list)
    verdict: Optional[str] = None                # "valid" | "deflected" | "unclear"
    severity: Optional[Severity] = None
    children: list["DebateNode"] = Field(default_factory=list)
    evidence_in_paper: Optional[EvidenceSpan] = None

    def iter_leaves(self):
        if not self.children:
            yield self
            return
        for c in self.children:
            yield from c.iter_leaves()


DebateNode.model_rebuild()


class SkepticTree(BaseModel):
    """One Skeptic's full critique tree."""
    category: LimitationCategory
    root: DebateNode


class ForestResult(BaseModel):
    """The framework's output: five skeptic trees plus derived limitations."""
    paper_id: str
    trees: list[SkepticTree] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    # Token / cost counters, etc.
    meta: dict = Field(default_factory=dict)


if __name__ == "__main__":
    # Sanity check: round-trip through JSON.
    p = Paper(id="arxiv:2109.10500", title="HyperExpan", abstract="...")
    l = Limitation(
        id="arxiv:2109.10500:lim-001",
        paper_id=p.id,
        description="Depth of 13 is a shallow hyperbolic embedding, little distortion observable",
        category=LimitationCategory.SCOPE,
        severity=Severity.MINOR,
        source=Source(kind=SourceType.EXPERT_ANNOTATION, citation="our own")
    )
    print(l.model_dump_json(indent=2))
