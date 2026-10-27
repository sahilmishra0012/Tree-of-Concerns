"""
Category-Guided Hierarchical Retrieval with Evidence Escalation (CHREE).

Two-phase retrieval for the Forest-of-Limitations framework:

Phase 1 (argue stage): Category-conditioned retrieval.
  Each skeptic category has a retrieval profile specifying:
    - Which paper sections are most relevant (by title pattern)
    - Category-specific query templates that target where limitations hide

Phase 2 (revise stage): Evidence Escalation.
  After the paper persona responds, the skeptic re-retrieves using the
  paper's rebuttal as the query — finding the exact passages the paper
  cited in its defense, enabling targeted scrutiny.

Implementation: TF-IDF over section-aware chunks. No external models needed.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from nltk.tokenize import sent_tokenize


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A ~3-sentence segment of a paper section."""
    text: str
    section_title: str
    section_idx: int
    chunk_idx: int

    @property
    def id(self) -> str:
        return f"s{self.section_idx}c{self.chunk_idx}"


def chunk_paper(sections: list[dict], sentences_per_chunk: int = 3) -> list[Chunk]:
    """Chunk paper sections into ~3-sentence segments."""
    chunks = []
    for si, sec in enumerate(sections):
        title = sec.get("title", "")
        text = sec.get("text", "")
        if not text.strip():
            continue
        sents = sent_tokenize(text)
        for ci in range(0, len(sents), sentences_per_chunk):
            chunk_text = " ".join(sents[ci:ci + sentences_per_chunk])
            if len(chunk_text.strip()) < 20:
                continue
            chunks.append(Chunk(
                text=chunk_text,
                section_title=title,
                section_idx=si,
                chunk_idx=ci // sentences_per_chunk,
            ))
    return chunks


# ---------------------------------------------------------------------------
# Section relevance profiles per category
# ---------------------------------------------------------------------------

# Patterns for matching section titles to categories
SECTION_PROFILES = {
    "scope": {
        "primary": [r"abstract", r"intro", r"conclusion", r"discussion"],
        "secondary": [r"experiment", r"result", r"evaluation"],
        "query_templates": [
            "claims generality broad applicability comprehensive",
            "evaluation covers domains tasks settings",
            "limitations of scope coverage",
        ],
    },
    "methodology": {
        "primary": [r"experiment", r"evaluation", r"baseline", r"setup", r"result"],
        "secondary": [r"method", r"approach", r"model"],
        "query_templates": [
            "comparison baseline controlled fair tuning",
            "ablation study isolate component",
            "evaluation protocol metric confound",
        ],
    },
    "theoretical": {
        "primary": [r"method", r"approach", r"model", r"theory", r"proof", r"analysis"],
        "secondary": [r"prelim", r"background", r"formul"],
        "query_templates": [
            "assumption condition holds requires",
            "theorem proof lemma proposition",
            "approximation bound guarantee convergence",
        ],
    },
    "reproducibility": {
        "primary": [r"experiment", r"setup", r"implement", r"training", r"detail"],
        "secondary": [r"appendix", r"supplement", r"hyper"],
        "query_templates": [
            "hyperparameter learning rate batch size seed",
            "code data available released",
            "training details epochs optimizer",
        ],
    },
    "fairness": {
        "primary": [r"data", r"dataset", r"evaluation", r"ethic"],
        "secondary": [r"experiment", r"result", r"limit"],
        "query_templates": [
            "demographic subgroup bias disparity",
            "dataset composition representation",
            "fairness equity sensitive attribute",
        ],
    },
}


def _matches_patterns(title: str, patterns: list[str]) -> bool:
    title_lower = title.lower()
    return any(re.search(p, title_lower) for p in patterns)


def get_category_relevant_chunks(
    chunks: list[Chunk],
    category: str,
    max_chunks: int = 20,
) -> list[Chunk]:
    """Filter and rank chunks by section relevance to the category."""
    profile = SECTION_PROFILES.get(category, SECTION_PROFILES["methodology"])
    primary = []
    secondary = []
    other = []
    for c in chunks:
        if _matches_patterns(c.section_title, profile["primary"]):
            primary.append(c)
        elif _matches_patterns(c.section_title, profile["secondary"]):
            secondary.append(c)
        else:
            other.append(c)
    # Prioritize primary > secondary > other, up to max_chunks
    ranked = primary + secondary + other
    return ranked[:max_chunks]


# ---------------------------------------------------------------------------
# TF-IDF scoring
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer with stopword removal."""
    STOP = {"the", "a", "an", "of", "to", "in", "for", "on", "with", "and",
            "or", "but", "is", "are", "was", "were", "be", "been", "being",
            "this", "that", "these", "those", "it", "its", "we", "our",
            "which", "can", "may", "should", "would", "could", "do", "does",
            "did", "use", "uses", "used", "using", "not", "no", "has", "have",
            "had", "from", "by", "as", "at", "also", "more", "than", "each",
            "such", "where", "when", "how", "what", "all", "both", "between",
            "into", "through", "during", "before", "after", "above", "below",
            "then", "there", "here", "other", "some", "only", "very", "will"}
    tokens = re.findall(r"[a-z][a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOP and len(t) > 2]


def _compute_idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    """IDF over a list of tokenized documents."""
    n = len(corpus_tokens)
    df = Counter()
    for doc in corpus_tokens:
        for t in set(doc):
            df[t] += 1
    return {t: math.log((n + 1) / (freq + 1)) + 1 for t, freq in df.items()}


def tfidf_score(query_tokens: list[str], doc_tokens: list[str], idf: dict[str, float]) -> float:
    """TF-IDF cosine-like score between query and document."""
    if not query_tokens or not doc_tokens:
        return 0.0
    # Query TF-IDF vector (sparse)
    q_tf = Counter(query_tokens)
    d_tf = Counter(doc_tokens)
    # Dot product
    score = 0.0
    q_norm = 0.0
    for t, tf in q_tf.items():
        w = tf * idf.get(t, 1.0)
        q_norm += w * w
        if t in d_tf:
            score += w * (d_tf[t] * idf.get(t, 1.0))
    d_norm = sum((tf * idf.get(t, 1.0)) ** 2 for t, tf in d_tf.items())
    if q_norm == 0 or d_norm == 0:
        return 0.0
    return score / (math.sqrt(q_norm) * math.sqrt(d_norm))


# ---------------------------------------------------------------------------
# Phase 1: Category-Guided Retrieval (for argue stage)
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Result of a retrieval call."""
    chunks: list[Chunk]
    scores: list[float]
    context_text: str  # concatenated top-k chunks ready for prompt insertion


def retrieve_for_category(
    all_chunks: list[Chunk],
    category: str,
    top_k: int = 5,
) -> RetrievalResult:
    """
    Phase 1: Category-Guided Hierarchical Retrieval.
    
    1. Filter chunks to category-relevant sections
    2. Score against category-specific query templates via TF-IDF
    3. Return top-k chunks as focused context
    """
    # Step 1: section-aware filtering
    relevant = get_category_relevant_chunks(all_chunks, category, max_chunks=50)
    if not relevant:
        relevant = all_chunks[:30]  # fallback

    # Step 2: TF-IDF scoring against category query templates
    profile = SECTION_PROFILES.get(category, SECTION_PROFILES["methodology"])
    query_text = " ".join(profile["query_templates"])
    query_tokens = _tokenize(query_text)

    corpus_tokens = [_tokenize(c.text) for c in relevant]
    idf = _compute_idf(corpus_tokens)

    scored = []
    for i, chunk in enumerate(relevant):
        s = tfidf_score(query_tokens, corpus_tokens[i], idf)
        scored.append((s, chunk))

    # Sort by score descending
    scored.sort(key=lambda x: -x[0])
    top = scored[:top_k]

    chunks_out = [c for _, c in top]
    scores_out = [s for s, _ in top]
    context = "\n\n".join(
        f"[{c.section_title} | segment {c.id}]: {c.text}" for c in chunks_out
    )
    return RetrievalResult(chunks=chunks_out, scores=scores_out, context_text=context)


# ---------------------------------------------------------------------------
# Phase 2: Evidence Escalation (for revise stage)
# ---------------------------------------------------------------------------

def retrieve_escalation(
    all_chunks: list[Chunk],
    paper_rebuttal: str,
    top_k: int = 5,
) -> RetrievalResult:
    """
    Phase 2: Evidence Escalation.
    
    After the paper persona responds, re-retrieve using the paper's
    rebuttal text as the query. This finds the exact passages the paper
    cited in its defense, enabling the skeptic to scrutinize them.
    """
    # Tokenize the rebuttal as query
    query_tokens = _tokenize(paper_rebuttal)
    if not query_tokens:
        # Fallback: return first k chunks
        top = all_chunks[:top_k]
        context = "\n\n".join(
            f"[{c.section_title} | segment {c.id}]: {c.text}" for c in top
        )
        return RetrievalResult(chunks=top, scores=[0.0] * len(top), context_text=context)

    corpus_tokens = [_tokenize(c.text) for c in all_chunks]
    idf = _compute_idf(corpus_tokens)

    scored = []
    for i, chunk in enumerate(all_chunks):
        s = tfidf_score(query_tokens, corpus_tokens[i], idf)
        scored.append((s, chunk))

    scored.sort(key=lambda x: -x[0])
    top = scored[:top_k]

    chunks_out = [c for _, c in top]
    scores_out = [s for s, _ in top]
    context = "\n\n".join(
        f"[{c.section_title} | segment {c.id}]: {c.text}" for c in chunks_out
    )
    return RetrievalResult(chunks=chunks_out, scores=scores_out, context_text=context)


# ---------------------------------------------------------------------------
# Combined interface for the orchestrator
# ---------------------------------------------------------------------------

@dataclass
class PaperRetriever:
    """Pre-computed retrieval index for a single paper."""
    paper_id: str
    all_chunks: list[Chunk] = field(default_factory=list)

    @classmethod
    def from_paper(cls, paper_id: str, sections: list[dict]) -> "PaperRetriever":
        chunks = chunk_paper(sections)
        return cls(paper_id=paper_id, all_chunks=chunks)

    def retrieve_phase1(self, category: str, top_k: int = 5) -> RetrievalResult:
        """Phase 1: Category-guided retrieval for argue stage."""
        return retrieve_for_category(self.all_chunks, category, top_k=top_k)

    def retrieve_phase2(self, paper_rebuttal: str, top_k: int = 5) -> RetrievalResult:
        """Phase 2: Evidence escalation for revise stage."""
        return retrieve_escalation(self.all_chunks, paper_rebuttal, top_k=top_k)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path

    # Load a sample paper
    papers_root = Path("data/corpus/papers")
    sample = None
    for d in sorted(papers_root.iterdir()):
        pp = d / "paper.json"
        if pp.exists():
            sample = json.loads(pp.read_text())
            break
    if not sample:
        print("No sample paper found")
        exit(1)

    print(f"Paper: {sample['title'][:60]}")
    print(f"Sections: {len(sample.get('sections', []))}")

    sections = [{"title": s["title"], "text": s["text"]} for s in sample.get("sections", [])]
    retriever = PaperRetriever.from_paper(sample["id"], sections)
    print(f"Total chunks: {len(retriever.all_chunks)}")

    # Phase 1 for each category
    for cat in ["scope", "methodology", "theoretical", "reproducibility", "fairness"]:
        r = retriever.retrieve_phase1(cat, top_k=3)
        print(f"\n--- {cat} (Phase 1, top-3) ---")
        for i, (chunk, score) in enumerate(zip(r.chunks, r.scores)):
            print(f"  [{score:.3f}] [{chunk.section_title}] {chunk.text[:80]}...")

    # Phase 2 with a fake rebuttal
    fake_rebuttal = "We acknowledge that our evaluation is limited to a single dataset, but the ablation study in Table 3 demonstrates robustness across different hyperparameter settings."
    r2 = retriever.retrieve_phase2(fake_rebuttal, top_k=3)
    print(f"\n--- Evidence Escalation (Phase 2, top-3) ---")
    for i, (chunk, score) in enumerate(zip(r2.chunks, r2.scores)):
        print(f"  [{score:.3f}] [{chunk.section_title}] {chunk.text[:80]}...")
