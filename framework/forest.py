"""
Forest-of-Limitations: the top-level framework.

For a single paper:
  - One Paper persona.
  - Five Skeptic personas (one per category).
  - Each Skeptic independently builds a critique tree rooted at a
    category-level concern.
  - The moderator rules each node valid / deflected / unclear and may
    recommend expansion.
  - Output: a forest (list of trees), plus a flat list of Limitation objects
    derived from the valid leaves.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional

from framework.debate import run_debate_node
from framework.personas import PaperPersona, Skeptic, make_skeptic_bench
from llm.client import LLMClient
from schema.models import (
    DebateNode,
    ForestResult,
    Limitation,
    LimitationCategory,
    Paper,
    Severity,
    SkepticTree,
    Source,
    SourceType,
)

log = logging.getLogger(__name__)


CATEGORY_ROOT_TOPICS = {
    LimitationCategory.SCOPE: (
        "Are the paper's claimed scope and generalization supported by its experiments?",
        "The Scope Skeptic challenges whether the paper's evaluation actually supports the breadth of claims made (domains, languages, tasks, data sizes).",
    ),
    LimitationCategory.METHODOLOGY: (
        "Do the paper's experimental choices bias the comparison in its favor?",
        "The Methodology Skeptic looks for unfair baselines, missing controls, data leakage, and evaluation metrics that favor the proposed method by construction.",
    ),
    LimitationCategory.REPRODUCIBILITY: (
        "Can this paper be faithfully reproduced from the information provided?",
        "The Reproducibility Skeptic audits missing hyperparameters, seed sensitivity, undocumented preprocessing, and unavailable assets.",
    ),
    LimitationCategory.FAIRNESS: (
        "Does the paper's evaluation mask subgroup disparities or demographic gaps?",
        "The Fairness Skeptic probes for aggregate metrics that hide subgroup gaps and blind spots in the data or analysis.",
    ),
    LimitationCategory.THEORETICAL: (
        "Are the paper's theoretical assumptions and proofs internally consistent and matched to its experiments?",
        "The Theoretical Skeptic verifies definitions, assumptions, and whether the formal setting matches the experimental one.",
    ),
}


def _leaf_to_limitation(
    paper: Paper,
    category: LimitationCategory,
    leaf: DebateNode,
    idx: int,
) -> Optional[Limitation]:
    """Convert a valid leaf debate node into a Limitation record.

    A node counts as valid if:
      - moderator verdict == "valid" AND (no panel, OR panel didn't reject)
    """
    # Accept nodes whose final verdict is "valid" — rejected_by_panel / deflected / unclear all fail
    if leaf.verdict != "valid":
        return None
    # Use the most-recent skeptic turn (after revision) as the description
    skeptic_turns = [t for t in leaf.turns if t.role == "skeptic"]
    if not skeptic_turns:
        return None
    description = skeptic_turns[-1].text.strip()
    # Use the leaf's (possibly panel-adjusted) category and severity
    severity = leaf.severity or Severity.MINOR
    # Attach panel metadata if present
    panel_turns = [t for t in leaf.turns if t.role == "panel"]
    panel_meta = panel_turns[-1].meta if panel_turns else {}
    uid = hashlib.sha1(f"{paper.id}|{leaf.node_id}|{idx}".encode()).hexdigest()[:8]
    lim = Limitation(
        id=f"{paper.id}:fol-{uid}",
        paper_id=paper.id,
        description=description,
        category=leaf.category,        # may have been reassigned by panel
        severity=severity,
        evidence_in_paper=leaf.evidence_in_paper,
        source=Source(
            kind=SourceType.EXPERT_ANNOTATION,
            citation="forest-of-limitations",
        ),
        llm_confidence=None,
    )
    # Attach panel annotations in annotator_notes (machine-readable JSON blob)
    if panel_meta:
        import json as _json
        lim.annotator_notes = (
            (lim.annotator_notes or "")
            + "\n[panel] " + _json.dumps(panel_meta, default=str)
        )
    return lim


def run_forest_of_limitations(
    paper: Paper,
    llm: LLMClient,
    moderator_llm: Optional[LLMClient] = None,
    categories: Optional[list[LimitationCategory]] = None,
    max_depth: int = 2,
    max_children: int = 2,
    total_expansion_budget_per_category: int = 4,
    verbose: bool = True,
) -> ForestResult:
    """
    Run the full forest.

    Args:
      paper:           target paper
      llm:             backbone for paper + skeptic personas
      moderator_llm:   backbone for the moderator; if None, reuse `llm`
      categories:      restrict to a subset of categories (defaults to all five)
      max_depth:       max depth of each skeptic's tree (root = 0)
      max_children:    max sub-concerns per node
      total_expansion_budget_per_category: caps total debates per skeptic
    """
    moderator_llm = moderator_llm or llm
    categories = categories or list(LimitationCategory)

    paper_persona = PaperPersona(paper=paper, llm=llm)

    trees: list[SkepticTree] = []
    all_limitations: list[Limitation] = []
    t0 = time.time()
    usage_start = (
        llm.total_usage.input_tokens + llm.total_usage.output_tokens
        if hasattr(llm, "total_usage") else 0
    )

    for cat in categories:
        skeptic = Skeptic(category=cat, llm=llm)
        topic, topic_desc = CATEGORY_ROOT_TOPICS[cat]
        if verbose:
            print(f"\n=== {skeptic.name} ===", flush=True)

        root, _ = run_debate_node(
            skeptic=skeptic,
            paper_persona=paper_persona,
            moderator=moderator_llm,
            topic=topic,
            topic_description=topic_desc,
            parent_id=f"root-{cat.value}",
            depth=0,
            max_depth=max_depth,
            max_children=max_children,
            max_expansions_remaining=total_expansion_budget_per_category,
        )
        tree = SkepticTree(category=cat, root=root)
        trees.append(tree)

        # Collect leaf limitations
        for i, leaf in enumerate(root.iter_leaves()):
            lim = _leaf_to_limitation(paper, cat, leaf, idx=i)
            if lim:
                all_limitations.append(lim)

        if verbose:
            _print_tree(root, indent="  ")

    meta: dict = {
        "duration_s": time.time() - t0,
        "model": getattr(llm, "model", "?"),
    }
    if hasattr(llm, "total_usage"):
        meta["total_tokens"] = (
            llm.total_usage.input_tokens + llm.total_usage.output_tokens - usage_start
        )
        meta["input_tokens"] = llm.total_usage.input_tokens
        meta["output_tokens"] = llm.total_usage.output_tokens

    return ForestResult(
        paper_id=paper.id,
        trees=trees,
        limitations=all_limitations,
        meta=meta,
    )


def _print_tree(node: DebateNode, indent: str = "") -> None:
    tag = f"[{node.verdict or '?'}"
    if node.severity:
        tag += f"/{node.severity.value}"
    tag += "]"
    print(f"{indent}{tag} {node.topic[:90]}")
    for c in node.children:
        _print_tree(c, indent + "  ")
