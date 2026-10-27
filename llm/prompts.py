"""
Centralized prompt registry.

All system + user prompt templates live here, one per task. Each prompt has:
  - a unique name (for versioning and logging)
  - a version string
  - a system string
  - a render_user(**kwargs) callable that builds the user message

Rationale:
  - Makes prompt iteration a single-file edit (no hunting through modules).
  - Lets us A/B test prompt versions and log which version produced what.
  - Separates "what we ask the LLM" from "when we ask it".

To add a new prompt: define a @register-ed class subclassing `Prompt`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Type

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Prompt base + registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, "Prompt"] = {}


@dataclass
class Prompt:
    """A versioned prompt: a system string + a user-message renderer."""
    name: str
    version: str
    system: str
    render_user: Callable[..., str]
    # Optional associated response schema for JSON-typed completion
    response_schema: Optional[Type[BaseModel]] = None
    # Recommended inference hyperparameters
    max_tokens: int = 1500
    temperature: float = 0.2
    description: str = ""

    def __post_init__(self):
        _REGISTRY[self.name] = self

    def build_messages(self, **kwargs) -> list[dict]:
        return [{"role": "user", "content": self.render_user(**kwargs)}]


def get(name: str) -> Prompt:
    if name not in _REGISTRY:
        raise KeyError(f"No prompt named {name!r}. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name]


def all_prompts() -> dict[str, Prompt]:
    return dict(_REGISTRY)


# ---------------------------------------------------------------------------
# Response schemas (shared)
# ---------------------------------------------------------------------------

from pydantic import Field


class LLMLimitation(BaseModel):
    description: str = Field(description="1-2 sentence limitation, specific to paper P")
    category: str = Field(description="One of: scope, methodology, reproducibility, fairness, theoretical")
    severity: str = Field(description="One of: minor, major")
    evidence_quote_from_paper: str = Field(description="Exact quote from paper P. Empty if none.")
    evidence_section: str = Field(description="Section of P where the evidence appears. Empty if unknown.")
    source_quote: str = Field(description="Exact quote from the source that surfaced this limitation.")
    confidence: float = Field(description="LLM's 0-1 confidence this is a real unstated limitation")


class LLMLimitationList(BaseModel):
    limitations: list[LLMLimitation]
    reasoning: str = Field(description="Brief rationale for what was found (or nothing).")


class DedupGroup(BaseModel):
    representative_description: str
    member_ids: list[str]
    category: str
    severity: str


class DedupResult(BaseModel):
    groups: list[DedupGroup]


class SkepticArgument(BaseModel):
    topic: str
    description: str
    evidence_quote: str
    evidence_section: str
    severity_guess: str


class PaperResponse(BaseModel):
    acknowledges: bool
    response: str
    citation_quote: str


class SkepticRevision(BaseModel):
    revised_description: str
    concedes: bool


class ModeratorVerdict(BaseModel):
    verdict: str
    severity: str
    should_expand: bool
    expansion_prompts: list[str] = Field(default_factory=list)
    reasoning: str


class MatchResult(BaseModel):
    is_match: bool
    reasoning: str


class PanelReview(BaseModel):
    """
    A post-moderator cross-skeptic review. After a claim is ruled valid by
    the per-category moderator, a panel representing all 5 skeptic
    perspectives re-evaluates it: is the category correct? is it
    subsumed by another skeptic's concern? is severity right? or should
    it be rejected entirely as a false positive?
    """
    verdict: str  # "endorse" | "reclassify" | "downgrade" | "merge" | "reject"
    final_category: str  # scope | methodology | reproducibility | fairness | theoretical
    final_severity: str  # minor | major
    cross_category_concerns: list[str] = Field(
        default_factory=list,
        description="Other categories that also apply (short labels). Empty if unimodal.",
    )
    reasoning: str  # 1-3 sentences


# ---------------------------------------------------------------------------
# Shared text blocks
# ---------------------------------------------------------------------------

CATEGORY_HELP = """\
Categories:
  - scope: over-claimed generalization, untested conditions
  - methodology: biased eval, missing baseline, confound, data leakage
  - reproducibility: missing hyperparams, unavailable data, seed sensitivity
  - fairness: subgroup disparities, demographic gaps, biased data
  - theoretical: unjustified assumption, circular reasoning, proof gap
"""


# ---------------------------------------------------------------------------
# 1. Limitation extraction (Tier A)
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """\
You are an expert research scientist helping build a benchmark of *unstated limitations* in scientific papers.

You will be given:
  (a) A target paper P (title, abstract, and relevant sections).
  (b) A "source" — a piece of external evidence (a later paper that cites P, or an OpenReview weakness comment) that appears to criticize or raise concern about P.

Your job: extract limitations of P that are raised by the source but NOT explicitly acknowledged by P itself.

CRITERIA for a valid limitation:
  1. Specific. Not "the method has limitations"; something concrete like "assumes a balanced tree, which WordNet-Noun violates at depth 15+".
  2. Grounded in P. Provide an exact quote from P showing where the limitation hides.
  3. Surfaced by the source. Provide an exact quote from the source.
  4. NOT already acknowledged in P's own "limitations" / "future work" section.
  5. Categorized into one of the five categories.

{category_help}

If the source does NOT raise a valid unstated limitation (e.g., it just cites P as prior work or agrees with P), return an empty list.

Be conservative. It is better to return nothing than to manufacture a limitation.

# OUTPUT FORMAT — CRITICAL

You must output ONLY a single JSON object, nothing else. No prose, no analysis, no markdown code fences, no explanations. Your ENTIRE response must start with "{{" and end with "}}".

The JSON object has EXACTLY this structure (use these EXACT field names):

{{
  "limitations": [
    {{
      "description": "<1-2 sentence description of the limitation, specific to paper P>",
      "category": "<one of: scope | methodology | reproducibility | fairness | theoretical>",
      "severity": "<one of: minor | major>",
      "evidence_quote_from_paper": "<exact quote from paper P showing where the limitation hides; empty string if none>",
      "evidence_section": "<section of P where the quote appears; empty string if unknown>",
      "source_quote": "<exact quote from the source that surfaces this limitation>",
      "confidence": <float 0.0-1.0>
    }}
  ],
  "reasoning": "<1-3 sentence rationale for what was found (or why nothing was found)>"
}}

RULES:
- Use the EXACT field names above. Do not rename, abbreviate, or invent alternatives.
- "limitations" is a list; it may be empty [].
- Every value is required. Use empty string "" where no evidence is available.
- Output JSON and nothing else. Do not add any text before or after the JSON object.
"""


def _render_extract_user(paper_context: str, source_label: str, source_text: str) -> str:
    return f"""\
TARGET PAPER (P):
{paper_context}

---

SOURCE ({source_label}):
{source_text}

---

Extract any unstated limitations of P that this source surfaces. Be conservative and specific.
"""


Prompt(
    name="extract_limitation_v1",
    version="1.0",
    system=_EXTRACT_SYSTEM.format(category_help=CATEGORY_HELP),
    render_user=_render_extract_user,
    response_schema=LLMLimitationList,
    max_tokens=1500,
    temperature=0.2,
    description="Tier A: extract candidate limitations from one (paper, source) pair.",
)


# ---------------------------------------------------------------------------
# 2. Dedup clustering
# ---------------------------------------------------------------------------

_DEDUP_SYSTEM = """\
You are clustering candidate limitations of a scientific paper into equivalent groups.

Two limitations are "equivalent" if they refer to the same underlying issue, even if phrased differently.
Two limitations are NOT equivalent if they concern different aspects, methods, or assumptions.

Output a list of groups. Each group has:
  - representative_description: a single merged, clear description
  - member_ids: list of input limitation IDs in this group
  - category: the most appropriate category for the merged group
  - severity: the highest severity across members (major > minor)

Every input ID must appear in exactly one group.
"""


def _render_dedup_user(input_listing: str) -> str:
    return f"Input limitations:\n\n{input_listing}\n\nCluster them."


Prompt(
    name="dedup_limitations_v1",
    version="1.0",
    system=_DEDUP_SYSTEM,
    render_user=_render_dedup_user,
    response_schema=DedupResult,
    max_tokens=3000,
    temperature=0.0,
    description="Cluster near-equivalent candidate limitations after Tier A extraction.",
)


# ---------------------------------------------------------------------------
# 3. Skeptic argue / paper respond / skeptic revise / moderator
# ---------------------------------------------------------------------------

_SKEPTIC_SYSTEM_TEMPLATE = """\
{prior}

You are rigorous but fair. You only raise a concern if you can ground it in the paper text.
Never manufacture a quote. If the paper contradicts your concern, say so.
Be specific and terse.

{retrieved_context_block}

Paper context:
{paper_context}
"""

SKEPTIC_PRIORS = {
    "scope": """\
You are a Scope Skeptic. You look for mismatches between the paper's *claims* and what it *actually tested*.
Probe for:
  - Claims of generality ("our method works on any domain") that are supported only by benchmarks in one domain.
  - Claims of scaling ("scales to millions of X") supported only at smaller sizes.
  - Cross-lingual / cross-task claims tested only on English / a single task.
  - "Zero-shot" claims that rely on pretraining data the paper does not audit.
  - Implicit data distribution assumptions that do not match real deployment.""",
    "methodology": """\
You are a Methodology Skeptic. You look for holes in the experimental setup that bias the comparison.
Probe for:
  - Baselines tuned less carefully than the proposed method.
  - Missing obvious baselines (e.g., a simpler method that might match performance).
  - Evaluation metrics that favor the proposed method by construction.
  - Data leakage: test-set overlap with train or pretraining.
  - Ablation studies that add instead of remove (inflating perceived contribution).
  - Single-dataset claims where multi-dataset evidence is standard in the field.""",
    "reproducibility": """\
You are a Reproducibility Skeptic. You assess whether the paper can be faithfully reproduced.
Probe for:
  - Missing or vague hyperparameters (learning rate, batch size, seed, schedule).
  - Single-seed results where variance matters.
  - Undisclosed data preprocessing choices.
  - Unavailable or undocumented training data.
  - Hardware / random-init sensitivity not characterized.
  - Evaluation code not released.""",
    "fairness": """\
You are a Fairness Skeptic. You look for subgroup gaps and demographic blind spots.
Probe for:
  - Aggregate metrics that hide disparities across race / gender / age / language.
  - Training data with known demographic skew, with no audit.
  - Benchmarks known to underrepresent minority groups.
  - Claims of fairness based on a single fairness metric.
  - Downstream applications with disparate-impact risk that the paper does not discuss.""",
    "theoretical": """\
You are a Theoretical Skeptic. You check the soundness of proofs, definitions, and assumptions.
Probe for:
  - Assumptions stated in the theorem that fail in the experimental setting.
  - Definitions that drift between the formal and experimental sections.
  - Circular reasoning (the metric is designed to reward the method).
  - Proofs that skip "we can easily show" non-obvious steps.
  - Reductions to prior work that change the setting.""",
}


def build_skeptic_system(category: str, paper_context: str, retrieved_context: str = "") -> str:
    if retrieved_context:
        block = f"""RETRIEVED EVIDENCE (category-relevant segments from the paper — focus your concern here):
{retrieved_context}

---
"""
    else:
        block = ""
    return _SKEPTIC_SYSTEM_TEMPLATE.format(
        prior=SKEPTIC_PRIORS[category],
        paper_context=paper_context,
        retrieved_context_block=block,
    )


def _render_skeptic_argue_user(topic: str, topic_description: str) -> str:
    return f"""\
The topic of this debate node is: "{topic}".
Topic description: {topic_description}

Propose ONE specific concern under this topic. Ground it in a specific quote from the paper.
If you cannot find a concrete concern, return a very short description and leave evidence_quote empty; the moderator will terminate this branch.

# OUTPUT FORMAT — CRITICAL

Output ONLY a JSON object, nothing else. No prose, no markdown code fences, no commentary.
Your ENTIRE response must start with "{{" and end with "}}".

Use EXACTLY these field names:

{{
  "topic": "<1-sentence concern title>",
  "description": "<the concern, 2-4 sentences>",
  "evidence_quote": "<exact quote from the paper the concern hinges on; empty string if none>",
  "evidence_section": "<section the quote is from; empty string if unknown>",
  "severity_guess": "<one of: minor | major>"
}}
"""


Prompt(
    name="skeptic_argue_v1",
    version="1.0",
    system="",   # system built dynamically from skeptic category + paper context
    render_user=_render_skeptic_argue_user,
    response_schema=SkepticArgument,
    max_tokens=600,
    temperature=0.3,
    description="Skeptic proposes a specific concern at a debate node.",
)


_PAPER_SYSTEM_TEMPLATE = """\
You are the authors of the paper below. You will be asked to respond to a critical concern raised by a skeptical reviewer.

Rules:
  - You MUST ground your response in the actual content of the paper, quoting where possible.
  - If the paper genuinely does not address the concern, acknowledge the limitation honestly. Do not fabricate.
  - Be specific and terse. Cite section names where relevant.

PAPER:
{paper_context}
"""


def build_paper_system(paper_context: str) -> str:
    return _PAPER_SYSTEM_TEMPLATE.format(paper_context=paper_context)


def _render_paper_respond_user(
    topic: str, description: str, evidence_quote: str, evidence_section: str
) -> str:
    return f"""\
A skeptical reviewer has raised a concern about your paper:
  Concern topic: "{topic}"
  Concern description: {description}
  Evidence quoted from your paper: "{evidence_quote}" (section: {evidence_section})

Respond. You may acknowledge (if the concern is valid), rebut with a specific counter-quote from the paper, or clarify.

# OUTPUT FORMAT — CRITICAL

Output ONLY a JSON object, nothing else. No prose, no markdown code fences, no commentary.
Your ENTIRE response must start with "{{" and end with "}}".

Use EXACTLY these field names:

{{
  "acknowledges": <true or false; true if the paper's authors would concede this concern>,
  "response": "<the authors' rebuttal or acknowledgment, 2-4 sentences>",
  "citation_quote": "<exact quote from the paper that rebuts (or acknowledges) the concern; empty string if none>"
}}
"""


Prompt(
    name="paper_respond_v1",
    version="1.0",
    system="",
    render_user=_render_paper_respond_user,
    response_schema=PaperResponse,
    max_tokens=500,
    temperature=0.2,
    description="Paper persona responds to a skeptic's concern.",
)


def _render_skeptic_revise_user(
    original_description: str,
    evidence_quote: str,
    paper_response: str,
    acknowledges: bool,
    citation_quote: str,
) -> str:
    return f"""\
You raised: {original_description}
Evidence quoted: "{evidence_quote}"

The paper responded:
  response: {paper_response}
  acknowledges: {acknowledges}
  counter-quote: "{citation_quote}"

Revise your concern. If the paper's response genuinely addresses it, concede. Otherwise, sharpen the concern focusing on what remains unaddressed.

# OUTPUT FORMAT — CRITICAL

Output ONLY a JSON object, nothing else. No prose, no markdown code fences, no commentary.
Your ENTIRE response must start with "{{" and end with "}}".

Use EXACTLY these field names:

{{
  "revised_description": "<the concern, revised in light of the response>",
  "concedes": <true or false; true if you retract the concern after the response>
}}
"""


Prompt(
    name="skeptic_revise_v1",
    version="1.0",
    system="",
    render_user=_render_skeptic_revise_user,
    response_schema=SkepticRevision,
    max_tokens=400,
    temperature=0.2,
    description="Skeptic revises or concedes after paper's response.",
)


_MODERATOR_SYSTEM = """\
You are a neutral moderator judging a debate between a skeptic and a paper's authors.
Rule valid only if the concern is concrete, grounded in the paper, and the paper did not adequately address it.
Rule deflected if the paper credibly addresses the concern.
Rule unclear if there is not enough information.
"""


def _render_moderator_user(
    topic: str, category: str,
    argument_description: str, evidence_quote: str,
    paper_response: str, citation_quote: str, acknowledges: bool,
    revised_description: str, concedes: bool,
    max_children: int,
) -> str:
    return f"""\
Topic: {topic}
Category: {category}

Skeptic initial concern: {argument_description}
Evidence quoted: "{evidence_quote}"

Paper's response: {paper_response}
Paper's counter-quote: "{citation_quote}"
Paper acknowledges concern: {acknowledges}

Skeptic's revision: {revised_description}
Skeptic concedes: {concedes}

Rule on the debate. Decide whether to expand (at most {max_children} sub-concerns) if the concern is valid and has deeper sub-aspects that remain unclear.

# OUTPUT FORMAT — CRITICAL

Output ONLY a JSON object, nothing else. No prose, no markdown code fences, no commentary.
Your ENTIRE response must start with "{{" and end with "}}".

Use EXACTLY these field names:

{{
  "verdict": "<one of: valid | deflected | unclear>",
  "severity": "<one of: minor | major>",
  "should_expand": <true or false>,
  "expansion_prompts": ["<sub-concern topic>", ...],
  "reasoning": "<1-3 sentence rationale>"
}}

If should_expand is false, expansion_prompts should be an empty list [].
"""


Prompt(
    name="moderator_verdict_v1",
    version="1.0",
    system=_MODERATOR_SYSTEM,
    render_user=_render_moderator_user,
    response_schema=ModeratorVerdict,
    max_tokens=500,
    temperature=0.1,
    description="Moderator rules on a debate node and decides expansion.",
)


# ---------------------------------------------------------------------------
# 5. Evaluation: limitation semantic match
# ---------------------------------------------------------------------------

_MATCH_SYSTEM = """\
You are judging whether two limitations of a scientific paper refer to the same underlying concern.
Consider them the same if they point to the same weakness, even if worded differently (e.g., "missing ablation on X" and "ablation study does not isolate X").
Consider them different if they concern different aspects, methods, or assumptions.
Be conservative: if unsure, answer False.
"""


def _render_match_user(a: str, b: str) -> str:
    return f"Limitation A: {a}\n\nLimitation B: {b}\n\nSame concern?"


Prompt(
    name="match_limitations_v1",
    version="1.0",
    system=_MATCH_SYSTEM,
    render_user=_render_match_user,
    response_schema=MatchResult,
    max_tokens=200,
    temperature=0.0,
    description="Evaluation: LLM judge for matching two limitations.",
)


# ---------------------------------------------------------------------------
# 6. Panel Review (cross-skeptic adjudication — NOVEL beyond ToD)
# ---------------------------------------------------------------------------

_PANEL_SYSTEM = """\
You are a review PANEL representing all 5 skeptic perspectives on a claim about a scientific paper.

A category-specific skeptic has produced a claimed unstated limitation of paper P, which passed through a debate with the paper's authors and a moderator has ruled it "valid". Your job is to cross-review:

  1. Is the claim actually in the right category? A claim framed as "methodology" might really be theoretical (the assumption is unjustified in the proof) or scope (the evaluation was narrow). Reassign if needed.

  2. Is it subsumed by a concern that would be raised by a different skeptic? E.g., a methodology claim that's really about data leakage might duplicate what a reproducibility skeptic would raise.

  3. Is the severity calibrated correctly from the panel's wider perspective? Sometimes a category-specific skeptic overstates; sometimes understates.

  4. Is the claim substantive at all, or should the panel REJECT it as a false positive (e.g., the "evidence" doesn't actually support the concern; or the authors DID address it elsewhere in the paper)?

The 5 perspectives:
  - Scope: "does the paper over-claim generalization?"
  - Methodology: "are experimental comparisons fair and controls sound?"
  - Reproducibility: "can this be reproduced from what's reported?"
  - Fairness: "does it mask subgroup disparities?"
  - Theoretical: "are the assumptions and proofs sound?"

Output ONE verdict:
  - "endorse":     keep as-is
  - "reclassify":  keep but change final_category (e.g., methodology→theoretical)
  - "downgrade":   keep but final_severity=minor
  - "merge":       this overlaps materially with multiple categories; list them in cross_category_concerns
  - "reject":      the claim is a false positive; drop from benchmark

# OUTPUT FORMAT — CRITICAL

Output ONLY a JSON object with EXACTLY this structure:

{
  "verdict": "endorse | reclassify | downgrade | merge | reject",
  "final_category": "scope | methodology | reproducibility | fairness | theoretical",
  "final_severity": "minor | major",
  "cross_category_concerns": ["<category name>", ...],
  "reasoning": "<1-3 sentence rationale>"
}

No prose before or after. Start with { and end with }.
"""


def _render_panel_user(
    topic: str,
    category: str,
    severity: str,
    description: str,
    evidence_quote: str,
    paper_response: str,
    moderator_reasoning: str,
) -> str:
    return f"""\
CLAIM UNDER REVIEW:
  Topic: {topic}
  Category (as claimed by the specialist skeptic): {category}
  Severity (as claimed): {severity}

  Description of the unstated limitation:
  {description}

  Evidence quote from the paper:
  "{evidence_quote}"

  Paper authors' response during debate:
  {paper_response}

  Moderator's rationale for ruling valid:
  {moderator_reasoning}

Now: the panel reviews this claim. Output the JSON only.
"""


Prompt(
    name="panel_review_v1",
    version="1.0",
    system=_PANEL_SYSTEM,
    render_user=_render_panel_user,
    response_schema=PanelReview,
    max_tokens=500,
    temperature=0.1,
    description=(
        "Cross-skeptic panel review after the per-category moderator rules valid. "
        "Can reclassify, downgrade, merge, or reject the claim. Novel over ToD."
    ),
)


# ---------------------------------------------------------------------------
# 7. Baseline: zero-shot limitation listing (for paper comparison)
# ---------------------------------------------------------------------------

_BASELINE_ZS_SYSTEM = """\
You are an expert research scientist reading a scientific paper.
Your task: list the paper's **unstated limitations** — weaknesses the authors did NOT themselves acknowledge.

CRITERIA for a valid limitation:
  1. Specific and concrete (not "has limitations").
  2. Grounded in the paper text (cite a supporting quote when possible).
  3. NOT already addressed in the paper's own Limitations / Future Work section.
  4. Categorized into one of five categories.

{category_help}

# OUTPUT FORMAT — CRITICAL

You must output ONLY a single JSON object, nothing else. No prose, no analysis, no markdown code fences, no explanations. Your ENTIRE response must start with "{{" and end with "}}".

The JSON object has EXACTLY this structure (use these EXACT field names):

{{
  "limitations": [
    {{
      "description": "<1-2 sentence description of the unstated limitation>",
      "category": "<one of: scope | methodology | reproducibility | fairness | theoretical>",
      "severity": "<one of: minor | major>",
      "evidence_quote_from_paper": "<exact quote from the paper that hints at this limitation; empty string if none>",
      "evidence_section": "<section of the paper where the quote appears; empty string if unknown>",
      "source_quote": "",
      "confidence": <float 0.0-1.0>
    }}
  ],
  "reasoning": "<1-2 sentence rationale>"
}}

RULES:
- Use the EXACT field names above.
- "limitations" is a list; output up to 8 of the most important unstated limitations.
- Every value is required. Use empty string "" where no evidence is available.
- "source_quote" must be "" (no external source in this baseline).
- Output JSON and nothing else.
"""


def _render_baseline_zs_user(paper_context: str) -> str:
    return f"""\
PAPER:
{paper_context}

---

List the most important unstated limitations of this paper. Be specific, concrete, and grounded in the paper text.
"""


Prompt(
    name="baseline_zero_shot_v1",
    version="1.0",
    system=_BASELINE_ZS_SYSTEM.format(category_help=CATEGORY_HELP),
    render_user=_render_baseline_zs_user,
    response_schema=LLMLimitationList,
    max_tokens=2000,
    temperature=0.2,
    description="Baseline: zero-shot LLM prompt to list a paper's unstated limitations (no debate, no personas).",
)


# ---------------------------------------------------------------------------
# 8. Baseline: single-skeptic chain-of-thought
# ---------------------------------------------------------------------------

_BASELINE_SS_SYSTEM = """\
You are a senior research scientist playing the role of an EXPERT SKEPTIC reviewer of a scientific paper.
You have deep expertise across all aspects of empirical and theoretical research: study design, baselines, evaluation, reproducibility, fairness, theoretical assumptions, and scope of claims.

Your job: identify the paper's **unstated limitations** — weaknesses the authors did NOT themselves acknowledge.

Work step by step in your head:
  1) Identify what the paper CLAIMS.
  2) Identify what the paper ACKNOWLEDGES as a limitation.
  3) Find gaps between claims and evidence that the paper does NOT admit.
  4) For each such gap, write a specific, concrete concern.

CRITERIA for a valid limitation:
  1. Specific and concrete (not generic "has limitations").
  2. Grounded in the paper text.
  3. NOT already addressed in the paper's own Limitations / Future Work section.
  4. Categorized into one of five categories.

{category_help}

# OUTPUT FORMAT — CRITICAL

You must output ONLY a single JSON object, nothing else. No prose, no analysis, no markdown code fences. Your ENTIRE response must start with "{{" and end with "}}".

The JSON object has EXACTLY this structure (use these EXACT field names):

{{
  "limitations": [
    {{
      "description": "<1-2 sentence description of the unstated limitation>",
      "category": "<one of: scope | methodology | reproducibility | fairness | theoretical>",
      "severity": "<one of: minor | major>",
      "evidence_quote_from_paper": "<exact quote from the paper supporting this concern; empty string if none>",
      "evidence_section": "<section where the quote appears; empty string if unknown>",
      "source_quote": "",
      "confidence": <float 0.0-1.0>
    }}
  ],
  "reasoning": "<2-4 sentence rationale — your step-by-step thinking leading to the concerns>"
}}

RULES:
- Use the EXACT field names above.
- "limitations" is a list; output up to 10 of the most important unstated limitations.
- Every value is required. Use empty string "" where no evidence is available.
- "source_quote" must be "" (no external source in this baseline).
- Output JSON and nothing else.
"""


def _render_baseline_ss_user(paper_context: str) -> str:
    return f"""\
PAPER:
{paper_context}

---

You are reviewing this paper as an expert skeptic. Think step by step through the paper's claims and identify concrete unstated limitations across methodology, scope, theory, reproducibility, and fairness. Output the JSON object described above.
"""


Prompt(
    name="baseline_single_skeptic_v1",
    version="1.0",
    system=_BASELINE_SS_SYSTEM.format(category_help=CATEGORY_HELP),
    render_user=_render_baseline_ss_user,
    response_schema=LLMLimitationList,
    max_tokens=2500,
    temperature=0.3,
    description="Baseline: single generalist skeptic with chain-of-thought reasoning (no forest, no panel).",
)


if __name__ == "__main__":
    for name, p in all_prompts().items():
        print(f"{name:30s}  v{p.version}  schema={p.response_schema.__name__ if p.response_schema else '-'}")
        print(f"    {p.description}")
