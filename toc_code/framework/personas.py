"""
Framework: Paper persona + 5 specialized Skeptic personas.

Prompts (system prompts per role, specialized priors per category) are defined
in `llm.prompts` and assembled here per-paper at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm.client import LLMClient
from llm.prompts import build_paper_system, build_skeptic_system, SKEPTIC_PRIORS
from schema.models import LimitationCategory, Paper


# ---------------------------------------------------------------------------
# Personas (thin wrappers over prompt builders + LLM client)
# ---------------------------------------------------------------------------

@dataclass
class Skeptic:
    """An LLM-backed skeptic persona for one category."""
    category: LimitationCategory
    llm: LLMClient
    name: Optional[str] = None

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.category.value.capitalize()}Skeptic"

    def system_prompt(self, paper: Paper, paper_context_chars: int = 20000) -> str:
        return build_skeptic_system(
            category=self.category.value,
            paper_context=paper.as_context(max_chars=paper_context_chars),
        )

    @property
    def prior(self) -> str:
        return SKEPTIC_PRIORS[self.category.value]


@dataclass
class PaperPersona:
    """An LLM-backed persona that defends the paper against skeptic attacks."""
    paper: Paper
    llm: LLMClient
    name: str = "Paper"

    def system_prompt(self, paper_context_chars: int = 30000) -> str:
        return build_paper_system(self.paper.as_context(max_chars=paper_context_chars))


def make_skeptic_bench(llm: LLMClient) -> list[Skeptic]:
    """Return one Skeptic of each category, all sharing the same LLM client."""
    return [Skeptic(category=c, llm=llm) for c in LimitationCategory]


if __name__ == "__main__":
    llm = LLMClient.from_env()
    bench = make_skeptic_bench(llm)
    print(f"Loaded {len(bench)} skeptics:")
    for s in bench:
        print(f"  - {s.name}  ({len(s.prior)} chars of prior)")
