"""
Extract the paper's OWN "Limitations" section (and "Discussion", "Broader Impacts",
"Future Work" when they contain acknowledged caveats).

This is the set of limitations the authors already STATED. For the LimitTree
benchmark, this set is used as:
  (a) negative/exclusion filter — an extracted unstated limitation is invalid
      if it paraphrases something in this section, and
  (b) ground-truth for "self-acknowledged limitations" analysis.

We rely on section titles parsed from ar5iv HTML. Matching is case-insensitive
and tolerant of ACL/NeurIPS conventions ("Limitations", "Limitations and
Future Work", "Broader Impacts", etc.).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from schema.models import Paper, PaperSection


# ---------------------------------------------------------------------------
# Section-title patterns
# ---------------------------------------------------------------------------

# Strong (almost certainly a limitations section)
_STRONG_TITLES = re.compile(
    r"\b("
    r"limitations?"
    r"|limitations\s+and\s+future\s+work"
    r"|limitations?,?\s+and\s+(future\s+work|broader\s+impact|ethical)"
    r"|discussion\s+of\s+limitations"
    r"|ethics?\s+statement"
    r"|known\s+limitations"
    r"|scope\s+and\s+limitations"
    r"|caveats"
    r")\b",
    re.IGNORECASE,
)

# Soft (may contain limitations; only used if paper has no strong section)
_SOFT_TITLES = re.compile(
    r"\b("
    r"discussion"
    r"|broader\s+impacts?"
    r"|future\s+work"
    r"|conclusion\s+and\s+future\s+work"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class StatedLimitation:
    """A single limitation explicitly stated by the authors."""
    text: str               # verbatim text of the statement
    section_title: str      # the section it came from
    bullet: bool = False    # True if this came from a bulleted list


@dataclass
class LimitationsSection:
    """All the self-stated limitations extracted from one paper."""
    has_strong_section: bool
    section_titles_found: list[str]
    limitations: list[StatedLimitation]
    raw_section_text: str   # concatenated section text (for debugging)

    @property
    def is_empty(self) -> bool:
        return not self.limitations


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _sections_matching(sections: list[PaperSection], pattern: re.Pattern) -> list[PaperSection]:
    return [s for s in sections if pattern.search(s.title)]


def _strip_section_number(title: str) -> str:
    """Strip leading section numbering like '5 Limitations', '5.1 Limitations', 'A.1 ...'."""
    return re.sub(r"^\s*(?:[A-Z]\.?\s*)?\d+(?:\.\d+)*\.?\s*", "", title).strip()


def _split_into_statements(text: str) -> list[tuple[str, bool]]:
    """
    Split the raw section text into statements.
    Returns list of (statement, from_bullet).
    """
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []

    out: list[tuple[str, bool]] = []

    # Look for bulleted markers. In ar5iv HTML text, bullets may appear as
    # "* ", "• ", "- ", or as markdown numbered lists. We detect them by
    # looking for patterns in the flowing text.
    bullet_regex = re.compile(r"(?<=[.!?])\s+(?:\*|•|-|\u2022)\s+|\s+(?:\*|•|-|\u2022)\s+(?=[A-Z])")
    if bullet_regex.search(text):
        parts = bullet_regex.split(text)
        for p in parts:
            p = p.strip(" .•*-\u2022")
            if p:
                out.append((p, True))
        if out:
            return out

    # Sentence split (avoid splitting abbreviations like "e.g.", "et al.", "Fig.")
    # Simple approach: split on ". " but only when followed by a capital letter.
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    for s in sents:
        s = s.strip()
        # Filter out trivial fragments
        if len(s) < 30:
            continue
        out.append((s, False))
    return out


def extract_stated_limitations(paper: Paper) -> LimitationsSection:
    """
    Return the paper's self-acknowledged limitations.

    Strategy:
      1. If any section title matches `_STRONG_TITLES`, use only those sections.
      2. Else, fall back to `_SOFT_TITLES`. (These may contain limitations
         but also other material; we still use them but flag `has_strong_section=False`.)
      3. Split the section text into statements (bulleted or sentences).
    """
    sections = paper.sections or []

    strong = _sections_matching(sections, _STRONG_TITLES)
    if strong:
        used = strong
        has_strong = True
    else:
        used = _sections_matching(sections, _SOFT_TITLES)
        has_strong = False

    if not used:
        return LimitationsSection(
            has_strong_section=False,
            section_titles_found=[],
            limitations=[],
            raw_section_text="",
        )

    raw_parts = []
    statements: list[StatedLimitation] = []
    for sec in used:
        title = _strip_section_number(sec.title)
        raw_parts.append(f"[{title}]\n{sec.text}")
        for stmt, is_bullet in _split_into_statements(sec.text):
            statements.append(StatedLimitation(
                text=stmt,
                section_title=title,
                bullet=is_bullet,
            ))

    return LimitationsSection(
        has_strong_section=has_strong,
        section_titles_found=[_strip_section_number(s.title) for s in used],
        limitations=statements,
        raw_section_text="\n\n".join(raw_parts),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json, logging, sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchmark.paper_fetch import fetch_arxiv_paper

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--arxiv", type=str, required=True)
    args = ap.parse_args()

    p = fetch_arxiv_paper(args.arxiv, with_full_text=True)
    print(f"Paper: {p.title}  ({len(p.sections)} sections)")
    extracted = extract_stated_limitations(p)
    print(f"\nStrong section found: {extracted.has_strong_section}")
    print(f"Section titles: {extracted.section_titles_found}")
    print(f"Extracted {len(extracted.limitations)} statements:")
    for i, lim in enumerate(extracted.limitations[:10]):
        bullet = "•" if lim.bullet else " "
        print(f"  {bullet} [{lim.section_title[:30]}] {lim.text[:200]}")
    if len(extracted.limitations) > 10:
        print(f"  ...and {len(extracted.limitations) - 10} more")
