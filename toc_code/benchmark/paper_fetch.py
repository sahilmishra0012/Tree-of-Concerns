"""
Paper fetching utilities.

- Fetch paper metadata and (where possible) full text from arXiv.
- Fetch citing papers from Semantic Scholar.
- Parse LaTeX-exported or ar5iv HTML into Paper + PaperSection objects.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

import requests

from schema.models import Paper, PaperSection

log = logging.getLogger(__name__)


ARXIV_ABS_URL = "https://arxiv.org/abs/{id}"
AR5IV_HTML_URL = "https://ar5iv.labs.arxiv.org/html/{id}"
SEMSCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"


# ---------------------------------------------------------------------------
# arXiv fetch (metadata + full text via ar5iv HTML)
# ---------------------------------------------------------------------------

def fetch_arxiv_paper(arxiv_id: str, with_full_text: bool = True) -> Paper:
    """
    Given an arXiv id (e.g., "2109.10500"), return a Paper with metadata and
    section-parsed full text (via ar5iv HTML).
    """
    import arxiv as arxiv_api

    # Metadata via arxiv API
    search = arxiv_api.Search(id_list=[arxiv_id])
    result = next(iter(arxiv_api.Client().results(search)))

    paper = Paper(
        id=f"arxiv:{arxiv_id}",
        title=result.title.strip(),
        abstract=result.summary.strip(),
        authors=[a.name for a in result.authors],
        year=result.published.year if result.published else None,
        venue=None,
        arxiv_id=arxiv_id,
    )

    if with_full_text:
        sections, full_text = _fetch_ar5iv_sections(arxiv_id)
        if sections:
            paper.sections = sections
            paper.full_text = full_text

    return paper


def _fetch_ar5iv_sections(arxiv_id: str) -> tuple[list[PaperSection], str]:
    """Best-effort parse of ar5iv HTML into sections."""
    url = AR5IV_HTML_URL.format(id=arxiv_id)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        log.warning("ar5iv fetch failed for %s: %s", arxiv_id, e)
        return [], ""

    html = r.text
    # Strip HTML tags very coarsely. For a real paper, this keeps enough
    # structure to find sections while dropping math notation we don't need.
    # We keep <h1>, <h2>, <h3> as markers, then strip the rest.
    # Simpler: use the pre-formatted MathJax-less text inside <p> and heading tags.

    # Replace section headings with markers we can split on.
    heading_pattern = re.compile(
        r"<(?P<tag>h[1-3])[^>]*>\s*(?:<[^>]+>)*\s*(?P<title>.*?)(?:<[^>]+>)*\s*</(?P=tag)>",
        re.DOTALL | re.IGNORECASE,
    )
    replaced_html = heading_pattern.sub(
        lambda m: f"\n\n§§SECTION§§ {_strip_html(m.group('title')).strip()}\n\n",
        html,
    )
    text = _strip_html(replaced_html)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    parts = text.split("§§SECTION§§")
    # First part is pre-first-heading content (title/abstract/authors);
    # we discard it here since Paper.abstract/title already come from arxiv API.
    sections: list[PaperSection] = []
    for i, chunk in enumerate(parts[1:], start=1):
        lines = [l.strip() for l in chunk.strip().split("\n", 1)]
        title = lines[0] if lines else f"Section {i}"
        body = lines[1].strip() if len(lines) > 1 else ""
        if not body:
            continue
        sections.append(PaperSection(title=title[:200], text=body, order=i))

    full_text = "\n\n".join(f"## {s.title}\n\n{s.text}" for s in sections)
    return sections, full_text


def _strip_html(s: str) -> str:
    """Very coarse HTML -> text."""
    s = re.sub(r"<script[^>]*>.*?</script>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<style[^>]*>.*?</style>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    # Decode common entities
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#x27;", "'")
    return s


# ---------------------------------------------------------------------------
# Semantic Scholar: find papers that cite a given paper
# ---------------------------------------------------------------------------

def fetch_citing_papers(
    arxiv_id: str,
    max_results: int = 50,
    sleep_s: float = 0.5,
) -> list[dict]:
    """
    Return a list of papers that cite the given arXiv paper.
    Each dict has: paperId, title, abstract, year, authors, venue,
    externalIds (arxiv, doi), contexts (sentences in the citing paper
    surrounding the citation).

    Uses Semantic Scholar Graph API (public, rate-limited).
    """
    paper_id = f"ARXIV:{arxiv_id}"
    citing = []
    offset = 0
    limit = min(100, max_results)

    while len(citing) < max_results:
        url = f"{SEMSCHOLAR_BASE}/paper/{paper_id}/citations"
        params = {
            "fields": "paperId,title,abstract,year,authors,externalIds,venue,contexts",
            "offset": offset,
            "limit": limit,
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                log.warning("Rate limited; sleeping 10s")
                time.sleep(10)
                continue
            r.raise_for_status()
        except Exception as e:
            log.warning("Semantic Scholar citing-fetch failed: %s", e)
            break

        data = r.json()
        items = data.get("data", [])
        for item in items:
            citing_paper = item.get("citingPaper") or {}
            contexts = item.get("contexts", [])
            citing_paper["contexts"] = contexts
            citing.append(citing_paper)
        if len(items) < limit:
            break
        offset += limit
        time.sleep(sleep_s)
        if len(citing) >= max_results:
            break

    return citing[:max_results]


def fetch_paper_by_ids(
    paper_id: str,
    fields: str = "title,abstract,externalIds,year,authors,venue",
) -> Optional[dict]:
    """Fetch paper metadata by Semantic Scholar id or other identifier."""
    url = f"{SEMSCHOLAR_BASE}/paper/{paper_id}"
    try:
        r = requests.get(url, params={"fields": fields}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning("Semantic Scholar paper fetch failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Critique-sentence heuristic filter
# ---------------------------------------------------------------------------

_CRITIQUE_CUES = [
    r"\bunlike\b",
    r"\bin contrast\b",
    r"\bhowever\b",
    r"\blimitation\b",
    r"\bfails to\b",
    r"\bdoes not\b",
    r"\bcannot\b",
    r"\bassume[sd]?\b",
    r"\bignore[sd]?\b",
    r"\black[s]?\b",
    r"\bweakness\b",
    r"\bover(claim|stat|simplif)",
    r"\bmissing\b",
    r"\bbias\b",
]
_CRITIQUE_RE = re.compile("|".join(_CRITIQUE_CUES), re.IGNORECASE)


def filter_critique_contexts(contexts: list[str], target_hint: Optional[str] = None) -> list[str]:
    """
    Given citation contexts (sentences) from Semantic Scholar, return those
    that look like critiques. Optionally filter further by a target hint
    (e.g., a substring that must appear near the citation).
    """
    out = []
    for c in contexts:
        if not _CRITIQUE_RE.search(c):
            continue
        if target_hint and target_hint.lower() not in c.lower():
            continue
        out.append(c)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = fetch_arxiv_paper("2109.10500", with_full_text=True)
    print(f"TITLE: {p.title}")
    print(f"AUTHORS: {p.authors}")
    print(f"YEAR: {p.year}")
    print(f"SECTIONS ({len(p.sections)}):")
    for s in p.sections[:5]:
        print(f"  - {s.title}  ({len(s.text)} chars)")
    print()
    print("Fetching first 10 citing papers...")
    citing = fetch_citing_papers("2109.10500", max_results=10)
    print(f"Got {len(citing)} citations")
    for c in citing[:3]:
        print(f"  - {c.get('title')}  ({c.get('year')})")
        for ctx in c.get("contexts", [])[:2]:
            print(f"    ctx: {ctx[:200]}")
