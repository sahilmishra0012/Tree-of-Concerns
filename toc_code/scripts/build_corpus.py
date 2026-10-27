"""
Corpus construction for LimitTree.

Pipeline:
  Phase 1 (discover): query Semantic Scholar bulk search for well-cited ML/NLP
    papers 2020-2024 that have an arXiv ID.
  Phase 2 (fetch): for each candidate, pull:
      - arXiv metadata
      - ar5iv full text
      - citing papers via Semantic Scholar (abstract + citation contexts)
    filter citing contexts for critique-like sentences.
  Phase 3 (filter): keep papers with >= MIN_CRITIQUE contexts across citers.

Output directory layout:
  corpus_root/
    manifest.json                      # summary of the run
    seeds.json                         # all discovered seeds
    papers/
      arxiv_XXXX.XXXXX/
        paper.json                     # schema.Paper object
        citing_papers.json             # raw list of citing paper dicts
        critique_contexts.json         # filtered critique sentences per source
        status.json                    # {"fetched": true, "n_critique": X, ...}

Resumable: if a paper's status.json says fetched=true, skip it.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

# Make sure we can run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.limitations_section import extract_stated_limitations
from benchmark.openreview_fetch import fetch_openreview_bundle
from benchmark.paper_fetch import (
    AR5IV_HTML_URL,
    SEMSCHOLAR_BASE,
    _fetch_ar5iv_sections,
    filter_critique_contexts,
)
from schema.models import Paper, PaperSection

log = logging.getLogger("build_corpus")


SS_HEADERS = {"User-Agent": "limit-tree/0.1 (research)"}


# ---------------------------------------------------------------------------
# Phase 1: discovery
# ---------------------------------------------------------------------------

# Broad topical queries (Semantic Scholar `query` is full-text; must be non-empty).
# We combine each with venue filters to narrow down.
_BROAD_QUERIES = [
    "neural network",
    "language model",
    "transformer",
    "deep learning",
    "reinforcement learning",
    "representation learning",
    "graph neural network",
    "pretraining",
    "fine-tuning",
    "diffusion model",
    "contrastive learning",
    "attention mechanism",
    "self-supervised",
    "retrieval augmented",
    "prompt engineering",
]

# Target venues — we use Semantic Scholar's `venue` filter (separate from `query`).
TARGET_VENUES_FILTER = [
    "Neural Information Processing Systems",
    "International Conference on Machine Learning",
    "International Conference on Learning Representations",
    "Annual Meeting of the Association for Computational Linguistics",
    "Conference on Empirical Methods in Natural Language Processing",
    "North American Chapter of the Association for Computational Linguistics",
    "Conference of the European Chapter of the Association for Computational Linguistics",
    "Conference on Computer Vision and Pattern Recognition",
    "International Conference on Computer Vision",
    "European Conference on Computer Vision",
    "Transactions of the Association for Computational Linguistics",
    "Findings of the Association for Computational Linguistics",
    "AAAI Conference on Artificial Intelligence",
    "International Joint Conference on Artificial Intelligence",
]


@dataclass
class SeedPaper:
    semscholar_id: str
    arxiv_id: Optional[str]
    title: str
    year: Optional[int]
    venue: Optional[str]
    citation_count: int
    fields_of_study: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "semscholar_id": self.semscholar_id,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "year": self.year,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "fields_of_study": self.fields_of_study,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SeedPaper":
        return cls(**d)


def _ss_get(url: str, params: dict, max_retries: int = 6) -> dict:
    """GET with retry + backoff on 429 / 5xx."""
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=SS_HEADERS, timeout=30)
        except Exception as e:
            log.warning("SS network error %s; retrying in %.0fs", e, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            log.warning("SS %s: retrying in %.0fs", r.status_code, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"SS bulk search giving up on {url} after {max_retries} retries")


def discover_seeds(
    n_target: int = 500,
    year_range: str = "2020-2024",
    min_citations: int = 100,
    throttle_s: float = 1.0,
    queries: Optional[list[str]] = None,
    venues: Optional[list[str]] = None,
) -> list[SeedPaper]:
    """
    Use Semantic Scholar bulk search to assemble seed candidates with
    arXiv IDs. We cross {broad topical queries} × {venue filters} to get
    diverse, well-cited papers from ML/NLP/CV.
    """
    out: dict[str, SeedPaper] = {}

    queries = queries or _BROAD_QUERIES
    venues = venues or TARGET_VENUES_FILTER
    venues_csv = ",".join(venues)

    for query_text in queries:
        if len(out) >= n_target:
            break
        log.info("Querying term=%r across venues ...", query_text)
        token: Optional[str] = None
        page = 0
        while len(out) < n_target:
            params = {
                "query": query_text,
                "fields": "paperId,title,abstract,year,venue,externalIds,fieldsOfStudy,citationCount,authors",
                "year": year_range,
                "minCitationCount": min_citations,
                "limit": 100,
                "fieldsOfStudy": "Computer Science",
                "venue": venues_csv,
            }
            if token:
                params["token"] = token
            try:
                data = _ss_get(f"{SEMSCHOLAR_BASE}/paper/search/bulk", params)
            except Exception as e:
                log.warning("bulk search failed for %r: %s", query_text, e)
                break
            items = data.get("data", []) or []
            if not items:
                break
            added = 0
            for it in items:
                ext = it.get("externalIds") or {}
                arxiv_id = ext.get("ArXiv")
                if not arxiv_id:
                    continue
                sid = it["paperId"]
                if sid in out:
                    continue
                out[sid] = SeedPaper(
                    semscholar_id=sid,
                    arxiv_id=arxiv_id,
                    title=(it.get("title") or "").strip(),
                    year=it.get("year"),
                    venue=it.get("venue"),
                    citation_count=it.get("citationCount") or 0,
                    fields_of_study=it.get("fieldsOfStudy") or [],
                )
                added += 1
            page += 1
            token = data.get("token")
            log.info("  query=%r page=%d  new_total=%d  added_this_page=%d",
                     query_text, page, len(out), added)
            if not token:
                break
            if page >= 20:   # safety cap per query
                break
            time.sleep(throttle_s)

    seeds = list(out.values())
    random.shuffle(seeds)
    # Sort by citation count (desc) so we prefer well-cited papers
    seeds.sort(key=lambda s: s.citation_count, reverse=True)
    return seeds[:n_target]


# ---------------------------------------------------------------------------
# Phase 2: per-paper fetch
# ---------------------------------------------------------------------------

def fetch_arxiv_metadata(arxiv_id: str) -> Optional[dict]:
    """Fast arXiv metadata fetch via the query API. Returns None on failure."""
    import arxiv as arxiv_api
    try:
        search = arxiv_api.Search(id_list=[arxiv_id])
        result = next(iter(arxiv_api.Client(page_size=1, delay_seconds=3.0).results(search)))
        return {
            "title": result.title.strip(),
            "abstract": result.summary.strip(),
            "authors": [a.name for a in result.authors],
            "year": result.published.year if result.published else None,
        }
    except Exception as e:
        log.warning("arxiv metadata fetch failed for %s: %s", arxiv_id, e)
        return None


def fetch_citing_with_contexts(
    paper_id: str,
    max_citers: int = 60,
    throttle_s: float = 0.8,
) -> list[dict]:
    """
    paper_id: e.g. "ARXIV:2109.10500" or a SS paper id.
    Returns a list of dicts with at least: title, year, abstract, externalIds, contexts.
    """
    citing: list[dict] = []
    offset = 0
    limit = min(100, max_citers)
    while len(citing) < max_citers:
        url = f"{SEMSCHOLAR_BASE}/paper/{paper_id}/citations"
        params = {
            "fields": "paperId,title,abstract,year,authors,externalIds,venue,contexts",
            "offset": offset,
            "limit": limit,
        }
        try:
            data = _ss_get(url, params)
        except Exception as e:
            log.warning("citations fetch failed for %s: %s", paper_id, e)
            break
        items = data.get("data", [])
        if not items:
            break
        for item in items:
            cp = item.get("citingPaper") or {}
            cp["contexts"] = item.get("contexts", [])
            citing.append(cp)
        if len(items) < limit:
            break
        offset += limit
        time.sleep(throttle_s)
    return citing[:max_citers]


def build_critique_sources(citing_papers: list[dict]) -> list[dict]:
    """
    For each citing paper, extract the critique-like sentences. Return a list
    of sources that have at least one critique context.
    """
    out = []
    for cp in citing_papers:
        critique = filter_critique_contexts(cp.get("contexts") or [])
        if not critique:
            continue
        ext = cp.get("externalIds") or {}
        out.append({
            "paperId": cp.get("paperId"),
            "title": cp.get("title"),
            "year": cp.get("year"),
            "venue": cp.get("venue"),
            "arxiv_id": ext.get("ArXiv"),
            "doi": ext.get("DOI"),
            "abstract": cp.get("abstract"),
            "critique_contexts": critique,
            "all_contexts": cp.get("contexts") or [],
        })
    return out


def fetch_paper_bundle(
    seed: SeedPaper,
    paper_dir: Path,
    max_citers: int = 60,
    throttle_s: float = 0.8,
    fetch_full_text: bool = True,
) -> dict:
    """Fetch and save all data for one paper. Returns a status dict."""
    paper_dir.mkdir(parents=True, exist_ok=True)
    status_path = paper_dir / "status.json"

    # Resume: if already marked fetched, skip
    if status_path.exists():
        try:
            st = json.loads(status_path.read_text())
            if st.get("fetched"):
                return st
        except Exception:
            pass

    status = {
        "semscholar_id": seed.semscholar_id,
        "arxiv_id": seed.arxiv_id,
        "title": seed.title,
        "fetched": False,
        "n_citers": 0,
        "n_critique_sources": 0,
        "n_critique_contexts": 0,
        "has_full_text": False,
        "has_limitations_section": False,
        "n_stated_limitations": 0,
        "has_openreview": False,
        "n_openreview_reviews": 0,
        "n_openreview_weakness_points": 0,
    }

    # --- Paper metadata + full text ---
    meta = fetch_arxiv_metadata(seed.arxiv_id)
    if not meta:
        status["error"] = "arxiv_metadata_fetch_failed"
        status_path.write_text(json.dumps(status, indent=2))
        return status

    sections: list[PaperSection] = []
    full_text = ""
    if fetch_full_text:
        try:
            sections, full_text = _fetch_ar5iv_sections(seed.arxiv_id)
        except Exception as e:
            log.warning("ar5iv fetch failed for %s: %s", seed.arxiv_id, e)

    paper = Paper(
        id=f"arxiv:{seed.arxiv_id}",
        title=meta["title"],
        abstract=meta["abstract"],
        authors=meta["authors"],
        year=meta.get("year") or seed.year,
        venue=seed.venue,
        arxiv_id=seed.arxiv_id,
        sections=sections,
        full_text=full_text or None,
    )
    (paper_dir / "paper.json").write_text(paper.model_dump_json(indent=2))
    status["has_full_text"] = bool(sections)

    # --- Self-stated Limitations section ---
    stated = extract_stated_limitations(paper)
    (paper_dir / "stated_limitations.json").write_text(json.dumps({
        "has_strong_section": stated.has_strong_section,
        "section_titles_found": stated.section_titles_found,
        "limitations": [
            {"text": lim.text, "section_title": lim.section_title, "bullet": lim.bullet}
            for lim in stated.limitations
        ],
        "raw_section_text": stated.raw_section_text,
    }, indent=2))
    status["has_limitations_section"] = stated.has_strong_section
    status["n_stated_limitations"] = len(stated.limitations)

    # --- OpenReview (best effort; ICLR / NeurIPS / COLM only) ---
    or_bundle = None
    if paper.title:
        try:
            or_bundle = fetch_openreview_bundle(title=paper.title, arxiv_id=seed.arxiv_id)
        except Exception as e:
            log.warning("OpenReview fetch failed for %s: %s", seed.arxiv_id, e)
    if or_bundle is not None:
        (paper_dir / "openreview.json").write_text(
            json.dumps(or_bundle.to_dict(), indent=2, default=str)
        )
        status["has_openreview"] = True
        status["n_openreview_reviews"] = len(or_bundle.reviews)
        status["n_openreview_weakness_points"] = sum(
            len(r.weakness_points) for r in or_bundle.reviews
        )
    else:
        status["has_openreview"] = False
        status["n_openreview_reviews"] = 0
        status["n_openreview_weakness_points"] = 0

    # --- Citing papers + critique contexts ---
    citing = fetch_citing_with_contexts(
        f"ARXIV:{seed.arxiv_id}", max_citers=max_citers, throttle_s=throttle_s,
    )
    (paper_dir / "citing_papers.json").write_text(json.dumps(citing, indent=2, default=str))

    critique_sources = build_critique_sources(citing)
    (paper_dir / "critique_contexts.json").write_text(
        json.dumps(critique_sources, indent=2, default=str)
    )

    status.update({
        "fetched": True,
        "n_citers": len(citing),
        "n_critique_sources": len(critique_sources),
        "n_critique_contexts": sum(len(s["critique_contexts"]) for s in critique_sources),
    })
    status_path.write_text(json.dumps(status, indent=2))
    return status


# ---------------------------------------------------------------------------
# Phase 3 + driver
# ---------------------------------------------------------------------------

def _safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def run_corpus(
    out_dir: Path,
    target_kept: int = 300,
    n_seeds: int = 500,
    year_range: str = "2020-2024",
    min_citations: int = 100,
    min_critique_contexts: int = 2,
    min_or_weaknesses: int = 3,
    max_citers: int = 60,
    throttle_s: float = 0.8,
    seeds_only: bool = False,
    resume: bool = True,
    max_fetch: Optional[int] = None,
    queries: Optional[list[str]] = None,
    venues: Optional[list[str]] = None,
    seeds_filename: str = "seeds.json",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    papers_dir = out_dir / "papers"
    papers_dir.mkdir(exist_ok=True)

    # Phase 1: seeds
    seeds_path = out_dir / seeds_filename
    if resume and seeds_path.exists():
        log.info("Loading cached seeds from %s", seeds_path)
        seeds = [SeedPaper.from_dict(d) for d in json.loads(seeds_path.read_text())]
    else:
        log.info("Discovering seeds (n_target=%d)...", n_seeds)
        seeds = discover_seeds(
            n_target=n_seeds,
            year_range=year_range,
            min_citations=min_citations,
            queries=queries,
            venues=venues,
        )
        seeds_path.write_text(json.dumps([s.to_dict() for s in seeds], indent=2))
        log.info("Wrote %d seeds to %s", len(seeds), seeds_path)

    if seeds_only:
        return {"n_seeds": len(seeds), "n_kept": 0, "papers_dir": str(papers_dir)}

    # Phase 2: fetch (resumable, stops once we have target_kept)
    log.info("Phase 2: fetching up to %d seeds (target kept=%d)...", len(seeds), target_kept)
    kept: list[dict] = []
    skipped_no_critique: list[dict] = []
    errors: list[dict] = []
    n_fetched_now = 0
    for i, seed in enumerate(seeds):
        if max_fetch is not None and n_fetched_now >= max_fetch:
            log.info("  reached max_fetch=%d", max_fetch)
            break
        if len(kept) >= target_kept:
            log.info("  reached target_kept=%d", target_kept)
            break
        pid = f"arxiv_{_safe_id(seed.arxiv_id)}"
        paper_dir = papers_dir / pid
        log.info("[%d/%d] %s  (%d citations)", i + 1, len(seeds), seed.arxiv_id, seed.citation_count)
        try:
            st = fetch_paper_bundle(
                seed, paper_dir,
                max_citers=max_citers,
                throttle_s=throttle_s,
            )
        except Exception as e:
            log.warning("  fetch failed: %s", e)
            errors.append({"arxiv_id": seed.arxiv_id, "error": str(e)})
            continue
        n_fetched_now += 1
        if not st.get("fetched"):
            errors.append(st)
            continue

        # Keep criterion:
        #   OpenReview-tier: has_openreview AND enough weakness points
        #   OR citing-paper tier: enough critique contexts
        n_or_weaknesses = st.get("n_openreview_weakness_points", 0)
        n_critique = st.get("n_critique_contexts", 0)
        if n_or_weaknesses >= min_or_weaknesses or n_critique >= min_critique_contexts:
            kept.append(st)
            or_flag = f"OR:{n_or_weaknesses}" if st.get("has_openreview") else ""
            log.info("    KEEP  critique=%d  %s", n_critique, or_flag)
        else:
            skipped_no_critique.append(st)
            log.info("    skip (OR=%d, critique=%d)", n_or_weaknesses, n_critique)

    manifest = {
        "target_kept": target_kept,
        "n_seeds_examined": i + 1 if seeds else 0,
        "n_fetched_this_run": n_fetched_now,
        "n_kept_total": len(kept),
        "n_skipped_no_critique": len(skipped_no_critique),
        "n_errors": len(errors),
        "settings": {
            "year_range": year_range,
            "min_citations": min_citations,
            "min_critique_contexts": min_critique_contexts,
            "max_citers": max_citers,
        },
        "kept_arxiv_ids": [k["arxiv_id"] for k in kept],
        "skipped_arxiv_ids": [k["arxiv_id"] for k in skipped_no_critique],
        "errors": errors,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("Done. Manifest: %s", out_dir / "manifest.json")
    return manifest


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="corpus root directory")
    ap.add_argument("--target_kept", type=int, default=300)
    ap.add_argument("--n_seeds", type=int, default=500)
    ap.add_argument("--year_range", type=str, default="2020-2024")
    ap.add_argument("--min_citations", type=int, default=100)
    ap.add_argument("--min_critique_contexts", type=int, default=2)
    ap.add_argument("--min_or_weaknesses", type=int, default=3,
                    help="keep if OpenReview has >= this many weakness points")
    ap.add_argument("--max_citers", type=int, default=60)
    ap.add_argument("--throttle_s", type=float, default=0.8)
    ap.add_argument("--seeds_only", action="store_true",
                    help="only discover seeds, no per-paper fetching")
    ap.add_argument("--max_fetch", type=int, default=None,
                    help="limit number of new papers fetched this run")
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()

    manifest = run_corpus(
        out_dir=Path(args.out),
        target_kept=args.target_kept,
        n_seeds=args.n_seeds,
        year_range=args.year_range,
        min_citations=args.min_citations,
        min_critique_contexts=args.min_critique_contexts,
        min_or_weaknesses=args.min_or_weaknesses,
        max_citers=args.max_citers,
        throttle_s=args.throttle_s,
        seeds_only=args.seeds_only,
        resume=not args.no_resume,
        max_fetch=args.max_fetch,
    )
    print("\n" + "=" * 70)
    print("CORPUS BUILD SUMMARY")
    print("=" * 70)
    for k, v in manifest.items():
        if isinstance(v, list):
            print(f"{k}: ({len(v)} items)")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
