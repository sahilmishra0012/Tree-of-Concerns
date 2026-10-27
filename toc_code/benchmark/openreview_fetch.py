"""
OpenReview fetcher.

Given an arXiv paper, try to locate the matching OpenReview submission and
fetch its reviews + author rebuttals. We use:
  - the OpenReview public API v2 (https://api2.openreview.net)
  - fallback to API v1 (https://api.openreview.net) for older venues

For each review we extract:
  - the reviewer's "weaknesses" bullet list (OpenReview review forms are
    structured, usually with fields like `summary`, `strengths`, `weaknesses`,
    `questions`, `rating`, `confidence`)
  - the author's rebuttal(s) that reply to that review
  - a light-weight "addressed" heuristic (did authors reply to this review?)

The LLM does the *fine-grained* per-weakness-point "addressed / unaddressed"
classification downstream; this module just delivers the raw structured data.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ReviewerWeakness:
    """One bullet/point from a reviewer's weaknesses field."""
    text: str
    bullet: bool = False   # True if this came from a bulleted list
    order: int = 0


@dataclass
class AuthorReply:
    """A reply from the authors (rebuttal / comment) to this review."""
    text: str
    timestamp: Optional[int] = None   # OpenReview cdate (epoch ms)
    signature: str = ""


@dataclass
class Review:
    reviewer: str                              # e.g., "Reviewer_abc1"
    rating: Optional[str] = None               # rating or score field
    confidence: Optional[str] = None
    summary: str = ""
    strengths: str = ""
    weaknesses: str = ""                       # full text as written
    questions: str = ""
    weakness_points: list[ReviewerWeakness] = field(default_factory=list)
    author_replies: list[AuthorReply] = field(default_factory=list)
    review_id: str = ""
    invitation: str = ""


@dataclass
class OpenReviewBundle:
    forum_id: str
    venue_id: str
    paper_title: str
    decision: Optional[str] = None
    reviews: list[Review] = field(default_factory=list)
    n_total_replies: int = 0

    def to_dict(self) -> dict:
        return {
            "forum_id": self.forum_id,
            "venue_id": self.venue_id,
            "paper_title": self.paper_title,
            "decision": self.decision,
            "n_reviews": len(self.reviews),
            "n_total_replies": self.n_total_replies,
            "reviews": [self._review_to_dict(r) for r in self.reviews],
        }

    @staticmethod
    def _review_to_dict(r: Review) -> dict:
        return {
            "reviewer": r.reviewer,
            "rating": r.rating,
            "confidence": r.confidence,
            "summary": r.summary,
            "strengths": r.strengths,
            "weaknesses": r.weaknesses,
            "questions": r.questions,
            "weakness_points": [
                {"text": w.text, "bullet": w.bullet, "order": w.order}
                for w in r.weakness_points
            ],
            "author_replies": [
                {"text": ar.text, "timestamp": ar.timestamp, "signature": ar.signature}
                for ar in r.author_replies
            ],
            "review_id": r.review_id,
            "invitation": r.invitation,
        }


# ---------------------------------------------------------------------------
# Weakness-text → bullet points
# ---------------------------------------------------------------------------

_BULLET_MARKER_RE = re.compile(
    r"(?:^|(?<=[\n\r]))\s*(?:[-*•\u2022]|\(?\d+[).]|\(?[a-z][).])\s+",
    re.MULTILINE,
)


def split_weaknesses(text: str) -> list[ReviewerWeakness]:
    """
    Split a weaknesses field into individual concerns.

    Reviewers typically write either:
      - a bulleted list ("- X\n- Y\n- Z")
      - a numbered list ("1. X\n2. Y")
      - a prose paragraph

    We try bullets first; fall back to sentence splits.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # Find bullet markers
    markers = list(_BULLET_MARKER_RE.finditer(text))
    if len(markers) >= 2:
        points = []
        order = 0
        for i, m in enumerate(markers):
            start = m.end()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            chunk = text[start:end].strip()
            if len(chunk) < 5:
                continue
            points.append(ReviewerWeakness(text=chunk, bullet=True, order=order))
            order += 1
        if points:
            return points

    # Prose: sentence split
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    out = []
    for i, s in enumerate(sents):
        s = s.strip()
        if len(s) < 30:
            continue
        out.append(ReviewerWeakness(text=s, bullet=False, order=i))
    return out


# ---------------------------------------------------------------------------
# OpenReview clients (v2 first, fall back to v1)
# ---------------------------------------------------------------------------

_V2_URL = "https://api2.openreview.net"
_V1_URL = "https://api.openreview.net"


class _OpenReview:
    """Thin wrapper that tries v2 first, falls back to v1. Public data only."""

    def __init__(self):
        import openreview
        self._openreview = openreview
        try:
            self.v2 = openreview.api.OpenReviewClient(baseurl=_V2_URL)
        except Exception as e:
            log.warning("OpenReview v2 client init failed: %s", e)
            self.v2 = None
        try:
            self.v1 = openreview.Client(baseurl=_V1_URL)
        except Exception as e:
            log.warning("OpenReview v1 client init failed: %s", e)
            self.v1 = None

    def search_by_title(self, title: str, source: str = "all", limit: int = 25):
        """Search via BOTH v2 and v1 APIs (they cover different time ranges).

        v2 is for OpenReview's newer data (2023+); v1 is where older papers
        (ICLR 2018-2023, NeurIPS 2021-2022 blind_submissions) live. Merge
        results and let the caller dedupe by forum id.
        """
        results: list[dict] = []
        seen_forum_ids: set[str] = set()
        for client, tag in [(self.v2, "v2"), (self.v1, "v1")]:
            if client is None:
                continue
            try:
                notes = client.search_notes(term=title, source=source, limit=limit)
            except TypeError:
                # v1 client doesn't support `source=`; fall back
                try:
                    notes = client.search_notes(term=title, limit=limit)
                except Exception as e:
                    log.debug("search_notes v1 failed: %s", e)
                    continue
            except Exception as e:
                log.debug("search_notes %s failed: %s", tag, e)
                continue
            for n in notes:
                d = _note_to_dict(n)
                d["_api_tag"] = tag
                results.append(d)
        # Return all and the tag (we'll pick per-forum later). The caller's
        # downstream lookups (get_forum_replies) will try both APIs too.
        return results, "both"

    def get_forum_replies(self, forum_id: str, tag: str = "both") -> list[dict]:
        """Return all notes attached to a forum. Tries both APIs if needed."""
        attempts = []
        if tag in ("v2", "both") and self.v2 is not None:
            attempts.append(("v2", self.v2))
        if tag in ("v1", "both") and self.v1 is not None:
            attempts.append(("v1", self.v1))
        for cname, client in attempts:
            try:
                notes = client.get_all_notes(forum=forum_id)
            except Exception as e:
                log.debug("get_all_notes %s failed: %s", cname, e)
                continue
            if notes:
                return [_note_to_dict(n) for n in notes]
        return []


def _note_to_dict(note) -> dict:
    """Normalise an openreview-py note object into a plain dict."""
    # content may be a dict of {field: {value: ...}} (v2) or {field: ...} (v1)
    content = {}
    raw_content = getattr(note, "content", {}) or {}
    for k, v in raw_content.items():
        if isinstance(v, dict) and "value" in v:
            content[k] = v["value"]
        else:
            content[k] = v

    # invitations may be under .invitations (v2 list) or .invitation (v1 str)
    invitations = getattr(note, "invitations", None) or (
        [getattr(note, "invitation", None)] if getattr(note, "invitation", None) else []
    )
    invitations = [i for i in invitations if i]

    signatures = getattr(note, "signatures", None) or []

    return {
        "id": getattr(note, "id", None),
        "forum": getattr(note, "forum", None),
        "replyto": getattr(note, "replyto", None),
        "cdate": getattr(note, "cdate", None),
        "invitations": invitations,
        "signatures": signatures,
        "content": content,
    }


# ---------------------------------------------------------------------------
# Review extraction
# ---------------------------------------------------------------------------

_REVIEW_INVITATION_RE = re.compile(r"/Official_Review|/Review\b", re.IGNORECASE)
_AUTHOR_SIG_RE = re.compile(r"/Authors\b", re.IGNORECASE)
_DECISION_INVITATION_RE = re.compile(r"/Decision\b|/Meta_Review\b", re.IGNORECASE)

# OpenReview hosts multiple note types for the same paper title:
#   - The actual conference submission ("ICLR.cc/2024/Conference/-/Submission"
#     or the older ".../Blind_Submission")
#   - DBLP mirror records ("DBLP.org/-/Record")
#   - Public-article / ORCID mirrors
# We want the conference submission (which has reviews attached).
_CONFERENCE_INVITATION_RE = re.compile(
    r"/(?:Submission|Blind_Submission|Post_Submission|Camera_Ready_Revision)\b",
    re.IGNORECASE,
)
_MIRROR_INVITATION_RE = re.compile(
    r"^(?:dblp\.org|DBLP\.org|OpenReview\.net/Public_Article|ORCID\.org)/",
    re.IGNORECASE,
)


def _is_conference_submission(note: dict) -> bool:
    invs = note.get("invitations") or []
    for inv in invs:
        if _CONFERENCE_INVITATION_RE.search(inv or ""):
            return True
    return False


def _is_mirror_record(note: dict) -> bool:
    invs = note.get("invitations") or []
    if not invs:
        return False
    return all(_MIRROR_INVITATION_RE.match(i or "") for i in invs)


def _is_review(note: dict) -> bool:
    return any(_REVIEW_INVITATION_RE.search(i or "") for i in note.get("invitations", []))


def _is_author_reply(note: dict) -> bool:
    return any(_AUTHOR_SIG_RE.search(s or "") for s in note.get("signatures", []))


def _is_decision(note: dict) -> bool:
    return any(_DECISION_INVITATION_RE.search(i or "") for i in note.get("invitations", []))


def _clean_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    return str(v).strip()


def _get_rating(content: dict) -> Optional[str]:
    for k in ("rating", "recommendation", "overall_score", "score", "soundness"):
        if k in content:
            return _clean_text(content[k])
    return None


def _get_confidence(content: dict) -> Optional[str]:
    for k in ("confidence", "reviewer_confidence"):
        if k in content:
            return _clean_text(content[k])
    return None


def _build_review(note: dict) -> Review:
    content = note.get("content") or {}

    # Structured schema (ICLR/NeurIPS 2023+, COLM, TMLR): explicit weaknesses field.
    weaknesses = _clean_text(
        content.get("weaknesses")
        or content.get("weaknesses_and_limitations")
        or ""
    )

    # Legacy schema (NeurIPS 2021-2022, older ICLR): one big review field. We
    # store it in `weaknesses` so downstream weakness parsing sees it — the
    # split_weaknesses function will do its best; the LLM downstream will
    # then separate true weaknesses from strengths/other content.
    if not weaknesses:
        prose = _clean_text(
            content.get("main_review")
            or content.get("review")
            or content.get("comment")
            or ""
        )
        if prose:
            # Try to extract a "Weaknesses" / "Cons" section if present
            ws = _extract_weakness_subsection(prose)
            weaknesses = ws if ws else prose

    return Review(
        reviewer=_reviewer_of(note),
        rating=_get_rating(content),
        confidence=_get_confidence(content),
        summary=_clean_text(
            content.get("summary") or content.get("summary_of_the_paper") or ""
        ),
        strengths=_clean_text(
            content.get("strengths") or content.get("strengths_and_weaknesses", "") or ""
        ),
        weaknesses=weaknesses,
        questions=_clean_text(content.get("questions") or ""),
        weakness_points=split_weaknesses(weaknesses),
        review_id=note.get("id", ""),
        invitation=(note.get("invitations") or [""])[0],
    )


def _extract_weakness_subsection(prose: str) -> str:
    """
    For monolithic 'main_review' fields, try to find a 'Weaknesses' / 'Cons'
    / 'Limitations' subsection and return only that. Falls back to empty
    string if no such section found.
    """
    # Common section headers reviewers use inside prose reviews
    patterns = [
        r"(?:^|\n)\s*(?:[#*]{1,3}\s*)?(?:weaknesses?|cons|limitations|negatives?|concerns?)[:\s]*\n(.+?)(?:\n\s*(?:[#*]{1,3}\s*)?(?:strengths?|pros|summary|suggestions?|questions?|minor|recommendation|rating|overall)[:\s]|\Z)",
    ]
    for pat in patterns:
        m = re.search(pat, prose, re.IGNORECASE | re.DOTALL)
        if m:
            text = m.group(1).strip()
            if len(text) > 30:
                return text
    return ""


def _reviewer_of(note: dict) -> str:
    sigs = note.get("signatures") or []
    for s in sigs:
        if s:
            # Last path component, e.g. "Reviewer_abc1"
            return s.rstrip("/").split("/")[-1]
    return ""


def _title_similarity(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"\w+", a.lower()))
    b_tokens = set(re.findall(r"\w+", b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_openreview_bundle(
    title: str,
    arxiv_id: Optional[str] = None,
    min_title_sim: float = 0.7,
    throttle_s: float = 0.5,
) -> Optional[OpenReviewBundle]:
    """
    Locate the OpenReview submission whose title matches `title` and return
    the structured review bundle. Returns None if no match.

    Args:
      title:          paper title (case-insensitive; tokens compared)
      arxiv_id:       optional; not used for lookup but stored in the bundle
      min_title_sim:  minimum token-overlap fraction to consider a hit (0-1)
    """
    try:
        or_client = _OpenReview()
    except Exception as e:
        log.warning("OpenReview client construction failed: %s", e)
        return None

    hits, tag = or_client.search_by_title(title, source="all", limit=25)
    if not hits:
        return None

    # Group search hits by forum id. For each forum, keep the "best" note
    # (prefer a Submission note with a non-empty title) — this gives us the
    # canonical root of each matching forum.
    forums: dict[str, dict] = {}
    for h in hits:
        fid = h.get("forum") or h.get("id")
        if not fid:
            continue
        current = forums.get(fid)
        if current is None:
            forums[fid] = h
        elif _is_conference_submission(h) and not _is_conference_submission(current):
            forums[fid] = h

    # Rank forums: conference submissions first, then by title similarity.
    ranked = []
    for fid, h in forums.items():
        if _is_mirror_record(h):
            continue
        t = _clean_text((h.get("content") or {}).get("title") or "")
        if not t:
            # The best note for this forum lacks a title (e.g., a review
            # note). Fall back to any sibling hit from the same forum that
            # does have one.
            for alt in hits:
                if (alt.get("forum") or alt.get("id")) == fid:
                    alt_t = _clean_text((alt.get("content") or {}).get("title") or "")
                    if alt_t:
                        t = alt_t
                        break
        sim = _title_similarity(title, t)
        is_conf = _is_conference_submission(h) or any(
            _is_conference_submission(alt)
            for alt in hits
            if (alt.get("forum") or alt.get("id")) == fid
        )
        ranked.append((is_conf, sim, t, fid, h))
    if not ranked:
        log.info("Only mirror records found for %r", title)
        return None

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    is_conf, sim, best_title, forum_id, best = ranked[0]
    if sim < min_title_sim:
        log.info("Best OpenReview match for %r is %r (sim=%.2f); rejecting",
                 title, best_title, sim)
        return None
    if not is_conf:
        log.info("Best match %r is not a conference submission; it won't have reviews",
                 best_title)
        return None
    log.info("OpenReview match: %r (sim=%.2f), forum=%s", best_title, sim, forum_id)

    time.sleep(throttle_s)
    replies = or_client.get_forum_replies(forum_id, tag=tag)

    # Build review objects + attach author replies
    review_notes = [n for n in replies if _is_review(n)]
    author_replies = [n for n in replies if _is_author_reply(n)]
    decisions = [n for n in replies if _is_decision(n)]

    reviews: list[Review] = []
    for rn in review_notes:
        rev = _build_review(rn)
        # Attach any author note that replies to this review (directly or via descendant)
        direct = [a for a in author_replies if a.get("replyto") == rn["id"]]
        for a in direct:
            content = a.get("content") or {}
            text = _clean_text(
                content.get("comment") or content.get("response") or content.get("rebuttal") or ""
            )
            if not text:
                # Fall back to concatenation of all string fields
                text = "\n\n".join(
                    f"[{k}] {_clean_text(v)}"
                    for k, v in content.items()
                    if isinstance(v, str) and _clean_text(v)
                )
            rev.author_replies.append(AuthorReply(
                text=text,
                timestamp=a.get("cdate"),
                signature=(a.get("signatures") or [""])[0],
            ))
        reviews.append(rev)

    # Figure out venue from invitation string
    venue_id = ""
    if reviews:
        inv = reviews[0].invitation or ""
        # typical: "ICLR.cc/2024/Conference/Submission123/-/Official_Review"
        m = re.match(r"([^/]+/[^/]+(?:/[^/]+)?)/", inv)
        if m:
            venue_id = m.group(1)

    decision = None
    if decisions:
        content = decisions[0].get("content") or {}
        decision = _clean_text(
            content.get("decision") or content.get("recommendation") or ""
        )

    bundle = OpenReviewBundle(
        forum_id=forum_id,
        venue_id=venue_id,
        paper_title=best_title or title,
        decision=decision,
        reviews=reviews,
        n_total_replies=len(replies),
    )
    return bundle


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json, logging as _logging
    _logging.basicConfig(level=_logging.INFO)

    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True, help="paper title (used for search)")
    ap.add_argument("--arxiv", default=None)
    args = ap.parse_args()

    bundle = fetch_openreview_bundle(title=args.title, arxiv_id=args.arxiv)
    if bundle is None:
        print("No OpenReview match found.")
        raise SystemExit(1)

    print(f"\nForum: {bundle.forum_id}")
    print(f"Venue: {bundle.venue_id}")
    print(f"Title: {bundle.paper_title}")
    print(f"Decision: {bundle.decision}")
    print(f"Reviews: {len(bundle.reviews)}  ({bundle.n_total_replies} total replies)")
    for r in bundle.reviews:
        print(f"\n--- {r.reviewer}  rating={r.rating}  conf={r.confidence} ---")
        print(f"  summary: {r.summary[:200]}")
        if r.weakness_points:
            print(f"  {len(r.weakness_points)} weakness points:")
            for wp in r.weakness_points[:5]:
                print(f"    [{'•' if wp.bullet else '-'}] {wp.text[:200]}")
        if r.author_replies:
            print(f"  {len(r.author_replies)} author replies (first: {r.author_replies[0].text[:120]!r})")
