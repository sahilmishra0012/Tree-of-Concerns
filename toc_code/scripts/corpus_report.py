"""
Corpus statistics report.

Reads all status.json files under `data/corpus/papers/`, plus the per-paper
critique_contexts.json / openreview.json / stated_limitations.json where
available, and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def load_stats(papers_dir: Path) -> list[dict]:
    out = []
    for d in sorted(papers_dir.iterdir()):
        if not d.is_dir():
            continue
        sp = d / "status.json"
        if not sp.exists():
            continue
        out.append(json.loads(sp.read_text()))
    return out


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{num / denom * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default="data/corpus")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    root = Path(args.corpus)
    papers_dir = root / "papers"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None

    stats = load_stats(papers_dir)

    kept = [
        s for s in stats
        if (s.get("has_openreview") and s.get("n_openreview_weakness_points", 0) >= 2)
        or s.get("n_critique_contexts", 0) >= 2
    ]

    print("=" * 70)
    print(f"LimitTree corpus report  —  {root}")
    print("=" * 70)
    print(f"Seeds discovered:  {len(json.loads((root/'seeds.json').read_text())) if (root/'seeds.json').exists() else 'n/a'}")
    print(f"Fetched total:     {len(stats)}")
    print(f"Kept total:        {len(kept)}")
    if manifest:
        print(f"Target kept:       {manifest.get('target_kept')}")
        print(f"Errors:            {len(manifest.get('errors', []))}")
    print()

    # Source availability
    n_full_text = sum(1 for s in kept if s.get("has_full_text"))
    n_stated = sum(1 for s in kept if s.get("has_limitations_section"))
    n_or = sum(1 for s in kept if s.get("has_openreview"))
    n_gold = sum(
        1 for s in kept
        if s.get("has_openreview") and s.get("n_openreview_weakness_points", 0) >= 2
    )
    n_silver = sum(
        1 for s in kept
        if not (s.get("has_openreview") and s.get("n_openreview_weakness_points", 0) >= 2)
        and s.get("n_critique_contexts", 0) >= 2
    )
    print("Source availability (of kept):")
    print(f"  has full text:              {n_full_text}  ({_pct(n_full_text, len(kept))})")
    print(f"  has Limitations section:    {n_stated}  ({_pct(n_stated, len(kept))})")
    print(f"  has OpenReview (any):       {n_or}  ({_pct(n_or, len(kept))})")
    print()
    print("Tier split:")
    print(f"  GOLD   (OpenReview ≥2 weak): {n_gold}  ({_pct(n_gold, len(kept))})")
    print(f"  SILVER (critique ≥2 only):   {n_silver}  ({_pct(n_silver, len(kept))})")
    print()

    # Volume of raw signal
    or_weak_counts = [s["n_openreview_weakness_points"] for s in kept if s.get("has_openreview")]
    critique_counts = [s["n_critique_contexts"] for s in kept if s.get("n_critique_contexts", 0) > 0]
    stated_counts = [s["n_stated_limitations"] for s in kept if s.get("n_stated_limitations", 0) > 0]
    citer_counts = [s["n_citers"] for s in kept if s.get("n_citers", 0) > 0]
    n_reviews_total = sum(s.get("n_openreview_reviews", 0) for s in kept)

    print("Signal volume:")
    print(f"  total citers scraped:        {sum(citer_counts)}")
    print(f"  total OpenReview reviews:    {n_reviews_total}")
    print(f"  total OpenReview weakness points: {sum(or_weak_counts)}")
    print(f"  total critique contexts:     {sum(critique_counts)}")
    print(f"  total stated limitations:    {sum(stated_counts)}")
    if or_weak_counts:
        print(f"  per-paper OpenReview weakness points: "
              f"median={statistics.median(or_weak_counts):.0f} "
              f"mean={statistics.mean(or_weak_counts):.1f} "
              f"max={max(or_weak_counts)}")
    if critique_counts:
        print(f"  per-paper critique contexts:          "
              f"median={statistics.median(critique_counts):.0f} "
              f"mean={statistics.mean(critique_counts):.1f} "
              f"max={max(critique_counts)}")
    print()

    # Top papers by OpenReview weakness volume
    or_papers = [s for s in kept if s.get("has_openreview")]
    or_papers.sort(key=lambda s: s.get("n_openreview_weakness_points", 0), reverse=True)
    print("Top 10 papers by OpenReview weakness points:")
    for s in or_papers[:10]:
        print(f"  {s['arxiv_id']}  weak={s['n_openreview_weakness_points']:3d}  {s['title'][:70]}")
    print()

    print("Top 10 papers by critique-context count:")
    cc_papers = sorted(kept, key=lambda s: s.get("n_critique_contexts", 0), reverse=True)
    for s in cc_papers[:10]:
        print(f"  {s['arxiv_id']}  crit={s['n_critique_contexts']:3d}  {s['title'][:70]}")
    print()

    # Seed venue breakdown
    seeds_path = root / "seeds.json"
    if seeds_path.exists():
        seeds_by_id = {
            s["arxiv_id"]: s for s in json.loads(seeds_path.read_text())
        }
        venue_counts = Counter()
        year_counts = Counter()
        for s in kept:
            seed = seeds_by_id.get(s["arxiv_id"], {})
            venue_counts[seed.get("venue") or "?"] += 1
            year_counts[seed.get("year") or "?"] += 1
        print("Venue distribution (kept):")
        for v, c in venue_counts.most_common(12):
            print(f"  {c:4d}  {v[:70]}")
        print()
        print("Year distribution (kept):")
        for y in sorted(year_counts):
            print(f"  {y}: {year_counts[y]}")

    # Write machine-readable summary
    summary = {
        "n_seeds": len(json.loads((root/'seeds.json').read_text())) if (root/'seeds.json').exists() else None,
        "n_fetched": len(stats),
        "n_kept": len(kept),
        "n_gold": n_gold,
        "n_silver": n_silver,
        "n_has_full_text": n_full_text,
        "n_has_stated_limitations": n_stated,
        "n_has_openreview": n_or,
        "total_reviews": n_reviews_total,
        "total_openreview_weakness_points": sum(or_weak_counts),
        "total_critique_contexts": sum(critique_counts),
        "total_stated_limitations": sum(stated_counts),
    }
    out_path = Path(args.out) if args.out else (root / "summary.json")
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {out_path}")


if __name__ == "__main__":
    main()
