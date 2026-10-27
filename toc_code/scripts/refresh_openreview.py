"""
Re-scan an existing corpus to refetch OpenReview data with the updated
fetcher (which now handles v1 API + legacy review schemas).

For each paper_dir under <corpus>/papers/:
  - Re-run fetch_openreview_bundle(title)
  - If we found a bundle with >= 2 reviews, overwrite openreview.json
  - Update status.json's openreview fields
  - Leave everything else untouched (don't re-fetch arxiv / citing papers)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.openreview_fetch import fetch_openreview_bundle
from schema.models import Paper


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("refresh_or")

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default="data/corpus")
    ap.add_argument("--throttle_s", type=float, default=0.3)
    ap.add_argument("--only_missing", action="store_true",
                    help="only refresh papers that currently have no OpenReview")
    args = ap.parse_args()

    papers_dir = Path(args.corpus) / "papers"
    dirs = sorted(d for d in papers_dir.iterdir() if d.is_dir())
    log.info("Refreshing OpenReview for %d papers in %s", len(dirs), papers_dir)

    n_checked = 0
    n_new_hits = 0
    n_new_weaknesses = 0
    n_now_has = 0
    for i, d in enumerate(dirs):
        status_path = d / "status.json"
        paper_path = d / "paper.json"
        if not status_path.exists() or not paper_path.exists():
            continue
        status = json.loads(status_path.read_text())
        if args.only_missing and status.get("has_openreview"):
            continue

        paper = Paper.model_validate(json.loads(paper_path.read_text()))
        n_checked += 1
        try:
            bundle = fetch_openreview_bundle(title=paper.title, arxiv_id=paper.arxiv_id)
        except Exception as e:
            log.warning("  [%d/%d] %s: OR fetch failed: %s", i + 1, len(dirs), paper.arxiv_id, e)
            time.sleep(args.throttle_s)
            continue

        if bundle is None or len(bundle.reviews) == 0:
            if (i + 1) % 20 == 0:
                log.info("[%d/%d] progress: new_hits=%d", i + 1, len(dirs), n_new_hits)
            time.sleep(args.throttle_s)
            continue

        prev = status.get("n_openreview_weakness_points", 0)
        prev_had = status.get("has_openreview", False)
        n_weak = sum(len(r.weakness_points) for r in bundle.reviews)

        (d / "openreview.json").write_text(
            json.dumps(bundle.to_dict(), indent=2, default=str)
        )
        status["has_openreview"] = True
        status["n_openreview_reviews"] = len(bundle.reviews)
        status["n_openreview_weakness_points"] = n_weak
        status_path.write_text(json.dumps(status, indent=2))

        if not prev_had:
            n_new_hits += 1
        n_new_weaknesses += (n_weak - prev)
        n_now_has += 1
        log.info("[%d/%d] %s: reviews=%d  weak=%d  (was: has=%s weak=%d)",
                 i + 1, len(dirs), paper.arxiv_id, len(bundle.reviews), n_weak,
                 prev_had, prev)
        time.sleep(args.throttle_s)

    log.info("Done. checked=%d  newly_hit=%d  total_now_has=%d  Δ_weaknesses=%d",
             n_checked, n_new_hits, n_now_has, n_new_weaknesses)


if __name__ == "__main__":
    main()
