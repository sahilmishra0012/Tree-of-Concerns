"""
Expand the LimitTree corpus with NLP/linguistics papers.

Uses NLP-specific topical queries and filters to ACL/EMNLP/NAACL/TACL-type
venues (plus COLM/TMLR which are on OpenReview).

Writes to the same corpus directory — paper folders named arxiv_* are shared,
so resumable via status.json. Uses a separate seeds file `seeds_nlp.json`
to avoid overwriting the main seeds list.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_corpus import run_corpus


# NLP-focused queries
NLP_QUERIES = [
    "named entity recognition",
    "machine translation",
    "question answering",
    "text summarization",
    "dialogue system",
    "sentiment analysis",
    "semantic parsing",
    "coreference resolution",
    "information extraction",
    "natural language inference",
    "reading comprehension",
    "text generation",
    "dependency parsing",
    "word sense disambiguation",
    "relation extraction",
    "cross-lingual transfer",
    "commonsense reasoning",
    "text classification",
    "grammatical error correction",
    "discourse parsing",
]

# Narrow venue filter to NLP-centric venues, plus OpenReview-accessible ones
NLP_VENUES = [
    "Annual Meeting of the Association for Computational Linguistics",
    "Conference on Empirical Methods in Natural Language Processing",
    "North American Chapter of the Association for Computational Linguistics",
    "Conference of the European Chapter of the Association for Computational Linguistics",
    "Transactions of the Association for Computational Linguistics",
    "Findings of the Association for Computational Linguistics",
    "International Conference on Computational Linguistics",
    "Conference on Computational Natural Language Learning",
    "Conference on Language Modeling",
    "Transactions on Machine Learning Research",
]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus")
    ap.add_argument("--target_kept", type=int, default=100,
                    help="target number of NEW kept NLP papers")
    ap.add_argument("--n_seeds", type=int, default=500)
    ap.add_argument("--year_range", type=str, default="2020-2024")
    ap.add_argument("--min_citations", type=int, default=50,
                    help="lower threshold for NLP (smaller community)")
    ap.add_argument("--min_critique_contexts", type=int, default=2)
    ap.add_argument("--min_or_weaknesses", type=int, default=2)
    ap.add_argument("--max_citers", type=int, default=60)
    ap.add_argument("--throttle_s", type=float, default=0.8)
    ap.add_argument("--seeds_only", action="store_true")
    ap.add_argument("--max_fetch", type=int, default=None)
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
        queries=NLP_QUERIES,
        venues=NLP_VENUES,
        seeds_filename="seeds_nlp.json",
    )
    print("\n" + "=" * 70)
    print("NLP EXPANSION SUMMARY")
    print("=" * 70)
    for k, v in manifest.items():
        if isinstance(v, list):
            print(f"{k}: ({len(v)} items)")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
