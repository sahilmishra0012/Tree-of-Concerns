"""
Annotation CLI.

A human annotator walks through LLM-proposed candidate limitations for a
paper and marks each one accepted / edited / rejected. Produces a JSONL
annotated file consumable by the evaluation script.

Usage:
    python -m benchmark.annotate \\
        --candidates data/arxiv_2109.10500/candidates.json \\
        --paper      data/arxiv_2109.10500/paper.json \\
        --output     data/arxiv_2109.10500/annotated.json

Hotkeys: (a)ccept  (e)dit  (r)eject  (s)kip  (q)uit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from schema.models import AnnotationStatus, Limitation, Paper


def _prompt(prompt: str, default: str = "") -> str:
    s = input(f"{prompt} ").strip()
    return s or default


def annotate_one(lim: Limitation, paper: Paper) -> Limitation:
    """Interactively annotate a single candidate limitation."""
    print("\n" + "=" * 72)
    print(f"PAPER: {paper.title}")
    print(f"ID:    {lim.id}")
    print(f"CATEGORY: {lim.category.value}  |  SEVERITY: {lim.severity.value}  |  LLM-CONF: {lim.llm_confidence}")
    print("-" * 72)
    print("DESCRIPTION:")
    print(f"  {lim.description}")
    if lim.evidence_in_paper:
        print("EVIDENCE IN PAPER:")
        print(f"  [section={lim.evidence_in_paper.section}]")
        print(f"  \"{lim.evidence_in_paper.text}\"")
    print(f"SOURCE: {lim.source.short()}")
    if lim.source.quote:
        print(f"  source quote: \"{lim.source.quote}\"")
    print("-" * 72)

    while True:
        choice = input("(a)ccept / (e)dit / (r)eject / (s)kip / (q)uit > ").strip().lower()
        if choice == "a":
            lim.status = AnnotationStatus.ACCEPTED
            return lim
        if choice == "r":
            reason = _prompt("Rejection reason (optional):")
            lim.status = AnnotationStatus.REJECTED
            if reason:
                lim.annotator_notes = (lim.annotator_notes or "") + f"\n[reject] {reason}"
            return lim
        if choice == "e":
            lim.original_description = lim.description
            new_desc = _prompt("New description (or blank to keep):", lim.description)
            lim.description = new_desc
            new_cat = _prompt(f"Category [{lim.category.value}]:", lim.category.value)
            try:
                from schema.models import LimitationCategory
                lim.category = LimitationCategory(new_cat.lower())
            except ValueError:
                print("  (invalid category; keeping)")
            new_sev = _prompt(f"Severity [{lim.severity.value}]:", lim.severity.value)
            try:
                from schema.models import Severity
                lim.severity = Severity(new_sev.lower())
            except ValueError:
                print("  (invalid severity; keeping)")
            lim.status = AnnotationStatus.EDITED
            note = _prompt("Annotator note (optional):")
            if note:
                lim.annotator_notes = (lim.annotator_notes or "") + f"\n[edit] {note}"
            return lim
        if choice == "s":
            print("  (skipped; keeping status CANDIDATE)")
            return lim
        if choice == "q":
            raise KeyboardInterrupt
        print("  unknown choice")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="path to JSON list of Limitation objects")
    ap.add_argument("--paper", required=True, help="path to JSON Paper object")
    ap.add_argument("--output", required=True, help="path to write annotated JSONL")
    ap.add_argument("--resume", action="store_true",
                    help="resume annotation, skipping already-annotated IDs from output")
    args = ap.parse_args()

    paper = Paper.model_validate(json.loads(Path(args.paper).read_text()))
    candidates_raw = json.loads(Path(args.candidates).read_text())
    candidates = [Limitation.model_validate(c) for c in candidates_raw]

    already_done = set()
    out_path = Path(args.output)
    if args.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                already_done.add(json.loads(line)["id"])
        print(f"Resuming; {len(already_done)} already annotated")

    # Open in append mode if resuming
    mode = "a" if args.resume else "w"
    with out_path.open(mode) as f:
        try:
            for i, lim in enumerate(candidates):
                if lim.id in already_done:
                    continue
                print(f"\n[{i+1}/{len(candidates)}]")
                annotated = annotate_one(lim, paper)
                f.write(annotated.model_dump_json() + "\n")
                f.flush()
        except KeyboardInterrupt:
            print("\nInterrupted. Progress saved.")
            sys.exit(0)
    print(f"\nDone. Wrote annotations to {out_path}")


if __name__ == "__main__":
    main()
