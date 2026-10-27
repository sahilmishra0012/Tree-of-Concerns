# ToC-Bench

Benchmark for unstated limitation extraction from scientific papers.

## Layout
  tocbench/
    README.md
    limitations.json           gold limitations for the train/dev split
    papers/                    314 paper records
    eval/
      limitations.json         gold limitations for the held-out eval split
      papers/                  100 paper records
      pilot_paper_ids.txt      20 paper IDs from the development pilot

## Counts
  Total corpus papers:        414
  Papers with gold limits:    390     (24 corpus papers have no surviving gold after the unstated filter)
  Total gold limitations:     1923
  Train/dev:                  314 papers, 1487 limitations
  Held-out eval:              100 papers, 436 limitations
    of which pilot:           20 papers

## Schema
  Limitation: paper_id, category {scope|methodology|theoretical|reproducibility|fairness},
              severity {major|minor}, description, evidence_in_paper{text,section}, source.
  Paper:      id, title, abstract, authors, year, venue, arxiv_id, doi, sections.

## License
CC-BY-4.0, research use only.
