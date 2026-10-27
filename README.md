# Tree-of-Concerns

This is the source code for "Tree-of-Concerns: Hierarchical Multi-Agent
Debate for Extracting Unstated Limitations for Scientific Critique" and
the accompanying ToC-Bench benchmark construction pipeline.

## Layout

  framework/    The Tree-of-Concerns multi-agent debate framework.
                forest.py orchestrates five specialized skeptic personas
                running parallel debate trees; debate.py runs the
                four-stage debate per node; personas.py defines the
                Skeptic and Paper-advocate roles.
  benchmark/    Benchmark construction utilities: paper fetching from
                arXiv + Semantic Scholar, OpenReview review extraction,
                LLM-assisted candidate-limitation extraction, and the
                unstated-limitation filter.
  llm/          LLM client wrappers (Anthropic Claude via AWS Bedrock).
                bedrock.py exposes BedrockSyncClient, a synchronous
                on-demand invoke_model interface used by all driver
                scripts.
  scripts/      Driver scripts (all synchronous, on-demand):
                  build_corpus.py        fetch corpus from arXiv/SS
                  refresh_openreview.py  fetch OpenReview reviews
                  expand_nlp.py          NLP venue expansion
                  run_tier_a.py          gold extraction (Stage 1 + 2)
                  run_framework.py       run ToC framework on papers
                  run_baseline.py        run zero-shot / single-skeptic /
                                         DIAGPaper baseline
                  run_eval_match.py      LLM-judge match against gold
                  run_demo.py            single-paper demo
                  analyze_all.py         summarize results
                  corpus_report.py       corpus statistics
  schema/       Pydantic models: Paper, Limitation, Source, etc.
  utils/        Reserved.

## Setup

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Configuration

All Bedrock calls are synchronous (one HTTP request per LLM call). You
need AWS credentials with permission to invoke Anthropic Claude on
AWS Bedrock. Configure via environment:

    export AWS_PROFILE=<your-profile>
    export AWS_DEFAULT_REGION=us-west-2

## Quick start

Single-paper demo of the framework:

    python scripts/run_demo.py --paper-id arxiv:2001.04643

Full benchmark + framework pipeline (synchronous; takes hours):

    # 1. Build the corpus and fetch evidence
    python scripts/build_corpus.py
    python scripts/refresh_openreview.py
    python scripts/expand_nlp.py

    # 2. Extract gold limitations (Stage 1 + Stage 2 in one driver)
    python scripts/run_tier_a.py \
        --papers_root data/corpus/papers \
        --out_dir data/tier_a

    # 3. Run the framework on all papers
    python scripts/run_framework.py \
        --papers_root data/corpus/papers \
        --out_dir data/framework_run

    # 4. Run baselines for comparison
    python scripts/run_baseline.py --papers_root data/corpus/papers \
        --out_dir data/baseline_zs --method zero_shot
    python scripts/run_baseline.py --papers_root data/corpus/papers \
        --out_dir data/baseline_ss --method single_skeptic
    python scripts/run_baseline.py --papers_root data/corpus/papers \
        --out_dir data/baseline_diag --method diagpaper

    # 5. Score system outputs against gold
    python scripts/run_eval_match.py \
        --gold     data/tier_a/limitations.json \
        --system   data/framework_run/framework_limitations.json \
        --out_dir  data/eval_match

## Models used

Default backbone: Anthropic Claude Opus 4.6 via AWS Bedrock
(`global.anthropic.claude-opus-4-6-v1`). The LLM-Likert evaluator
panel additionally uses GPT-4o (OpenAI) and Claude Opus 4.7 (Bedrock).

## License

CC-BY-4.0, research use only.
