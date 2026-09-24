# Benchmark Protocol

Every ScopeGraph run records its histories, queries, embedding and answer models, prompts, temperature, top-k budget, token budget, and scoring. Configuration, random seed, model identifiers, timestamp, git commit, and configuration hash are stored with each run. Generated benchmark data must not be presented under an external benchmark name. External datasets require validation and documented acquisition steps.

The offline runner uses deterministic providers only for plumbing tests. The research
suite contains exactly LongMemEval-S, LoCoMo, and MemoryAgentBench and runs all 6,157
official questions. LongMemEval uses its pinned GPT-4o judge, LoCoMo uses its official
category-aware F1, and MemoryAgentBench uses its task-specific exact/substring/recall
metrics plus its pinned LongMemEval and summarization judges. Raw execution and
aggregation remain separate:

```text
run_eval -> results/raw/*.jsonl -> run_report -> processed/tables/figures
```

The correction-persistence runner uses the ScopeGraph correction service with the
isolated Neo4j evaluation repository and records relapse at fixed future-session
offsets.

Synthetic results establish regression behavior only. External ScopeGraph results
must use each release's official source histories and grading protocol, with the
answer model, judge, top-k/context budget, warm-up policy, and latency boundary
recorded. Local smoke-test timings must not be presented as end-to-end latency.
