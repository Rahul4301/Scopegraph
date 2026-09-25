# Benchmark Protocol

Every ScopeGraph run records its histories, queries, embedding and answer models, prompts, temperature, top-k budget, token budget, and scoring. Configuration, random seed, model identifiers, timestamp, git commit, and configuration hash are stored with each run. Generated benchmark data must not be presented under an external benchmark name. External datasets require validation and documented acquisition steps.

The offline runner uses deterministic providers only for plumbing tests. The external
suite contains exactly LongMemEval-S, LoCoMo, and MemoryAgentBench and runs all 6,157
official questions. LongMemEval uses its pinned GPT-4o judge, LoCoMo uses its official
category-aware F1, and MemoryAgentBench uses its task-specific exact/substring/recall
metrics plus its pinned LongMemEval and summarization judges. Raw execution and
aggregation remain separate:

```text
run_eval -> results/raw/*.jsonl -> run_report -> processed/tables/figures
```

The thesis-specific controlled suite is CrossScopeMem. Each scenario is one account
with a global root, at least four simultaneous top-level projects, a nested repository,
standalone conversations, and multiple chronological sessions. It is reported
separately from external benchmark scores. One frozen extraction artifact is replayed
through full ScopeGraph, vector-only, flat-graph, and two-level session/global controls,
plus no-graph-traversal and no-temporal/status ablations. The flat-graph control also
serves as the proposal's no-hierarchy ablation, avoiding a duplicate run.

The correction-persistence runner uses the ScopeGraph correction service with the
isolated Neo4j evaluation repository and records relapse at fixed future-session
offsets.

CrossScopeMem establishes controlled evidence about scope isolation and component
causality, but not external validity by itself. External ScopeGraph results
must use each release's official source histories and grading protocol, with the
answer model, judge, top-k/context budget, warm-up policy, and latency boundary
recorded. Local smoke-test timings must not be presented as end-to-end latency.

Confidence intervals resample complete source histories/accounts, never individual
questions as if questions sharing one history were independent. Ablation comparisons
are paired on identical account/question IDs and include exact McNemar tests for binary
metrics.
