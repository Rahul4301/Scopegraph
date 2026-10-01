# Benchmark Protocol

Every ScopeGraph run records its histories, queries, embedding and answer models, prompts, temperature, top-k budget, token budget, and scoring. Configuration, random seed, model identifiers, timestamp, git commit, and configuration hash are stored with each run. Generated benchmark data must not be presented under an external benchmark name. External datasets require validation and documented acquisition steps.

The offline runner uses deterministic providers only for plumbing tests. The external
suite contains exactly LongMemEval-S, LoCoMo, and MemoryAgentBench and runs all 6,157
official questions. LongMemEval uses its pinned GPT-4o judge, LoCoMo uses a pinned GPT-4o
rubric judge (with its official category-aware F1 reported alongside), and MemoryAgentBench uses its task-specific exact/substring/recall
metrics plus its pinned LongMemEval and summarization judges. Raw execution and
aggregation remain separate:

```text
run_eval -> results/raw/*.jsonl -> run_report -> report.md + processed/tables
```

The thesis-specific controlled suite is CrossScopeMem. Each scenario is one account
with a global root, at least four simultaneous top-level projects, a nested repository,
standalone conversations, and multiple chronological sessions. It is reported
separately from external benchmark scores. One frozen extraction artifact is replayed
through full ScopeGraph, vector-only, vector-with-scope-filter, flat-graph, and two-level
session/global controls, plus no-graph-traversal and no-temporal/status ablations. The flat-graph control also
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

## Statistical conventions

- The resampling unit is the account (CrossScopeMem scenario) or conversation (LoCoMo).
  Intervals are 95% percentile bootstrap intervals, 10,000 resamples, seed 42.
- A pilot uses at least 10 accounts; the proposal target is 40–60. Reports mark anything
  under 10 accounts `underpowered`, and a single account produces no interval.
- Every paired difference is **full minus control**. For lower-is-better metrics
  (`cross_scope_contamination`, `any_cross_scope_contamination`, `stale_memory_error_rate`)
  a negative difference favours full.
- Exact McNemar tests are reported only for per-question 0/1 outcomes. They treat questions as
  independent; because questions share accounts, the account-level sign test is reported beside
  them and should be preferred when they disagree.
- Latencies are reported per dataset and protocol, split into retrieval, embedding, answer, and
  judge time. They are never summed across stages or compared across datasets.

## Protocol amendments

Amendments are appended with a date and never edit the original text above.

### 2026-10-01 — `vector_scope_filter` control (CrossScopeMem)

Adds a seventh condition: the `vector_only_control` ranker (cosine similarity only, no hierarchy,
graph, or recency weights) with a hard metadata filter restricting search to the question's
`allowed_scope_ids`. It separates "a correct access filter" from "ScopeGraph's hierarchy". Its
cross-scope contamination is 0 by construction because the metric is defined against the same
`allowed_scope_ids`; compare it on recall and precision, not contamination.

### 2026-10-01 — plain vector baseline for LoCoMo (outside the original protocol)

The original protocol runs only full ScopeGraph on external benchmarks. This amendment adds a
LoCoMo-only `vector_only` condition that reuses the same frozen extraction, embeddings, answer
prompt, answer model, and pinned judge, and changes only retrieval: semantic weight 1.0, no
lexical term, no graph expansion, no scope/temporal/confidence/recency weights. It exists so a
LoCoMo number can be compared with something other than itself. Results from it are labelled
`amendment` and are not part of the original pre-registered comparison. It is not run for
LongMemEval-S or MemoryAgentBench.

### 2026-10-01 — answer model receives scope and as-of time (CrossScopeMem)

Earlier CrossScopeMem live runs did not give the answer model the current scope or the query
timestamp, so "this project" had no referent (7 of 190 answers were `UNKNOWN` with the gold
evidence ranked first). The answerer now receives both. External-benchmark answering is
unchanged (a single opaque scope), so LoCoMo numbers remain comparable with earlier runs.
