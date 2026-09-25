# Implementation audit (2026-09-23)

This is an audit of the checked-in implementation and locally acquired benchmark
files. The research proposal and master prompt are not present in this repository or
its Git history, so this document does not claim a line-by-line proposal comparison.

## Confirmed alignment

- ScopeGraph is the only memory system implemented and exported.
- Production and research-facing evaluation commands use the isolated Neo4j
  evaluation service. The in-memory repository remains only as a unit-test and smoke
  test double.
- Memory records retain scope, temporal state, source-message provenance, graph
  relations, and reversible correction history.
- Raw evaluation records retain configuration and source fingerprints, model names,
  retrieval traces, latency, token counts, and storage statistics.

## Remaining evaluation cautions

1. All three selected complete releases validate and their official scoring protocols
   are wired in, but no full live result should be claimed until `make eval-suite`
   completes and its output is audited.
2. External examples are mapped to one custom scope beneath a global root. That tests
   retrieval over a memory history, but it does not test ScopeGraph's central claim
   about interference among multiple project/context scopes.
3. External corpora contain no meaningful competing-project structure, so architecture
   controls are not run on them. The separate CrossScopeMem account suite supplies the
   required competing scopes and reports vector-only, flat-graph, and two-level controls.
4. Retrieval timing excludes ingestion, extraction, embedding preparation, answer
   generation, and grading. It must not be presented as end-to-end latency.
5. The external runner is serial. Provider retries exist, but question-level
   concurrency and explicit monetary cost accounting do not.

## Engineering limits

- Exact cosine scoring scans every eligible memory in the selected scopes. Very large
  individual scopes need a measured vector-index design before scalability claims.
- The API has no authentication, account/tenant authorization, quotas, distributed workers,
  or production backup policy.

These gaps must be resolved before describing the outputs as official benchmark
scores or as proof of general usefulness. A small, clearly labeled pilot can support
an undergraduate prototype/feasibility claim, but not benchmark parity, superiority,
or broad generalization.
