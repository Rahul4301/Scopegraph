# Audit status (2026-09-21)

This audit checked the research proposal and master prompt against the implementation,
historical result artifacts, the Python and web test gates, and a live Neo4j round trip.
It is an engineering and protocol audit, not an independent replication.

## Result interpretation

| Artifact | What it can support | Important limitation |
| --- | --- | --- |
| `20260920T054404792447Z` | diagnosis of the original 3-scenario live run | old v2 generator, only 15 questions per system |
| `audit-report/frozen-replay` | effect of retrieval fixes on the exact saved extraction and embeddings | no new extraction, answer generation, or judge calls |
| `20260921T051029657338Z` | small v3 live-pipeline smoke result | only 3 synthetic scenarios; predates this final protocol audit |
| `20260921T154349106767Z` | current offline retrieval regression result | deterministic oracle extraction and hash embeddings, not model quality |

The current 10-scenario, difficulty-3 offline batch contains 190 questions per
system. ScopeGraph reached 0.9944 Recall@8, 0 cross-scope contamination, and a
0.392 ms median warm in-memory retrieval time. The vector baseline reached 0.7306
recall, 0.7507 contamination, and 0.813 ms median retrieval. Two missed gold items
were both multi-answer comparison cases where the deliberately weak hash embedder
ranked same-scope release distractors above one required fact. These numbers are
useful regression evidence only.

The retained 3-scenario live smoke batch reports 0.9825 exact match for ScopeGraph
versus 0.3684 for vector and flat graph and 0.4035 for two-level graph. It is too
small and synthetic to support a paper or product superiority claim.

The local scaling microbenchmark keeps 50 eligible memories in the current scope
while increasing total unrelated memories from 500 to 50,000. The final medians
were 0.261, 0.295, and 0.470 ms. This measures warm Python/in-memory scope indexing;
it is not Neo4j, network, ingestion, answer-model, or hosted-service latency.

## Position relative to Supermemory

Supermemory's published [LongMemEval-S report](https://supermemory.ai/research/longmembench/)
reports 95% Recall@15 with aggregation and roughly 720 mean context tokens. Its
open-source [MemoryBench](https://github.com/supermemoryai/memorybench) checkpoints
ingest, index, search, answer, evaluation, and reporting and compares accuracy,
search latency, and context tokens. ScopeGraph does not currently have an
apples-to-apples result against those numbers.

ScopeGraph's testable distinction is explicit nested session/context/global validity,
origin-scope contamination measurement, source provenance, and reversible structural
correction. Supermemory already provides temporal/relational memory, hybrid search,
and container isolation, so “uses a graph” or “supports scoped tags” is not a
defensible differentiator by itself.

Before claiming parity or superiority, run both systems on the same released
LongMemEval/LoCoMo examples with the same ingestion cutoff, answer model, judge,
top-k/context budget, warm-up policy, and latency boundary. Report confidence
intervals, failures, provider costs, and both warm and cold paths. A Supermemory API
key or a pinned self-hosted version is required for that external comparison.

## Remaining limits

- Exact cosine scoring still scans every eligible memory inside a selected scope.
  Unrelated scopes are now indexed away, but very large single scopes need a measured
  ANN/vector-index design before production-scale claims.
- The API has no authentication, tenant authorization, quotas, production backup
  policy, or distributed job execution. It should remain on trusted local networks.
- External adapters and resumable replay exist, but full released-dataset results
  have not been run in this repository.
- Scope-classification accuracy is emitted as a separate ingestion-stage artifact,
  rather than being duplicated across per-question retrieval records. It requires a
  live-extraction run; oracle-extraction runs explicitly mark it as not evaluated.
- The proposed 50-question parallel runner is intentionally not implemented. Its
  fairness and safety requirements are recorded in `docs/evaluation.md`.

## Verified gates

- Ruff: pass
- mypy strict mode: pass
- non-integration suite: 88 passed
- live Neo4j integration suite: 4 passed
- frontend TypeScript and production build: pass
- npm production dependency audit: 0 known vulnerabilities
- schema application, `/health`, `/config/status`, `/stats`, and Vite serving: pass

Two deprecation warnings come from the Starlette test client's current `httpx`
compatibility layer; they do not represent failing application behavior.
