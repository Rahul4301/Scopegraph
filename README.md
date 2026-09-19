# ScopeGraph

ScopeGraph is a research system for testing whether explicit session, project/context, and global memory scopes reduce cross-context retrieval errors in long-running LLM agents. It also tests whether editing persistent memory directly produces more durable corrections than adding a conversational correction.

The repository currently contains Phases 1 through 5: typed domain models, Neo4j persistence, provenance-aware ScopeGraph write/read paths, three controlled comparison baselines, and reversible audited corrections. It does not claim experimental results yet.

## Architecture

```text
SourceMessage -> Session -> Scope -> parent Scope -> Global root
      |                        ^
      +---- DERIVED_FROM ---- Memory

query -> semantic anchors -> permitted scopes -> bounded graph expansion
      -> temporal and scope ranking -> token-budgeted evidence
```

One physical Neo4j database represents three logical memory levels:

- session memory for temporary interaction state;
- scope memory for durable project, repository, course, client, task, or workspace facts;
- global memory for information intended to hold across contexts.

See [docs/architecture.md](docs/architecture.md), [docs/baselines.md](docs/baselines.md), [docs/corrections.md](docs/corrections.md), [docs/schema.md](docs/schema.md), and [docs/literature.md](docs/literature.md).

## Requirements

- Python 3.11+
- `uv`
- Docker Desktop or another Docker Compose runtime
- OpenAI-compatible LLM and embedding credentials for live extraction and retrieval

The Neo4j container is deliberately limited to a 512 MB heap and 256 MB page cache for Apple M1 machines with 8 GB RAM.

## Setup

```bash
cp .env.example .env
uv sync --extra dev
docker compose up -d neo4j
uv run python scripts/setup_neo4j.py
```

The default local credentials are development-only. Change `NEO4J_PASSWORD` and the matching `NEO4J_AUTH` value before exposing Neo4j beyond localhost.

## Run

```bash
uv run uvicorn scopegraph.api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation. `GET /health` checks Neo4j connectivity without exposing credentials.

`POST /retrieve` accepts a query, optional current scope and session, top-k, token budget, and optional evaluation timestamp. Retrieval searches the current session and scope, then ancestors and global memory; an unrelated scope is included only when its name appears in the query. The response includes each score component and traversal path.

Memory correction routes are grouped under `/memories/{id}`. Use `/prune/preview` before `/prune`; archive and prune operations can be reversed with `/restore`, and `/history` returns the append-only audit trail. `PATCH /memories/{id}` is an audited edit rather than an untracked property mutation.

## Quality checks

```bash
make check
```

The normal suite uses an in-memory repository and needs no services. The live Neo4j round-trip is opt-in and must use a disposable database:

```bash
SCOPEGRAPH_RUN_INTEGRATION=1 uv run pytest -m integration
```

## Configuration

`.env.example` contains all runtime secrets and provider settings. Versioned YAML files under `configs/` hold retrieval weights, memory policies, experiment controls, and logging configuration. Model names and API endpoints are never hardcoded into the memory engine.

## Phase status

Complete in Phase 1:

- repository structure and project tooling;
- Pydantic v2 models for scopes, sessions, source messages, memories, relationships, corrections, and retrieval traces;
- common `MemorySystem` interface for fair backend comparisons;
- resource-conscious Neo4j Docker Compose service;
- constraints, lookup indexes, and a content full-text index;
- parameterized Scope, Session, SourceMessage, and Memory CRUD;
- `/health`, scope, session, source-message, and memory endpoints;
- unit tests plus an opt-in live Neo4j integration test.

Complete in Phase 2:

- OpenAI-compatible structured JSON extraction behind a provider protocol;
- deterministic fake extraction for credential-free testing;
- exact source-message provenance validation;
- explicit-scope-first classification with visible uncertainty;
- normalization and same-scope duplicate suppression;
- conflict detection using normalized subject and predicate keys;
- non-destructive temporal supersession with graph edges;
- durability-gated session-to-scope consolidation;
- explicit or multi-scope-evidence global promotion;
- `POST /sessions/{id}/consolidate` using the configured live provider;
- offline unit coverage and a live Neo4j ingestion round trip.

Complete in Phase 3:

- an OpenAI-compatible embedding provider plus a persistent content-and-model-keyed SQLite cache;
- lazy embedding persistence on Neo4j memory nodes;
- semantic anchor selection restricted to the current session/scope hierarchy, global memory, and explicitly named scopes;
- allowlisted, cycle-safe graph expansion bounded by hop, node, and time limits;
- current versus historical temporal filtering;
- configurable semantic, scope, temporal, confidence, graph, and recency ranking;
- deterministic token-budget packing;
- `POST /retrieve` and per-result white-box retrieval traces;
- fixture coverage for scope isolation, session isolation, historical retrieval, caching, traversal, ranking, and token packing;
- a live Neo4j ingestion, embedding-persistence, and retrieval round trip.

Complete in Phase 4:

- `VectorMemory`, a flat semantic baseline with no graph traversal or scope filtering;
- `FlatGraphMemory`, a hybrid semantic/graph baseline with contextual scope validity disabled;
- `TwoLevelGraphMemory`, a session/global graph baseline that collapses durable contextual memories into global memory;
- the same extraction, embedding, provenance, temporal scoring, top-k, token budget, result, trace, statistics, and `MemorySystem` interfaces used by ScopeGraph;
- unit coverage for backend identity, visibility behavior, graph/no-graph behavior, and two-level session isolation;
- live Neo4j ingestion/retrieval round trips for all three baselines.

Complete in Phase 5:

- audited memory edits with revision increments and embedding invalidation;
- scope moves, archive, tombstone, restore, explicit supersession, and duplicate merge;
- dry-run prune previews separating graph neighbors from evidentiary dependents;
- dependency-safe pruning that marks only unsupported active dependents `needs_review`;
- guarded undo that restores affected dependents only when no newer revision exists;
- allowlisted add/remove relation corrections;
- append-only `CorrectionEvent` snapshots and per-memory revision history;
- correction API endpoints and live Neo4j edit/prune/undo coverage.

Deferred to the next specified phases: the web UI, benchmark adapters, and experiment outputs.

No deviation from the Phase 1 through 5 deliverables is known. The integration test is opt-in so `make test` stays deterministic and runnable without Docker; `make test-integration` exercises the real database when explicitly enabled. Phase 3 deliberately uses exact cosine scoring over the scope-filtered candidate set instead of a Neo4j vector index: this avoids fixing an embedding dimension in the schema and keeps provider changes reproducible on the target laptop. Phase 4 retains physical `scope_id` fields for compatibility with the common persistence schema, but VectorMemory and FlatGraphMemory never use them for retrieval validity. Each backend must use an isolated experiment repository because reset/namespacing belongs to the evaluation harness phase. Phase 5 performs no hard deletes; merge tombstones the duplicate and preserves a `SAME_AS` edge and combined provenance.

## Planned experiment outputs

Later evaluation phases will write append-only JSONL records to `results/raw/`, derived aggregates to `results/processed/`, plots to `results/figures/`, and tables to `results/tables/`. Generated outputs are ignored by Git; configurations and schemas remain versioned.

## License

MIT
