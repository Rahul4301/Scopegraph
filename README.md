# ScopeGraph

ScopeGraph is a research system for testing whether explicit session, project/context, and global memory scopes reduce cross-context retrieval errors in long-running LLM agents. It also tests whether editing persistent memory directly produces more durable corrections than adding a conversational correction.

The repository contains typed domain models, Neo4j persistence, provenance-aware ScopeGraph write/read paths, reversible audited corrections, the Memory Explorer, a synthetic evaluation harness, and local-file external benchmark adapters. It does not claim complete external benchmark results yet.

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

See [docs/architecture.md](docs/architecture.md), [docs/corrections.md](docs/corrections.md), [docs/schema.md](docs/schema.md), [docs/literature.md](docs/literature.md), and the latest [audit](docs/audit.md).

## Requirements

- Python 3.11+
- `uv`
- Docker Desktop or another Docker Compose runtime
- Node.js 20+ and npm for the Memory Explorer
- OpenAI-compatible LLM and embedding credentials for live extraction and retrieval

The Neo4j container is deliberately limited to a 512 MB heap and 256 MB page cache for Apple M1 machines with 8 GB RAM.

## Setup

```bash
cp .env.example .env
uv sync --extra dev
npm --prefix web install
docker compose up -d neo4j
uv run python scripts/setup_neo4j.py
```

The default local credentials are development-only. Change `NEO4J_PASSWORD` and the matching `NEO4J_AUTH` value before exposing Neo4j beyond localhost.

## Run

```bash
uv run uvicorn scopegraph.api.main:app --reload
npm --prefix web run dev
```

Open `http://127.0.0.1:5173` for the Memory Explorer and `http://127.0.0.1:8000/docs` for generated API documentation. Vite proxies `/api` to the local FastAPI process. `GET /health` checks Neo4j connectivity without exposing credentials.

`POST /retrieve` accepts a query, optional current scope and session, top-k, token budget, and optional evaluation timestamp. Retrieval searches the current session and scope, then ancestors and global memory; an unrelated scope is included only when its name appears in the query. The response includes each score component and traversal path.

Memory correction routes are grouped under `/memories/{id}`. Use `/prune/preview` before `/prune`; archive and prune operations can be reversed with `/restore`, and `/history` returns the append-only audit trail. `PATCH /memories/{id}` is an audited edit rather than an untracked property mutation.

The explorer renders the scope hierarchy and typed memory graph, exposes source-message provenance and revision history, and drives the same audited edit, move, archive, prune, restore, and merge routes used by automated experiments. Its retrieval debugger shows ranking components and traversal paths. The browser has no arbitrary Cypher endpoint.

## Quality checks

```bash
make check
npm --prefix web run build
```

Run the credential-free end-to-end smoke test:

```bash
make smoke
```

For a Neo4j-backed demo, start the database and apply the schema first:

```bash
make neo4j-up
make migrate
make demo
make export-graph OUTPUT=results/graph.json
```

`make reset-db` is an explicit destructive development reset. It requires no model API keys; live extraction and answer generation do require the provider variables in `.env`.

Unit tests and `make smoke` use an in-memory repository as a deterministic test double.
Research-facing evaluation commands use the isolated Neo4j evaluation service. The
live Neo4j integration test is opt-in:

```bash
SCOPEGRAPH_RUN_INTEGRATION=1 uv run pytest -m integration
```

This is a research prototype, not an internet-facing production service. The API
does not yet provide authentication, tenant isolation, quotas, distributed workers,
or a production migration/backup policy. Keep Neo4j and the API bound to trusted
local infrastructure until those controls are added.

## Configuration

`.env.example` contains all runtime secrets and provider settings. Versioned YAML files under `configs/` hold retrieval weights, memory policies, experiment controls, and logging configuration. Model names and API endpoints are never hardcoded into the memory engine.

## Phase status

Complete in Phase 1:

- repository structure and project tooling;
- Pydantic v2 models for scopes, sessions, source messages, memories, relationships, corrections, and retrieval traces;
- a `MemorySystem` interface for the ScopeGraph service;
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

Complete in Phase 5:

- audited memory edits with revision increments and embedding invalidation;
- scope moves, archive, tombstone, restore, explicit supersession, and duplicate merge;
- dry-run prune previews separating graph neighbors from evidentiary dependents;
- dependency-safe pruning that marks only unsupported active dependents `needs_review`;
- guarded undo that restores affected dependents only when no newer revision exists;
- allowlisted add/remove relation corrections;
- append-only `CorrectionEvent` snapshots and per-memory revision history;
- correction API endpoints and live Neo4j edit/prune/undo coverage.

Complete in Phase 6:

- a responsive React Memory Explorer with scope tree, Cytoscape graph, and node inspector;
- explicit graph subgraph, JSON/GraphML export, statistics, and provenance endpoints;
- source-message provenance, incoming/outgoing relationships, and revision history;
- edit, move, archive, prune-preview, restore, and merge dialogs backed by audited APIs;
- a retrieval trace debugger with per-component scores, traversal path, latency, and token use;
- redundant text, shape, border, and color status cues for accessibility;
- frontend type-checking and production build verification plus live Neo4j graph-query coverage.

Complete in Phase 7:

- deterministic CrossScopeMem generation for test-only retrieval diagnostics;
- retrieval, answer, contamination, stale-memory, latency, token, storage, scope-classification, and correction metrics;
- JSONL records, aggregation, Markdown tables, and SVG plots.

Complete in Phase 8:

- complete official adapters for LongMemEval-S, LoCoMo, and MemoryAgentBench;
- selective API acquisition with pinned checksums and no repository snapshots;
- official deterministic scorers and pinned LLM judges;
- full ScopeGraph runs on all official external questions; architecture controls and
  component ablations run separately on account-shaped CrossScopeMem scenarios.

Phase 9 reproducibility tooling is included through `make smoke`, `make migrate`, `make demo`, `make export-graph`, resumable benchmark checkpoints, bootstrap confidence intervals, and the documented full-check workflow. No full live released-dataset result is claimed until `make eval-suite` completes.

The integration test is opt-in so `make test` stays deterministic and runnable without Docker; `make test-integration` exercises the real database when explicitly enabled. Phase 3 deliberately uses exact cosine scoring over the scope-filtered candidate set instead of a Neo4j vector index: this avoids fixing an embedding dimension in the schema and keeps provider changes reproducible on the target laptop, but it remains a production-scale limitation for very large individual scopes. Phase 5 performs no hard deletes; merge tombstones the duplicate and preserves a `SAME_AS` edge and combined provenance. Phase 6 keeps graph reads behind bounded, fixed repository queries and limits exports to 500 memories; it never exposes arbitrary browser-authored Cypher.

## Experiment outputs

Evaluation writes append-only JSONL records to `results/raw/`, derived aggregates to `results/processed/`, plots to `results/figures/`, and tables to `results/tables/`. Generated outputs are ignored by Git; configurations and schemas remain versioned.

## License

MIT
