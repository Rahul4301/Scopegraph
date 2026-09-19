# ScopeGraph

ScopeGraph is a research system for testing whether explicit session, project/context, and global memory scopes reduce cross-context retrieval errors in long-running LLM agents. It also tests whether editing persistent memory directly produces more durable corrections than adding a conversational correction.

The repository currently contains Phases 1 and 2: typed domain models, configuration, Neo4j persistence, initial FastAPI routes, and the complete provenance-aware write path from session ingestion through conservative promotion. It does not claim experimental results yet.

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

See [docs/architecture.md](docs/architecture.md), [docs/schema.md](docs/schema.md), and [docs/literature.md](docs/literature.md).

## Requirements

- Python 3.11+
- `uv`
- Docker Desktop or another Docker Compose runtime
- hosted LLM and embedding API credentials only when later phases need live model calls

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

Deferred to the next specified phases: embeddings and retrieval, baselines, correction workflows, the web UI, benchmark adapters, and experiment outputs.

No deviation from the Phase 1 or Phase 2 deliverables is known. The integration test is opt-in so `make test` stays deterministic and runnable without Docker; `make test-integration` exercises the real database when explicitly enabled. Automatic cross-scope promotion deliberately requires semantically equivalent normalized statements; it does not infer a global preference from unrelated project subjects. More ambitious generalization remains a later research policy rather than an untracked LLM inference.

## Planned experiment outputs

Later evaluation phases will write append-only JSONL records to `results/raw/`, derived aggregates to `results/processed/`, plots to `results/figures/`, and tables to `results/tables/`. Generated outputs are ignored by Git; configurations and schemas remain versioned.

## License

MIT
