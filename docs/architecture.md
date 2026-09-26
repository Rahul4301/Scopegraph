# ScopeGraph Architecture

## Purpose and research boundary

ScopeGraph is a model-agnostic memory service built to test two claims rather than assume them: whether explicit contextual scope reduces retrieval interference, and whether structural correction persists better than conversational correction. Live evaluations run the ScopeGraph implementation against released benchmark questions and report failures as well as successes.

## Memory hierarchy

A workspace has exactly one active global root. Context scopes such as projects, repositories, courses, clients, tasks, and workspaces form a tree below it. A session belongs to one scope. Each memory declares both its containing scope and one logical level:

```text
global root
└── project or other context scope
    └── nested task scope
        └── current session
```

Session memories capture temporary state. Scope memories persist across sessions inside a bounded context. Global memories are intended to remain valid across contexts. Scope specificity is therefore an applicability constraint, not merely another similarity feature.

## Write path

The write path stores raw evidence separately from compact memories:

```text
Scope -> Session -> SourceMessage
                    |
                    +-> Memory -> Scope
```

Phase 2 validates structured extraction, normalizes candidate facts, suppresses duplicates, detects conflicts, and attaches provenance before persistence. Low-durability information remains session-level. Durable information can consolidate into the explicit current scope. Global storage requires an explicit general statement or equivalent evidence across the configured number of distinct scopes. Arbitrary model-generated Neo4j relationship types are forbidden. Semantic links use `RELATES_TO.kind` from an allowlist.

Conflicts never erase prior state. A new active memory points to the old memory with `SUPERSEDES` and `CONTRADICTS`; the old memory becomes superseded and receives a temporal validity end. Repeated cross-scope evidence creates a separate global memory supported by the contributing scope memories and their source messages.

## Read path

The Phase 3 retriever resolves the active scope, embeds the query, finds semantic anchors only within permitted scopes, performs bounded traversal, removes inactive or temporally invalid evidence, ranks the remaining memories, and packs each selected memory with its directly linked source messages into a shared token budget. Search eligibility follows current session, current scope, ancestor scopes, then global memory. A sibling scope is excluded unless its name is explicitly present in the query.

Embeddings are requested through an OpenAI-compatible provider. A local SQLite cache is keyed by the embedding model and content hash, and generated vectors are also persisted on memory nodes. Anchor similarity currently uses exact cosine scoring over the already scope-filtered candidate set. This is intentional for provider-independent embedding dimensions and the resource-constrained reference environment; a Neo4j vector index remains an optimization to evaluate at larger scale.

Graph expansion follows only `SUPERSEDES`, `CONTRADICTS`, `SAME_AS`, `SUPPORTS`, and `RELATES_TO`. It is capped by configurable hop and node limits, a wall-clock budget, cycle detection, and the same scope allowlist. Current-state queries exclude inactive or out-of-validity memories. Historical-language queries may include superseded and archived evidence and favor superseded state.

Final ranking combines configurable semantic, scope, temporal, confidence, graph-proximity, and recency components. Ranked summaries and non-duplicated verbatim provenance are packed without reordering until the request token budget is exhausted. This permits exact episodic answers without treating an entire ingested conversation as answer context.

Every result exposes component scores, source IDs, scope IDs, temporal validity, graph traversal paths, selection reasons, latency, and estimated token count. This white-box trace keeps retrieval quality measurable independently of answer generation.

## Correction path

Normal correction never hard-deletes a memory. An edit, move, archive, merge, tombstone, restore, supersession, or relation change creates an append-only `CorrectionEvent` containing before and after state, actor, reason, and optional undo target. Every changed memory increments its revision; content edits also clear the stored embedding so retrieval regenerates it.

Prune preview distinguishes graph neighbors from directed evidentiary dependencies. A target's outgoing `SUPPORTS` edges identify possible dependents. Independently supported memories remain unchanged, while active memories losing their only active support become `needs_review`. Confirmed pruning tombstones only the selected target. Restore is guarded by recorded revision numbers, so undo cannot silently overwrite a newer correction.

Merging copies source-message provenance to the canonical target, tombstones the duplicate, and retains `SAME_AS`; it never deletes either memory. Semantic relation edits are restricted to fixed relationship types and the existing `RELATES_TO.kind` allowlist.

## Inspection surface

The React Memory Explorer consumes explicit FastAPI schemas for graph slices, provenance, correction history, statistics, and retrieval traces. The browser can filter a bounded subgraph by scope or center it on one memory, but cannot submit Cypher. Memory and source-message nodes retain their full typed API record for inspection; corrections always pass through the audited correction service.

Status is communicated with written labels plus node shape, border pattern, and color. This makes active, superseded, archived, tombstoned, and review-required state distinguishable without relying on color alone.

## Implemented components

The Python package contains validated domain models, configuration loading, an asynchronous Neo4j client, schema creation, CRUD repositories, structured extraction, scope resolution, provenance-aware ingestion, consolidation, conflict handling, promotion, embeddings, scoped retrieval, bounded traversal, temporal filtering, ranking, token packing, retrieval traces, reversible correction workflows, graph inspection/export queries, and FastAPI routes. The React application provides the scope tree, graph explorer, inspector, correction dialogs, and retrieval debugger. The in-memory repository, static extractor, and deterministic test embedder keep tests credential-free; Neo4j and the OpenAI-compatible providers are the production paths.
