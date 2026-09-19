# Graph Schema

## Nodes

| Label | Purpose | Key properties |
| --- | --- | --- |
| `Scope` | Global root and bounded contexts | `id`, `name`, `scope_type`, `parent_scope_id`, `created_at`, `archived` |
| `Session` | One interaction period | `id`, `started_at`, `ended_at`, `scope_id`, `metadata_json` |
| `SourceMessage` | Immutable raw conversational evidence | `id`, `session_id`, `role`, `content`, `timestamp`, `turn_index` |
| `Memory` | Compact retrievable state | `id`, `content`, `memory_type`, `scope_level`, `scope_id`, confidence, status, validity dates, embedding/model, revision, metadata |
| `CorrectionEvent` | Append-only mutation audit | `id`, action, timestamp, actor, before/after JSON, reason, undo target |
| `ConsolidationRun` | Audited consolidation execution | `id`, timestamps, policy version, dry-run flag, stats JSON |

Identifiers are unique. Lookup indexes cover scope type and parent, session/source scope, memory scope, status, type, and update time. A full-text memory-content index supports a lexical fallback. Phase 3 stores provider-generated vectors directly on memory nodes and uses exact cosine ranking after scope filtering. A vector index is intentionally deferred until an embedding dimension is selected and scale measurements justify the provider-specific schema.

## Relationships

The schema uses `PARENT_OF`, `BELONGS_TO`, `PART_OF`, `DERIVED_FROM`, `SUPERSEDES`, `CONTRADICTS`, `SAME_AS`, `SUPPORTS`, `RELATES_TO`, `TARGETED`, and `TOUCHED`. `RELATES_TO.kind` is restricted to the application allowlist; untrusted text never becomes a Cypher relationship type.

## Temporal and revision semantics

Active, superseded, archived, tombstoned, and needs-review are explicit states. `valid_from` and `valid_to` describe when a fact applies, while `created_at` and `updated_at` describe record history. Mutations increment `revision`. Raw source messages remain separate and linked through `DERIVED_FROM`, allowing a compact memory to retain inspectable evidence.

During Phase 2 consolidation, normalized subject and predicate metadata identifies likely changing facts. A replacement marks the prior node superseded and creates both `SUPERSEDES` and `CONTRADICTS` edges while retaining its source evidence. Cross-scope promotion creates a distinct global node and connects contributing scope memories with `SUPPORTS`.

Phase 5 corrections increment the affected memory revision and append a `CorrectionEvent` with complete before/after JSON snapshots. `TARGETED` connects an event to every memory it changed. Events are never overwritten. History is therefore inspectable even though the current `Memory` node remains the efficient read model. Content edits remove stale embedding properties; moves update both `scope_id` and `BELONGS_TO`; merges copy `DERIVED_FROM` provenance, tombstone the duplicate, and retain `SAME_AS`.
