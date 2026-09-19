# Correction and Pruning Semantics

ScopeGraph corrections are soft, revisioned, and audited. Normal correction APIs never detach-delete graph data.

## Supported actions

- edit content and selected memory fields;
- move a memory to another scope;
- archive or tombstone a memory;
- restore an archive or prune event;
- merge a duplicate into a canonical memory;
- explicitly supersede an older memory;
- add or remove `SUPPORTS`, `SAME_AS`, `CONTRADICTS`, or allowlisted `RELATES_TO` edges.

Each action writes a `CorrectionEvent` containing actor, reason, before/after snapshots, and `undo_of` when applicable. `GET /memories/{id}/history` returns these immutable events in time order.

## Safe prune flow

`POST /memories/{id}/prune/preview` is read-only. It reports all bounded graph neighbors separately from directed `SUPPORTS` dependents. A dependent is proposed for `needs_review` only when the selected memory is its sole active evidentiary supporter. Other graph adjacency has no deletion semantics.

`POST /memories/{id}/prune` tombstones the selected memory and applies the previewed `needs_review` transitions. It does not delete nodes, provenance, or relationships.

`POST /memories/{id}/restore` uses the tombstone/archive event recorded by `undo_of`, or the latest applicable event. It restores dependent states only when their current revision still matches the prune event. If a newer edit exists, restoration stops instead of overwriting it.

## Merge behavior

The path memory is the duplicate source and `target_memory_id` is the canonical record. Merge copies unique `DERIVED_FROM` evidence to the target, increments both revisions, tombstones the duplicate, and creates `SAME_AS`. Both nodes and the audit event remain inspectable.
