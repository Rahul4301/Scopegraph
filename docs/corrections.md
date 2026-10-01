# Correction Semantics

ScopeGraph corrections are soft, revisioned, and audited. Normal correction APIs never detach-delete graph data.

## Supported actions

- edit content and selected memory fields;
- move a memory to another scope;
- archive a memory;
- restore an archived (or merge-tombstoned) memory;
- merge a duplicate into a canonical memory;
- explicitly supersede an older memory;
- add or remove `SUPPORTS`, `SAME_AS`, `CONTRADICTS`, or allowlisted `RELATES_TO` edges.

Each action writes a `CorrectionEvent` containing actor, reason, before/after snapshots, and `undo_of` when applicable. `GET /memories/{id}/history` returns these immutable events in time order.

## No manual prune

There is no user-facing prune: `POST /memories/{id}/prune` and `/prune/preview` and `CorrectionService.prune` were removed, and a `tombstone` correction request is rejected. Memories leave retrieval only by being archived, superseded, or merged away; none are deleted. Tombstoning now happens only as part of a merge. The explorer web app still shows prune controls; they no longer work.

`POST /memories/{id}/restore` uses the archive (or merge-tombstone) event recorded by `undo_of`, or the latest applicable event, and restores dependent states only when their current revision still matches that event. If a newer edit exists, restoration stops instead of overwriting it.

## Merge behavior

The path memory is the duplicate source and `target_memory_id` is the canonical record. Merge copies unique `DERIVED_FROM` evidence to the target, increments both revisions, tombstones the duplicate, and creates `SAME_AS`. Both nodes and the audit event remain inspectable.

## Removed memories and raw source search

Retrieval also searches original source turns. A turn whose memories are all archived or tombstoned is hidden from that search (and from neighbouring-turn context), so archiving a wrong memory removes the fact from retrieval rather than letting it return through its source message. Merging keeps the turn visible because the canonical memory inherits its provenance. Historical queries do not bring archived turns back through raw search; the archived memory itself can still appear.
