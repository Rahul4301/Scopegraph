# Memory Explorer UI

Phase 6 implements the Memory Explorer as a React, TypeScript, Vite, and Cytoscape application in `web/`.

## Run locally

Start Neo4j and the schema, then run the API and UI in separate terminals:

```bash
make neo4j-up
make schema
make api
```

```bash
make web-install
make web-dev
```

Open `http://127.0.0.1:5173`. The development server proxies `/api` to `http://127.0.0.1:8000`; set `VITE_API_BASE_URL` only when the API is hosted elsewhere.

## Explorer behavior

- The scope tree filters visible memories while retaining the hierarchy for orientation.
- The graph distinguishes scope, memory, and source-message nodes and shows typed directed edges.
- Selecting a memory loads its full record, source-message provenance, incoming/outgoing relationships, and append-only correction history.
- Correction dialogs support edit, move, archive, prune preview and confirmation, restore, and duplicate merge.
- The trace debugger submits a real retrieval request and shows the selected evidence, total latency/token use, component scores, reason, and traversal path. Selecting a result centers inspection on that memory.
- GraphML export uses the currently selected scope. The JSON form remains available from `GET /graph/export?format=json`.

Statuses are always written as text and additionally encoded through node borders, opacity, and color. The UI does not rely on color alone.

## API boundary

The UI uses the explicit `/graph/subgraph`, `/graph/export`, `/stats`, `/memories/{id}/provenance`, correction, and retrieval endpoints. Subgraph reads are bounded to 500 memories and use fixed allowlisted relationship types. No arbitrary Cypher is accepted from the browser.

## Verification

```bash
make check
make test-integration
make web-build
```

Offline API tests use the in-memory repository. The integration suite exercises the graph slice and source provenance against live Neo4j, and the frontend production build runs strict TypeScript checking.
