import { useCallback, useEffect, useMemo, useState } from "react";

import { api, graphExportUrl } from "./api";
import { GraphCanvas } from "./components/GraphCanvas";
import { Inspector, type CorrectionAction } from "./components/Inspector";
import { ScopeTree } from "./components/ScopeTree";
import { TraceDebugger } from "./components/TraceDebugger";
import type {
  CorrectionEvent,
  GraphNode,
  GraphSubgraph,
  Memory,
  MemoryProvenance,
  MemoryStats,
  PrunePreview,
  RetrievalResult,
  Scope,
} from "./types";

const EMPTY_STATS: MemoryStats = {
  backend_name: "scopegraph",
  scope_count: 0,
  session_count: 0,
  source_message_count: 0,
  memory_count: 0,
  relationship_count: 0,
};

export function App() {
  const [scopes, setScopes] = useState<Scope[]>([]);
  const [stats, setStats] = useState<MemoryStats>(EMPTY_STATS);
  const [health, setHealth] = useState<"checking" | "up" | "down">("checking");
  const [graph, setGraph] = useState<GraphSubgraph | null>(null);
  const [selectedScopeId, setSelectedScopeId] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [memory, setMemory] = useState<Memory | null>(null);
  const [provenance, setProvenance] = useState<MemoryProvenance | null>(null);
  const [history, setHistory] = useState<CorrectionEvent[]>([]);
  const [retrievedIds, setRetrievedIds] = useState<Set<string>>(new Set());
  const [includeInactive, setIncludeInactive] = useState(true);
  const [includeSources, setIncludeSources] = useState(false);
  const [graphLoading, setGraphLoading] = useState(true);
  const [correctionBusy, setCorrectionBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadHeader = useCallback(async () => {
    try {
      const [scopeData, statData, healthData] = await Promise.all([
        api.scopes(),
        api.stats(),
        api.health(),
      ]);
      setScopes(scopeData);
      setStats(statData);
      setHealth(healthData.neo4j === "up" ? "up" : "down");
    } catch (caught) {
      setHealth("down");
      setError(caught instanceof Error ? caught.message : "API unavailable");
    }
  }, []);

  const loadGraph = useCallback(async () => {
    setGraphLoading(true);
    try {
      const response = await api.graph({
        scopeId: selectedScopeId ?? undefined,
        includeInactive,
        includeSources,
      });
      setGraph(response);
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Graph could not be loaded");
    } finally {
      setGraphLoading(false);
    }
  }, [includeInactive, includeSources, selectedScopeId]);

  const loadMemory = useCallback(async (id: string) => {
    try {
      const [record, provenanceData, historyData] = await Promise.all([
        api.memory(id),
        api.provenance(id),
        api.history(id),
      ]);
      setMemory(record);
      setProvenance(provenanceData);
      setHistory(historyData);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Memory details unavailable");
    }
  }, []);

  useEffect(() => {
    void loadHeader();
  }, [loadHeader]);

  useEffect(() => {
    void loadGraph();
  }, [loadGraph]);

  const selectedNode = useMemo(
    () => graph?.nodes.find((node) => node.id === selectedNodeId) ?? null,
    [graph, selectedNodeId],
  );

  const selectNode = useCallback(
    (nodeId: string) => {
      setSelectedNodeId(nodeId);
      const node = graph?.nodes.find((item) => item.id === nodeId);
      if (node?.node_type === "memory") {
        void loadMemory(nodeId);
      } else {
        setMemory(null);
        setProvenance(null);
        setHistory([]);
        if (node?.node_type === "scope") setSelectedScopeId(nodeId);
      }
    },
    [graph, loadMemory],
  );

  const selectScope = (scopeId: string | null) => {
    setSelectedScopeId(scopeId);
    setSelectedNodeId(scopeId);
    setMemory(null);
    setProvenance(null);
    setHistory([]);
  };

  const correct = async (action: CorrectionAction, payload: Record<string, unknown>) => {
    if (!memory) throw new Error("Select a memory first");
    setCorrectionBusy(true);
    try {
      const result =
        action === "edit"
          ? await api.edit(memory.id, payload)
          : action === "move"
            ? await api.move(memory.id, payload)
            : action === "archive"
              ? await api.archive(memory.id, payload)
              : action === "prune"
                ? await api.prune(memory.id, payload)
                : action === "restore"
                  ? await api.restore(memory.id, payload)
                  : await api.merge(memory.id, payload);
      setSelectedNodeId(result.memory.id);
      setSelectedScopeId(result.memory.scope_id);
      const [refreshedGraph] = await Promise.all([
        api.graph({
          scopeId: result.memory.scope_id,
          includeInactive,
          includeSources,
        }),
        loadMemory(result.memory.id),
        loadHeader(),
      ]);
      setGraph(refreshedGraph);
    } finally {
      setCorrectionBusy(false);
    }
  };

  const previewPrune = (): Promise<PrunePreview> => {
    if (!memory) return Promise.reject(new Error("Select a memory first"));
    return api.prunePreview(memory.id);
  };

  const onRetrieval = (result: RetrievalResult) => {
    setRetrievedIds(new Set(result.items.map((item) => item.memory_id)));
  };

  const selectRetrievedMemory = useCallback(
    async (memoryId: string) => {
      setSelectedNodeId(memoryId);
      try {
        const focusedGraph = await api.graph({
          memoryId,
          includeInactive: true,
          includeSources,
        });
        setGraph(focusedGraph);
        await loadMemory(memoryId);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Memory details unavailable");
      }
    },
    [includeSources, loadMemory],
  );

  const relationships = graph?.edges.filter(
    (edge) => edge.source === selectedNodeId || edge.target === selectedNodeId,
  ) ?? [];
  const scopeLabel = scopes.find((scope) => scope.id === selectedScopeId)?.name ?? "All memory";

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="window-controls" aria-hidden="true"><i /><i /><i /></div>
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">✳</span>
          <div><strong>scopegraph</strong><span>~/memory-explorer</span></div>
        </div>
        <nav className="product-nav" aria-label="Product navigation">
          <button className="product-nav__item is-active" type="button">/graph</button>
          <button className="product-nav__item" type="button">/retrieve</button>
          <button className="product-nav__item" type="button">/history</button>
        </nav>
        <div className="system-stats" aria-label="System statistics">
          <span><b>{stats.scope_count}</b> scopes</span>
          <span><b>{stats.memory_count}</b> memories</span>
          <span><b>{stats.relationship_count}</b> edges</span>
        </div>
        <div className="topbar__runtime">
          <span className="model-pill"><span className="model-pill__dot" />claude-code · local</span>
          <div className={`health health--${health}`}>
            <span aria-hidden="true" />
            Neo4j {health === "checking" ? "checking" : health}
          </div>
        </div>
      </header>

      {error && <div className="global-error" role="alert"><strong>Connection issue</strong><span>{error}</span><button type="button" onClick={() => { setError(null); void loadHeader(); void loadGraph(); }}>Retry</button></div>}

      <main className="workspace">
        <aside className="panel scope-panel">
          <div className="panel-heading"><div><span className="eyebrow">workspace</span><h2><span className="prompt-char">›</span> scope tree</h2></div><span className="count-pill">{scopes.length}</span></div>
          <ScopeTree scopes={scopes} selectedScopeId={selectedScopeId} onSelect={selectScope} />
          <footer className="scope-legend">
            <h3>legend</h3>
            <span><i className="legend-dot legend-dot--session" /> Session</span>
            <span><i className="legend-dot legend-dot--scope" /> Context</span>
            <span><i className="legend-dot legend-dot--global" /> Global</span>
            <small>Status is also written on every memory node.</small>
          </footer>
        </aside>

        <section className="panel graph-panel">
          <header className="graph-toolbar">
            <div><span className="eyebrow">scopegraph inspect --live</span><h1><span className="prompt-char">›</span> {scopeLabel}</h1><span className="graph-subtitle">bounded traversal · provenance on · corrections reversible</span></div>
            <div className="graph-toolbar__actions">
              <label className="toggle"><input type="checkbox" checked={includeInactive} onChange={(event) => setIncludeInactive(event.target.checked)} /><span />Inactive</label>
              <label className="toggle"><input type="checkbox" checked={includeSources} onChange={(event) => setIncludeSources(event.target.checked)} /><span />Sources</label>
              <a className="export-link" href={graphExportUrl(selectedScopeId)}>export.json</a>
              <button className="icon-button" type="button" onClick={() => void loadGraph()} aria-label="Refresh graph">↻</button>
            </div>
          </header>
          <div className={`graph-wrap ${graphLoading ? "is-loading" : ""}`}>
            {graphLoading && <div className="loading-scrim"><span />Loading graph…</div>}
            <GraphCanvas graph={graph} selectedNodeId={selectedNodeId} retrievedIds={retrievedIds} onSelectNode={selectNode} />
          </div>
          <footer className="graph-footer"><span><b>{graph?.nodes.length ?? 0}</b> nodes</span><span><b>{graph?.edges.length ?? 0}</b> edges</span><span className="graph-footer__hint">Scroll to zoom · drag to pan · click any node to inspect</span></footer>
        </section>

        <aside className="panel inspector-panel">
          <div className="panel-heading panel-heading--sticky"><div><span className="eyebrow">selected node</span><h2><span className="prompt-char">›</span> inspector</h2></div>{selectedNode && <span className="node-kind">{selectedNode.node_type.replace("_", " ")}</span>}</div>
          <Inspector node={selectedNode} memory={memory} provenance={provenance} history={history} relationships={relationships} scopes={scopes} busy={correctionBusy} onCorrection={correct} onPrunePreview={previewPrune} onSelectNode={selectNode} />
        </aside>
      </main>

      <TraceDebugger scopes={scopes} initialScopeId={selectedScopeId} onResult={onRetrieval} onSelectMemory={(id) => void selectRetrievedMemory(id)} />
      <footer className="app-footer"><span><b>✳</b> scopegraph research terminal</span><span>esc clear · / help · neo4j connected</span></footer>
    </div>
  );
}
