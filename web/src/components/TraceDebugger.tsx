import { useEffect, useState, type FormEvent } from "react";

import { api } from "../api";
import type { RetrievalResult, Scope } from "../types";

export function TraceDebugger({
  scopes,
  initialScopeId,
  onResult,
  onSelectMemory,
}: {
  scopes: Scope[];
  initialScopeId: string | null;
  onResult: (result: RetrievalResult) => void;
  onSelectMemory: (memoryId: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [scopeId, setScopeId] = useState(initialScopeId ?? "");
  const [sessionId, setSessionId] = useState("");
  const [topK, setTopK] = useState("8");
  const [tokenBudget, setTokenBudget] = useState("1500");
  const [result, setResult] = useState<RetrievalResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setScopeId(initialScopeId ?? "");
  }, [initialScopeId]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    const scope = scopes.find((item) => item.id === scopeId);
    try {
      const response = await api.retrieve({
        query,
        current_scope: scope
          ? {
              id: scope.id,
              name: scope.name,
              scope_type: scope.scope_type,
              session_id: sessionId || null,
            }
          : null,
        top_k: Number(topK),
        token_budget: Number(tokenBudget),
      });
      setResult(response);
      onResult(response);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Retrieval failed");
    } finally {
      setLoading(false);
    }
  };

  const traceById = new Map(result?.trace.map((step) => [step.to_id, step]));

  return (
    <section className="trace-panel">
      <header className="trace-panel__header">
        <div>
          <span className="eyebrow">Retrieval debugger</span>
          <h2>Ask the memory graph</h2>
        </div>
        {result && (
          <div className="trace-metrics">
            <span><strong>{result.items.length}</strong> memories</span>
            <span><strong>{result.token_count}</strong> est. tokens</span>
            <span><strong>{result.retrieval_latency_ms.toFixed(1)}</strong> ms</span>
          </div>
        )}
      </header>
      <form className="trace-form" onSubmit={submit}>
        <label className="trace-form__query">
          <span>Query</span>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="What database does Beta use?" required />
        </label>
        <label><span>Scope</span><select value={scopeId} onChange={(event) => setScopeId(event.target.value)}><option value="">Global only</option>{scopes.filter((scope) => !scope.archived).map((scope) => <option value={scope.id} key={scope.id}>{scope.name}</option>)}</select></label>
        <label><span>Session ID</span><input value={sessionId} onChange={(event) => setSessionId(event.target.value)} placeholder="optional" /></label>
        <label><span>Top K</span><input type="number" min="1" max="100" value={topK} onChange={(event) => setTopK(event.target.value)} /></label>
        <label><span>Token budget</span><input type="number" min="1" value={tokenBudget} onChange={(event) => setTokenBudget(event.target.value)} /></label>
        <button className="run-query" type="submit" disabled={loading}>{loading ? "Searching…" : "Run retrieval"}<span aria-hidden="true">↗</span></button>
      </form>
      {error && <div className="trace-error" role="alert"><strong>Retrieval unavailable</strong><span>{error}</span></div>}
      {result && (
        <div className="trace-results">
          {result.items.map((item, index) => {
            const trace = traceById.get(item.memory_id);
            return (
              <article className="trace-result" key={item.memory_id}>
                <button type="button" className="trace-result__summary" onClick={() => onSelectMemory(item.memory_id)}>
                  <span className="rank">{String(index + 1).padStart(2, "0")}</span>
                  <span className="trace-result__content"><strong>{item.content}</strong><small>{item.scope_level} · {item.status} · {trace?.reason ?? "ranked evidence"}</small></span>
                  <span className="score">{item.score.toFixed(3)}</span>
                </button>
                <div className="score-grid" aria-label="Ranking score components">
                  {[
                    ["semantic", item.semantic_score],
                    ["scope", item.scope_score],
                    ["temporal", item.temporal_score],
                    ["graph", item.graph_score],
                    ["confidence", item.confidence],
                  ].map(([label, value]) => (
                    <div key={label as string}><span>{label}</span><div className="score-track"><i style={{ width: `${Number(value) * 100}%` }} /></div><b>{Number(value).toFixed(2)}</b></div>
                  ))}
                </div>
                {trace && <div className="trace-path"><span>Path</span><code>{trace.path.join(" → ")}</code><span>{trace.relation} · depth {trace.depth}</span></div>}
              </article>
            );
          })}
          {!result.items.length && <p className="empty-copy">No memory fit this query and token budget.</p>}
        </div>
      )}
    </section>
  );
}
