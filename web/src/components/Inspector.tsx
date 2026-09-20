import { useEffect, useState, type FormEvent } from "react";

import type {
  CorrectionEvent,
  GraphEdge,
  GraphNode,
  Memory,
  MemoryProvenance,
  PrunePreview,
  Scope,
} from "../types";
import { Modal } from "./Modal";

export type CorrectionAction = "edit" | "move" | "archive" | "prune" | "restore" | "merge";

interface InspectorProps {
  node: GraphNode | null;
  memory: Memory | null;
  provenance: MemoryProvenance | null;
  history: CorrectionEvent[];
  relationships: GraphEdge[];
  scopes: Scope[];
  busy: boolean;
  onCorrection: (action: CorrectionAction, payload: Record<string, unknown>) => Promise<void>;
  onPrunePreview: () => Promise<PrunePreview>;
  onSelectNode: (nodeId: string) => void;
}

const pretty = (value: string) => value.replaceAll("_", " ");
const shortId = (value: string) => (value.length > 22 ? `${value.slice(0, 10)}…${value.slice(-7)}` : value);

export function Inspector({
  node,
  memory,
  provenance,
  history,
  relationships,
  scopes,
  busy,
  onCorrection,
  onPrunePreview,
  onSelectNode,
}: InspectorProps) {
  const [dialog, setDialog] = useState<CorrectionAction | null>(null);
  const [content, setContent] = useState("");
  const [confidence, setConfidence] = useState("1");
  const [scopeId, setScopeId] = useState("");
  const [targetId, setTargetId] = useState("");
  const [reason, setReason] = useState("");
  const [preview, setPreview] = useState<PrunePreview | null>(null);
  const [dialogError, setDialogError] = useState<string | null>(null);

  useEffect(() => {
    if (!memory) return;
    setContent(memory.content);
    setConfidence(String(memory.confidence));
    setScopeId(memory.scope_id);
    setTargetId("");
    setReason("");
    setDialog(null);
    setPreview(null);
  }, [memory]);

  const openPrune = async () => {
    setDialog("prune");
    setPreview(null);
    setDialogError(null);
    try {
      setPreview(await onPrunePreview());
    } catch (error) {
      setDialogError(error instanceof Error ? error.message : "Preview failed");
    }
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!dialog) return;
    setDialogError(null);
    const payload: Record<string, unknown> = { actor: "memory-explorer", reason };
    if (dialog === "edit") {
      payload.content = content;
      payload.confidence = Number(confidence);
    } else if (dialog === "move") {
      payload.scope_id = scopeId;
    } else if (dialog === "merge") {
      payload.target_memory_id = targetId;
    }
    try {
      await onCorrection(dialog, payload);
      setDialog(null);
    } catch (error) {
      setDialogError(error instanceof Error ? error.message : "Correction failed");
    }
  };

  if (!node) {
    return (
      <div className="inspector-empty">
        <div className="inspector-empty__icon">◎</div>
        <h3>Select a node</h3>
        <p>Inspect state, provenance, relationships, and revision history.</p>
      </div>
    );
  }

  if (node.node_type !== "memory" || !memory) {
    return (
      <div className="inspector-content">
        <div className="inspector-heading">
          <span className="eyebrow">{pretty(node.node_type)}</span>
          <h2>{node.label}</h2>
          <code>{node.id}</code>
        </div>
        <section className="inspector-section">
          <h3>Node properties</h3>
          <dl className="property-list">
            {Object.entries(node.data).map(([key, value]) => (
              <div key={key}>
                <dt>{pretty(key)}</dt>
                <dd>{typeof value === "object" ? JSON.stringify(value) : String(value)}</dd>
              </div>
            ))}
          </dl>
        </section>
      </div>
    );
  }

  const canRestore = memory.status === "archived" || memory.status === "tombstoned";
  const activeScope = scopes.find((scope) => scope.id === memory.scope_id);
  const incoming = relationships.filter((edge) => edge.target === memory.id);
  const outgoing = relationships.filter((edge) => edge.source === memory.id);

  return (
    <div className="inspector-content">
      <div className="inspector-heading">
        <div className="inspector-heading__row">
          <span className={`status-badge status-badge--${memory.status}`}>
            {pretty(memory.status)}
          </span>
          <span className={`level-badge level-badge--${memory.scope_level}`}>
            {memory.scope_level}
          </span>
        </div>
        <h2>{memory.content}</h2>
        <code title={memory.id}>{shortId(memory.id)}</code>
      </div>

      <div className="action-grid">
        <button type="button" onClick={() => setDialog("edit")}>Edit</button>
        <button type="button" onClick={() => setDialog("move")}>Move</button>
        <button type="button" onClick={() => setDialog("merge")}>Merge</button>
        {canRestore ? (
          <button type="button" className="action-primary" onClick={() => setDialog("restore")}>
            Restore
          </button>
        ) : (
          <>
            <button type="button" onClick={() => setDialog("archive")}>Archive</button>
            <button type="button" className="action-danger" onClick={openPrune}>Prune</button>
          </>
        )}
      </div>

      <section className="inspector-section">
        <h3>Memory record</h3>
        <dl className="property-list">
          <div><dt>Scope</dt><dd>{activeScope?.name ?? memory.scope_id}</dd></div>
          <div><dt>Type</dt><dd>{pretty(memory.memory_type)}</dd></div>
          <div><dt>Confidence</dt><dd>{(memory.confidence * 100).toFixed(0)}%</dd></div>
          <div><dt>Revision</dt><dd>r{memory.revision}</dd></div>
          <div><dt>Updated</dt><dd>{new Date(memory.updated_at).toLocaleString()}</dd></div>
          <div><dt>Embedding</dt><dd>{memory.embedding_model ?? "not generated"}</dd></div>
        </dl>
      </section>

      <section className="inspector-section">
        <h3>Provenance <span className="count-pill">{provenance?.source_messages.length ?? 0}</span></h3>
        {provenance?.source_messages.length ? (
          <div className="source-list">
            {provenance.source_messages.map((source) => (
              <article key={source.id} className="source-card">
                <div><span className="tag">{source.role}</span><time>{new Date(source.timestamp).toLocaleString()}</time></div>
                <p>{source.content}</p>
                <code>{shortId(source.id)}</code>
              </article>
            ))}
          </div>
        ) : <p className="empty-copy">No source messages attached.</p>}
      </section>

      <section className="inspector-section">
        <h3>Relationships <span className="count-pill">{relationships.length}</span></h3>
        {[...outgoing.map((edge) => ({ edge, direction: "out" })), ...incoming.map((edge) => ({ edge, direction: "in" }))].map(({ edge, direction }) => {
          const other = direction === "out" ? edge.target : edge.source;
          return (
            <button className="relation-row" type="button" key={`${direction}-${edge.id}`} onClick={() => onSelectNode(other)}>
              <span className="relation-direction">{direction === "out" ? "→" : "←"}</span>
              <span><strong>{edge.kind ?? edge.relation}</strong><small>{shortId(other)}</small></span>
            </button>
          );
        })}
        {!relationships.length && <p className="empty-copy">No visible relationships.</p>}
      </section>

      <section className="inspector-section">
        <h3>Revision history <span className="count-pill">{history.length}</span></h3>
        <div className="history-list">
          {[...history].reverse().map((event) => (
            <details key={event.id} className="history-event">
              <summary>
                <span className="history-dot" aria-hidden="true" />
                <span><strong>{pretty(event.action)}</strong><small>{new Date(event.timestamp).toLocaleString()} · {event.actor}</small></span>
              </summary>
              {event.reason && <p>{event.reason}</p>}
              <pre>{JSON.stringify({ before: event.before, after: event.after }, null, 2)}</pre>
            </details>
          ))}
        </div>
        {!history.length && <p className="empty-copy">No corrections recorded.</p>}
      </section>

      {dialog && (
        <Modal title={{ edit: "Edit memory", move: "Move memory", archive: "Archive memory", prune: "Prune memory", restore: "Restore memory", merge: "Merge duplicate" }[dialog]} onClose={() => setDialog(null)}>
          <form onSubmit={submit} className="correction-form">
            {dialog === "edit" && (
              <>
                <label>Content<textarea value={content} onChange={(event) => setContent(event.target.value)} rows={5} required /></label>
                <label>Confidence<input type="number" min="0" max="1" step="0.01" value={confidence} onChange={(event) => setConfidence(event.target.value)} required /></label>
              </>
            )}
            {dialog === "move" && (
              <label>Destination scope<select value={scopeId} onChange={(event) => setScopeId(event.target.value)} required>{scopes.filter((scope) => !scope.archived).map((scope) => <option value={scope.id} key={scope.id}>{scope.name} · {scope.scope_type}</option>)}</select></label>
            )}
            {dialog === "merge" && (
              <label>Canonical target memory ID<input value={targetId} onChange={(event) => setTargetId(event.target.value)} placeholder="memory UUID" required /></label>
            )}
            {dialog === "archive" && <p className="dialog-callout">The memory will leave current retrieval but remain fully inspectable and restorable.</p>}
            {dialog === "restore" && <p className="dialog-callout">ScopeGraph will restore the latest archive or prune event only if no newer revision would be overwritten.</p>}
            {dialog === "prune" && (
              <div className="prune-preview">
                {!preview && !dialogError && <p>Calculating dependency impact…</p>}
                {preview && (
                  <>
                    <div className="safe-banner"><strong>Soft delete only</strong><span>No node or provenance will be hard-deleted.</span></div>
                    <h3>{preview.dependencies.length} evidentiary dependents</h3>
                    {preview.dependencies.map((impact) => (
                      <article key={impact.memory_id} className="impact-row">
                        <code>{shortId(impact.memory_id)}</code>
                        <span>{pretty(impact.current_status)} → <strong>{pretty(impact.proposed_status)}</strong></span>
                        <small>{impact.reason}</small>
                      </article>
                    ))}
                    {!preview.dependencies.length && <p className="empty-copy">No evidentiary dependents will change.</p>}
                  </>
                )}
              </div>
            )}
            <label>Reason<textarea value={reason} onChange={(event) => setReason(event.target.value)} rows={2} placeholder="Optional research note" /></label>
            {dialogError && <p className="form-error" role="alert">{dialogError}</p>}
            <footer className="modal__actions">
              <button type="button" onClick={() => setDialog(null)}>Cancel</button>
              <button className={dialog === "prune" ? "button-danger" : "button-primary"} type="submit" disabled={busy || (dialog === "prune" && !preview)}>
                {busy ? "Applying…" : dialog === "prune" ? "Confirm prune" : "Apply correction"}
              </button>
            </footer>
          </form>
        </Modal>
      )}
    </div>
  );
}
