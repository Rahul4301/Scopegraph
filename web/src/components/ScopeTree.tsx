import type { Scope } from "../types";

interface ScopeTreeProps {
  scopes: Scope[];
  selectedScopeId: string | null;
  onSelect: (scopeId: string | null) => void;
}

function ScopeBranch({
  scope,
  scopes,
  selectedScopeId,
  onSelect,
  depth,
}: {
  scope: Scope;
  scopes: Scope[];
  selectedScopeId: string | null;
  onSelect: (scopeId: string) => void;
  depth: number;
}) {
  const children = scopes.filter((item) => item.parent_scope_id === scope.id);
  return (
    <li>
      <button
        className={`scope-row ${selectedScopeId === scope.id ? "is-selected" : ""}`}
        style={{ paddingLeft: `${12 + depth * 16}px` }}
        onClick={() => onSelect(scope.id)}
        type="button"
        aria-pressed={selectedScopeId === scope.id}
      >
        <span className={`scope-glyph scope-glyph--${scope.scope_type}`} aria-hidden="true" />
        <span className="scope-row__text">
          <strong>{scope.name}</strong>
          <small>{scope.scope_type.replace("_", " ")}</small>
        </span>
        {scope.archived && <span className="tag tag--muted">archived</span>}
      </button>
      {children.length > 0 && (
        <ul>
          {children.map((child) => (
            <ScopeBranch
              key={child.id}
              scope={child}
              scopes={scopes}
              selectedScopeId={selectedScopeId}
              onSelect={onSelect}
              depth={depth + 1}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

export function ScopeTree({ scopes, selectedScopeId, onSelect }: ScopeTreeProps) {
  const roots = scopes.filter(
    (scope) => !scope.parent_scope_id || !scopes.some((item) => item.id === scope.parent_scope_id),
  );
  return (
    <nav className="scope-tree" aria-label="Memory scopes">
      <button
        className={`scope-row scope-row--all ${selectedScopeId === null ? "is-selected" : ""}`}
        onClick={() => onSelect(null)}
        type="button"
        aria-pressed={selectedScopeId === null}
      >
        <span className="scope-glyph scope-glyph--all" aria-hidden="true" />
        <span className="scope-row__text">
          <strong>All memory</strong>
          <small>unfiltered graph</small>
        </span>
      </button>
      <ul>
        {roots.map((scope) => (
          <ScopeBranch
            key={scope.id}
            scope={scope}
            scopes={scopes}
            selectedScopeId={selectedScopeId}
            onSelect={onSelect}
            depth={0}
          />
        ))}
      </ul>
      {scopes.length === 0 && <p className="empty-copy">No scopes yet.</p>}
    </nav>
  );
}
