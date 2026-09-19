# Phase 4 Baselines

Phase 4 implements three controls alongside `ScopeGraphMemorySystem`. They share the `MemorySystem` contract and return the same `RetrievalResult`, `RetrievedMemory`, and `TraversalStep` models.

| Behavior | VectorMemory | FlatGraphMemory | TwoLevelGraphMemory | ScopeGraph |
| --- | --- | --- | --- | --- |
| Semantic embeddings | yes | yes | yes | yes |
| Graph expansion | no | bounded | bounded | bounded |
| Current-session validity | no | no | yes | yes |
| Context/project validity | no | no | no | yes |
| Global memory | flat store | flat graph | yes | yes |
| Cross-context conflict domain | none | entire graph | global level | containing scope |

## Controlled inputs

Every backend accepts the same sessions and structured candidates, retains source-message provenance, uses the same embedding provider, and enforces the caller's top-k and token budget. FlatGraphMemory, TwoLevelGraphMemory, and ScopeGraph use the same ranking weights and traversal limits. FlatGraphMemory records a zero scope component because that signal is disabled; the constant does not influence ordering. VectorMemory ranks only by cosine similarity, as required for a vector-only control.

## Representation details

The shared persistence schema requires every memory to carry `scope_id` and `scope_level`. VectorMemory and FlatGraphMemory store the source context in those fields for auditing, but their retrieval paths deliberately ignore it. TwoLevelGraphMemory stores low-durability temporary state at session level and collapses durable context facts into the global root. This makes its missing intermediate context layer observable instead of silently simulating project scopes.

Graph baselines create supersession relationships inside their effective conflict domain. VectorMemory keeps independent extracted chunks and therefore retains conflicting statements as separate active records.

## Isolation

Run each backend in an isolated repository or disposable database. The Phase 4 implementations intentionally do not add experiment namespaces to production nodes; reset, namespace allocation, and multi-backend orchestration belong to the evaluation harness. Sharing one populated repository would mix candidate pools and invalidate the comparison.
