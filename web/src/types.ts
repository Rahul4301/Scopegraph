export type ScopeType =
  | "global"
  | "project"
  | "repository"
  | "course"
  | "client"
  | "task"
  | "workspace"
  | "custom";

export type ScopeLevel = "session" | "scope" | "global";
export type MemoryStatus =
  | "active"
  | "superseded"
  | "archived"
  | "tombstoned"
  | "needs_review";

export interface Scope {
  id: string;
  name: string;
  scope_type: ScopeType;
  parent_scope_id: string | null;
  created_at: string;
  archived: boolean;
}

export interface Memory {
  id: string;
  content: string;
  memory_type: string;
  scope_level: ScopeLevel;
  scope_id: string;
  confidence: number;
  status: MemoryStatus;
  created_at: string;
  updated_at: string;
  valid_from: string | null;
  valid_to: string | null;
  last_confirmed_at: string | null;
  embedding: number[] | null;
  embedding_model: string | null;
  revision: number;
  metadata: Record<string, unknown>;
  source_ids: string[];
}

export interface SourceMessage {
  id: string;
  session_id: string;
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  timestamp: string;
  turn_index: number;
}

export interface CorrectionEvent {
  id: string;
  action: string;
  timestamp: string;
  actor: string;
  before: Record<string, unknown>;
  after: Record<string, unknown>;
  reason: string;
  undo_of: string | null;
}

export interface GraphNode {
  id: string;
  node_type: "scope" | "memory" | "source_message";
  label: string;
  data: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
  kind: string | null;
}

export interface GraphSubgraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  generated_at: string;
}

export interface MemoryProvenance {
  memory: Memory;
  source_messages: SourceMessage[];
}

export interface MemoryStats {
  backend_name: string;
  scope_count: number;
  session_count: number;
  source_message_count: number;
  memory_count: number;
  relationship_count: number;
}

export interface ConfigStatus {
  llm_configured: boolean;
  embedding_configured: boolean;
}

export interface PruneImpact {
  memory_id: string;
  relation: string;
  current_status: MemoryStatus;
  proposed_status: MemoryStatus;
  other_active_support_ids: string[];
  reason: string;
}

export interface PrunePreview {
  memory_id: string;
  current_status: MemoryStatus;
  proposed_status: "tombstoned";
  dependencies: PruneImpact[];
  graph_neighbors: Array<{
    memory_id: string;
    relation: string;
    evidentiary_dependency: boolean;
  }>;
  hard_delete: false;
}

export interface CorrectionResult {
  event: CorrectionEvent;
  memory: Memory;
  affected_memories: Memory[];
}

export interface RetrievedMemory {
  memory_id: string;
  content: string;
  score: number;
  scope_id: string;
  scope_level: ScopeLevel;
  status: MemoryStatus;
  source_ids: string[];
  valid_from: string | null;
  valid_to: string | null;
  semantic_score: number;
  scope_score: number;
  temporal_score: number;
  graph_score: number;
  confidence: number;
}

export interface TraversalStep {
  from_id: string;
  to_id: string;
  relation: string;
  depth: number;
  path: string[];
  semantic_score: number | null;
  scope_score: number | null;
  temporal_score: number | null;
  graph_score: number | null;
  final_score: number | null;
  reason: string | null;
}

export interface RetrievalResult {
  items: RetrievedMemory[];
  retrieval_latency_ms: number;
  token_count: number;
  trace: TraversalStep[];
  backend_name: string;
}
