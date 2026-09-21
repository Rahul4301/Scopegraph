import type {
  CorrectionEvent,
  CorrectionResult,
  ConfigStatus,
  GraphSubgraph,
  Memory,
  MemoryProvenance,
  MemoryStats,
  PrunePreview,
  RetrievalResult,
  Scope,
} from "./types";

export const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "/api";

export function graphExportUrl(scopeId?: string | null): string {
  const params = new URLSearchParams({ format: "graphml" });
  if (scopeId) params.set("scope_id", scopeId);
  return `${API_BASE}/graph/export?${params}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(body?.detail ?? `${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<{ status: string; neo4j: string }>("/health"),
  configStatus: () => request<ConfigStatus>("/config/status"),
  scopes: () => request<Scope[]>("/scopes?include_archived=true"),
  stats: () => request<MemoryStats>("/stats"),
  memory: (id: string) => request<Memory>(`/memories/${encodeURIComponent(id)}`),
  provenance: (id: string) =>
    request<MemoryProvenance>(`/memories/${encodeURIComponent(id)}/provenance`),
  history: (id: string) =>
    request<CorrectionEvent[]>(`/memories/${encodeURIComponent(id)}/history`),
  graph: (options: {
    scopeId?: string;
    memoryId?: string;
    includeInactive: boolean;
    includeSources: boolean;
  }) => {
    const params = new URLSearchParams({
      include_inactive: String(options.includeInactive),
      include_sources: String(options.includeSources),
      limit: "300",
    });
    if (options.scopeId) params.set("scope_id", options.scopeId);
    if (options.memoryId) params.set("memory_id", options.memoryId);
    return request<GraphSubgraph>(`/graph/subgraph?${params}`);
  },
  edit: (id: string, body: Record<string, unknown>) =>
    request<CorrectionResult>(`/memories/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  move: (id: string, body: Record<string, unknown>) =>
    request<CorrectionResult>(`/memories/${encodeURIComponent(id)}/move`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  archive: (id: string, body: Record<string, unknown>) =>
    request<CorrectionResult>(`/memories/${encodeURIComponent(id)}/archive`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  prunePreview: (id: string) =>
    request<PrunePreview>(`/memories/${encodeURIComponent(id)}/prune/preview`, {
      method: "POST",
    }),
  prune: (id: string, body: Record<string, unknown>) =>
    request<CorrectionResult>(`/memories/${encodeURIComponent(id)}/prune`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  restore: (id: string, body: Record<string, unknown>) =>
    request<CorrectionResult>(`/memories/${encodeURIComponent(id)}/restore`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  merge: (id: string, body: Record<string, unknown>) =>
    request<CorrectionResult>(`/memories/${encodeURIComponent(id)}/merge`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  retrieve: (body: Record<string, unknown>) =>
    request<RetrievalResult>("/retrieve", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};
