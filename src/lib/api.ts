/**
 * Client for the FastAPI agent backend (see backend/README.md).
 *
 * Set NEXT_PUBLIC_SATQUERY_API to switch the workspace from the local mock
 * agent onto the real pipeline. When it is unset, `backendEnabled()` is false
 * and the store keeps using the scripted missions, so demos never depend on a
 * running Python process.
 */

import type { AgentStep, Asset, InputMode, QueryResult } from "./types";

export const API_BASE = (process.env.NEXT_PUBLIC_SATQUERY_API ?? "").replace(/\/+$/, "");

export function backendEnabled() {
  return API_BASE.length > 0;
}

export type Severity = "error" | "warning" | "info";

export type ValidationIssue = {
  code: string;
  severity: Severity;
  message: string;
  hint?: string | null;
  assetId?: string | null;
};

export type ValidationReport = {
  ok: boolean;
  mode: InputMode;
  assetCount: number;
  issues: ValidationIssue[];
  coRegistered?: boolean | null;
  overlapPercent?: number | null;
  summary: string;
};

export type RasterMeta = {
  width: number;
  height: number;
  bands: number;
  dtype: string;
  georeferenced: boolean;
  crs?: string | null;
  gsdMeters?: number | null;
  bounds?: number[] | null;
  driver?: string | null;
};

export type BackendAsset = Asset & {
  sessionId?: string | null;
  sizeBytes?: number;
  benchmarkOnlyFormat?: boolean;
  benchmarkDataset?: string | null;
  modalitySource?: string;
  meta?: RasterMeta | null;
};

export type BackendQueryResult = QueryResult & {
  queryId: string;
  sessionId: string | null;
  query: string;
  mode: InputMode;
  warnings: string[];
  inferenceBackend: string;
  assets: BackendAsset[];
  validation?: ValidationReport | null;
  elapsedMs: number;
  createdAt: string;
};

export type UploadResult = {
  sessionId: string;
  assets: BackendAsset[];
  mode: InputMode;
  validation: ValidationReport;
  suggested: string[];
};

export type ModelSpec = {
  key: string;
  requirement: string;
  primaryModel: string;
  fallbackModel?: string | null;
  benchmarks: string[];
  purpose: string;
  status: "planned" | "wired" | "available";
};

export type RegistryResponse = {
  models: ModelSpec[];
  tools: { name: string; task: string; role: string; modelKey: string; backend: string }[];
  weightsWired: number;
  total: number;
};

export class ApiError extends Error {
  code: string;
  issues: string[];

  constructor(message: string, code = "error", issues: string[] = []) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.issues = issues;
  }
}

/** Asset `src` comes back as an API path, so it needs the base URL prefixed. */
export function assetUrl(src: string) {
  if (/^(https?:|blob:|data:)/.test(src)) return src;
  return `${API_BASE}${src}`;
}

async function readError(response: Response) {
  try {
    const body = await response.json();
    const detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body?.detail ?? body);
    return new ApiError(detail, body?.code ?? `http-${response.status}`, body?.issues ?? []);
  } catch {
    return new ApiError(`${response.status} ${response.statusText}`, `http-${response.status}`);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (!backendEnabled()) throw new ApiError("NEXT_PUBLIC_SATQUERY_API is not set.", "not-configured");
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) throw await readError(response);
  return (await response.json()) as T;
}

export async function fetchHealth() {
  return request<{
    status: string;
    rasterio: boolean;
    weightsWired: number;
    mlEndpoint?: string | null;
    mlReachable?: boolean | null;
  }>("/api/health");
}

export async function fetchRegistry() {
  return request<RegistryResponse>("/api/registry");
}

export async function uploadAssets(
  files: File[],
  opts?: { sessionId?: string | null; mode?: InputMode; benchmarkDataset?: string }
): Promise<UploadResult> {
  const form = new FormData();
  files.forEach((file) => form.append("files", file, file.name));
  if (opts?.sessionId) form.append("session_id", opts.sessionId);
  if (opts?.mode) form.append("mode", opts.mode);
  if (opts?.benchmarkDataset) form.append("benchmark_dataset", opts.benchmarkDataset);

  const result = await request<UploadResult>("/api/assets", { method: "POST", body: form });
  return {
    ...result,
    assets: result.assets.map((asset) => ({ ...asset, src: assetUrl(asset.src) })),
  };
}

export async function validateAssets(assetIds: string[], mode?: InputMode) {
  return request<ValidationReport>("/api/assets/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ assetIds, mode }),
  });
}

export type QueryPayload = {
  query: string;
  assetIds: string[];
  sessionId?: string | null;
  mode?: InputMode;
};

function absolutise(result: BackendQueryResult): BackendQueryResult {
  return { ...result, assets: (result.assets ?? []).map((a) => ({ ...a, src: assetUrl(a.src) })) };
}

export async function runQuery(payload: QueryPayload) {
  const result = await request<BackendQueryResult>("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return absolutise(result);
}

export type StreamHandlers = {
  onTrace?: (steps: AgentStep[]) => void;
  onStep?: (step: AgentStep) => void;
  onResult?: (result: BackendQueryResult) => void;
};

/**
 * Runs the pipeline over SSE so the audit trail fills in as each stage
 * completes, rather than appearing all at once at the end.
 */
export async function streamQuery(
  payload: QueryPayload,
  handlers: StreamHandlers,
  signal?: AbortSignal
): Promise<BackendQueryResult> {
  if (!backendEnabled()) throw new ApiError("NEXT_PUBLIC_SATQUERY_API is not set.", "not-configured");

  const response = await fetch(`${API_BASE}/api/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) throw await readError(response);
  if (!response.body) throw new ApiError("Streaming is not supported by this browser.", "no-stream");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let final: BackendQueryResult | null = null;

  const handleFrame = (frame: string) => {
    let event = "message";
    const data: string[] = [];
    for (const line of frame.split("\n")) {
      if (line.startsWith("event: ")) event = line.slice(7).trim();
      else if (line.startsWith("data: ")) data.push(line.slice(6));
    }
    if (!data.length) return;
    const parsed = JSON.parse(data.join("\n"));

    if (event === "trace") handlers.onTrace?.(parsed.steps as AgentStep[]);
    else if (event === "step") handlers.onStep?.(parsed as AgentStep);
    else if (event === "result") {
      final = absolutise(parsed as BackendQueryResult);
      handlers.onResult?.(final);
    } else if (event === "error") {
      throw new ApiError(parsed.detail ?? "Pipeline failed.", parsed.code, parsed.issues ?? []);
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let split = buffer.indexOf("\n\n");
    while (split !== -1) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      if (frame.trim()) handleFrame(frame);
      split = buffer.indexOf("\n\n");
    }
  }
  if (buffer.trim()) handleFrame(buffer);

  if (!final) throw new ApiError("The pipeline closed without returning a result.", "no-result");
  return final;
}

export function reportUrl(sessionId: string, queryId?: string) {
  const suffix = queryId ? `?queryId=${encodeURIComponent(queryId)}` : "";
  return `${API_BASE}/api/sessions/${encodeURIComponent(sessionId)}/report${suffix}`;
}

export async function deleteSession(sessionId: string) {
  return request<{ deleted: boolean }>(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
}
