/**
 * Interim SatQuery LLM — used while the fine-tuned VLM is still training.
 *
 * Prefers the FastAPI backend (`NEXT_PUBLIC_SATQUERY_API` + /api/llm/ask)
 * so the key stays in backend/.env. If the backend is not running, falls
 * back to the same-origin Next route `/api/satquery-llm` which reads
 * SATQUERY_LLM_API_KEY from `.env.local`.
 */

import { API_BASE, backendEnabled } from "./api";

export type LlmAskResult = {
  answer: string;
  model: string;
  provider: string;
  confidence: number;
};

export type LlmAskInput = {
  query: string;
  task?: string;
  imageUrl?: string;
  context?: string;
};

async function imageUrlToPngB64(imageUrl: string): Promise<string | undefined> {
  try {
    const response = await fetch(imageUrl);
    if (!response.ok) return undefined;
    const blob = await response.blob();
    const dataUrl = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(blob);
    });
    const comma = dataUrl.indexOf(",");
    return comma >= 0 ? dataUrl.slice(comma + 1) : undefined;
  } catch {
    return undefined;
  }
}

export async function llmStatus(): Promise<{ configured: boolean; provider?: string; model?: string }> {
  try {
    if (backendEnabled()) {
      const response = await fetch(`${API_BASE}/api/llm/status`);
      if (response.ok) return (await response.json()) as { configured: boolean };
    }
    const response = await fetch("/api/satquery-llm");
    if (response.ok) return (await response.json()) as { configured: boolean };
  } catch {
    /* not configured */
  }
  return { configured: false };
}

export async function askSatQueryLlm(input: LlmAskInput): Promise<LlmAskResult | null> {
  const imageB64 = input.imageUrl ? await imageUrlToPngB64(input.imageUrl) : undefined;
  const body = JSON.stringify({
    query: input.query,
    task: input.task ?? "vqa",
    imageB64,
    context: input.context,
  });

  const targets: string[] = [];
  if (backendEnabled()) targets.push(`${API_BASE}/api/llm/ask`);
  targets.push("/api/satquery-llm");

  for (const url of targets) {
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      if (response.status === 503) continue;
      if (!response.ok) continue;
      const parsed = (await response.json()) as LlmAskResult;
      if (parsed.answer?.trim()) return parsed;
    } catch {
      /* try the next target */
    }
  }
  return null;
}
