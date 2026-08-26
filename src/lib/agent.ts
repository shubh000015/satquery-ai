import type { InputMode, Intent, Mission, QueryResult, TaskKind } from "./types";

const SPECIALISTS: Record<TaskKind, string[]> = {
  vqa: ["RSVQA-Adapter", "Scene Parser"],
  caption: ["VRS-Captioner", "Land-cover Lexicon"],
  grounding: ["RS-Grounder", "Box Refiner"],
  change: ["CD-Mapper", "Change Describer"],
  "change-vqa": ["CDVQA", "Temporal Aligner"],
  "cross-modal": ["OptSAR Fusion", "RSVQA-Adapter"],
  measure: ["Geometry", "GSD Scaler"],
};

export function classifyQuery(query: string, mode: InputMode): Intent {
  const q = query.toLowerCase();

  const has = (...words: string[]) => words.some((w) => q.includes(w));

  let task: TaskKind = "vqa";

  if (has("measure", "how far", "area of", "distance", "km²", "km2")) {
    task = "measure";
  } else if (has("flood", "inundat", "water-cover", "water covered", "affected settlement")) {
    task = mode === "cross-modal" ? "cross-modal" : "vqa";
  } else if (has("changed", "change", "increased", "decreased", "between these", "compared to", "expansion")) {
    task = has("has the", "did the", "increased", "decreased", "unchanged") ? "change-vqa" : "change";
  } else if (has("optical and sar", "both sensors", "built-up and water", "through cloud")) {
    task = "cross-modal";
  } else if (has("highlight", "locate", "point to", "ground", "where is", "which ships", "tanks", "ships in")) {
    task = "grounding";
  } else if (has("describe", "caption", "land-cover", "land cover", "scene", "what do you see")) {
    task = "caption";
  }

  if ((task === "change" || task === "change-vqa") && mode !== "bi-temporal") {
    return {
      task,
      label: task === "change-vqa" ? "Change VQA" : "Change description",
      specialists: SPECIALISTS[task],
      warning: "This query needs a two-date pair. Load a second date or open Gurugram Expansion.",
    };
  }

  if (task === "cross-modal" && mode !== "cross-modal" && !has("flood")) {
    return {
      task,
      label: "Optical–SAR fusion",
      specialists: SPECIALISTS[task],
      warning: "Cross-modal fusion expects an optical + SAR pair of the same footprint.",
    };
  }

  const labels: Record<TaskKind, string> = {
    vqa: "Visual question answering",
    caption: "Scene captioning",
    grounding: "Text-guided grounding",
    change: "Change description",
    "change-vqa": "Change VQA",
    "cross-modal": "Optical–SAR fusion",
    measure: "Mensuration",
  };

  return {
    task,
    label: labels[task],
    specialists: SPECIALISTS[task],
  };
}

export function resolveResult(query: string, mission: Mission): QueryResult {
  const q = query.toLowerCase();
  let best: { n: number; result: QueryResult } | null = null;

  for (const reply of mission.replies) {
    const n = reply.match.reduce((acc, token) => acc + (q.includes(token) ? 1 : 0), 0);
    if (n > 0 && (!best || n > best.n)) best = { n, result: reply.result };
  }

  const result = best?.result ?? mission.fallback;
  return {
    ...result,
    models: result.models.map((m) => ({ ...m })),
    trace: result.trace.map((s) => ({ ...s, status: "pending" })),
    layers: result.layers.map((l) => ({ ...l })),
  };
}

export const AGENT_TIMING = [420, 640, 520, 880, 760, 540];
