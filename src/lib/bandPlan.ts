/** Group uploaded files into scenes — 12 optical planes or 2 SAR planes each. */

export const S2_BANDS = [
  "B01", "B02", "B03", "B04", "B05", "B06",
  "B07", "B08", "B8A", "B09", "B11", "B12",
] as const;

export const SAR_BANDS = ["VV", "VH"] as const;

export type UploadKind = "optical" | "sar" | "fusion";

export type SceneKind = "optical" | "sar" | "rgb";

export type SceneDraft = {
  name: string;
  src: string;
  modality: "optical" | "sar" | "multispectral";
  bandNames: string[];
  sourceFiles: string[];
  missingBands: string[];
  complete: boolean;
};

export type PlannedScene = {
  kind: SceneKind;
  indices: number[];
  names: string[];
  required: readonly string[];
  missing: string[];
  complete: boolean;
  label: string;
};

export type BandPlan = {
  scenes: PlannedScene[];
  needsKind: boolean;
  ready: boolean;
  message: string;
};

const ALIASES: Record<string, string> = {
  B1: "B01", B01: "B01", BAND1: "B01", BAND01: "B01",
  B2: "B02", B02: "B02", BAND2: "B02", BAND02: "B02",
  B3: "B03", B03: "B03", BAND3: "B03", BAND03: "B03",
  B4: "B04", B04: "B04", BAND4: "B04", BAND04: "B04",
  B5: "B05", B05: "B05", BAND5: "B05", BAND05: "B05",
  B6: "B06", B06: "B06", BAND6: "B06", BAND06: "B06",
  B7: "B07", B07: "B07", BAND7: "B07", BAND07: "B07",
  B8: "B08", B08: "B08", BAND8: "B08", BAND08: "B08",
  B8A: "B8A", B08A: "B8A",
  B9: "B09", B09: "B09", BAND9: "B09", BAND09: "B09",
  B11: "B11", BAND11: "B11",
  B12: "B12", BAND12: "B12",
  VV: "VV", VH: "VH",
};

const TOKEN = /(?:^|[^A-Za-z0-9])(B8A|B08A|B0?[1-9]|B1[12]|BAND[_-]?0?[1-9]|BAND[_-]?1[12]|VV|VH)(?:[^A-Za-z0-9]|$)/i;

export function parseBandToken(filename: string): string | null {
  const base = filename.split(/[/\\]/).pop() ?? filename;
  const match = TOKEN.exec(base);
  if (!match) return null;
  let raw = match[1].toUpperCase().replace(/[-_]/g, "");
  if (raw.startsWith("BAND")) raw = `B${raw.slice(4)}`;
  return ALIASES[raw] ?? null;
}

export function sceneStem(filename: string): string {
  const base = (filename.split(/[/\\]/).pop() ?? filename).replace(/\.[^.]+$/, "");
  const token = parseBandToken(filename);
  let name = base;
  if (token) {
    const extras = new Set<string>([token]);
    if (token.startsWith("B") && /^\d+$/.test(token.slice(1))) {
      const n = Number(token.slice(1));
      extras.add(`B${n}`);
      extras.add(`B${String(n).padStart(2, "0")}`);
    }
    for (const variant of [...extras].sort((a, b) => b.length - a.length)) {
      name = name.replace(new RegExp(`(^|[_.\\-])${variant}(?=[_.\\-]|$)`, "i"), "$1");
    }
  }
  return name.replace(/[_\-.]+/g, "_").replace(/^_+|_+$/g, "").toLowerCase() || "scene";
}

function looksSar(filename: string) {
  const n = filename.toLowerCase();
  return ["sar", "risat", "sentinel-1", "sentinel1", "_s1_", "radar", "vv", "vh"].some((h) => n.includes(h));
}

function finish(
  kind: SceneKind,
  indices: number[],
  names: string[],
  expect?: UploadKind | null
): PlannedScene {
  const required = kind === "sar" ? SAR_BANDS : S2_BANDS;
  const unique = names.filter((n, i) => n && names.indexOf(n) === i);
  const missing = kind === "rgb" ? [] : required.filter((n) => !unique.includes(n));
  let complete = kind === "rgb" || missing.length === 0;
  if (kind === "sar" && !complete && indices.length === 1 && unique.length === 0 && expect !== "sar") {
    complete = true;
  }
  if (kind === "optical" && indices.length === 12 && unique.length === 0) {
    return {
      kind,
      indices,
      names: [...S2_BANDS],
      required,
      missing: [],
      complete: true,
      label: "Optical MSI (12 bands)",
    };
  }
  if (kind === "sar" && indices.length === 2 && unique.length === 0) {
    return {
      kind,
      indices,
      names: [...SAR_BANDS],
      required,
      missing: [],
      complete: true,
      label: "SAR VV+VH",
    };
  }
  const label =
    kind === "sar"
      ? complete
        ? `SAR ${unique.join("+") || "scene"}`
        : `SAR (${unique.length}/2 bands)`
      : kind === "optical"
        ? `Optical MSI (${unique.length}/12 bands)`
        : "RGB preview";
  return { kind, indices, names: unique, required, missing, complete, label };
}

export function planScenes(filenames: string[], expect?: UploadKind | null): BandPlan {
  if (!filenames.length) {
    return { scenes: [], needsKind: false, ready: false, message: "Upload a scene first." };
  }

  const tokens = filenames.map(parseBandToken);
  const stems = filenames.map(sceneStem);
  const groups = new Map<string, number[]>();
  const rgb: number[] = [];

  filenames.forEach((name, index) => {
    const token = tokens[index];
    const ext = name.split(".").pop()?.toLowerCase() ?? "";
    const isRgbChip = /^(png|jpe?g|webp)$/.test(ext) && !token;
    if (isRgbChip) {
      rgb.push(index);
      return;
    }
    const kind = token === "VV" || token === "VH" || looksSar(name) ? "sar" : "optical";
    const year = name.match(/20\d{2}/)?.[0] ?? "";
    const key = `${kind}:${stems[index]}:${year}`;
    const list = groups.get(key) ?? [];
    list.push(index);
    groups.set(key, list);
  });

  const scenes: PlannedScene[] = [];
  for (const [key, indices] of groups) {
    const kind: SceneKind = key.startsWith("sar:") ? "sar" : "optical";
    scenes.push(finish(kind, indices, indices.map((i) => tokens[i] ?? ""), expect));
  }
  for (const index of rgb) {
    scenes.push(finish("rgb", [index], ["B04", "B03", "B02"], expect));
  }

  if (expect === "fusion") {
    if (!scenes.some((s) => s.kind !== "sar")) {
      scenes.unshift(finish("optical", [], [], expect));
    }
    if (!scenes.some((s) => s.kind === "sar")) {
      scenes.push(finish("sar", [], [], expect));
    }
  } else if (expect === "optical" && !scenes.some((s) => s.kind === "optical" || s.kind === "rgb")) {
    scenes.push(finish("optical", [], [], expect));
  } else if (expect === "sar" && !scenes.some((s) => s.kind === "sar")) {
    scenes.push(finish("sar", [], [], expect));
  }

  scenes.sort((a, b) => (a.kind === "sar" ? 1 : 0) - (b.kind === "sar" ? 1 : 0));

  const unlabeled = filenames.filter((n, i) => !tokens[i] && !/\.(png|jpe?g|webp)$/i.test(n));
  const needsKind =
    !expect &&
    filenames.length === 2 &&
    unlabeled.length === 2 &&
    scenes.length === 1 &&
    scenes[0].kind === "optical" &&
    !scenes[0].complete;

  const present = scenes.filter((s) => s.indices.length);
  const blocking = present.filter(
    (s) => !s.complete || (s.kind === "rgb" && (expect === "optical" || expect === "fusion"))
  );
  const ready = present.length > 0 && blocking.length === 0;
  const message = blocking
    .map((s) =>
      s.kind === "sar"
        ? `SAR needs VV and VH. Missing: ${s.missing.join(", ") || "both polarisations"}.`
        : `Optical needs all 12 Sentinel-2 bands (B01–B12, no B10). Missing: ${s.missing.join(", ") || "the band TIFFs"}.`
    )
    .join(" ");

  return {
    scenes,
    needsKind,
    ready,
    message: message || "Scene bands are complete.",
  };
}

export function draftsFromPlan(plan: BandPlan, files: File[]): SceneDraft[] {
  return plan.scenes
    .filter((scene) => scene.indices.length)
    .map((scene) => {
      const sceneFiles = scene.indices.map((i) => files[i]).filter(Boolean);
      const preview =
        sceneFiles.find((f) => /B04|B08|VV/i.test(f.name)) ?? sceneFiles[0];
      return {
        name: scene.label,
        src: preview ? URL.createObjectURL(preview) : "",
        modality: scene.kind === "sar" ? "sar" : scene.names.length >= 4 ? "multispectral" : "optical",
        bandNames: scene.names,
        sourceFiles: sceneFiles.map((f) => f.name),
        missingBands: scene.missing,
        complete: scene.complete,
      };
    });
}

export const MAX_UPLOAD_FILES = 24;
