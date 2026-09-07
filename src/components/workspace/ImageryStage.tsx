"use client";

import { useRef } from "react";
import { useSatQuery } from "@/lib/store";
import type { Asset, CompareMode, LayerId } from "@/lib/types";

const MASK_LAYER: Record<string, LayerId> = {
  flood: "flood",
  channel: "water",
  gain: "change",
  ag: "vegetation",
};

const BOX_LAYER: Record<string, LayerId> = {
  settlement: "settlements",
  ship: "grounding",
  tank: "grounding",
  urban: "urban",
};

function gsdMeters(gsd: string) {
  const n = parseFloat(gsd);
  if (Number.isNaN(n)) return 10;
  if (gsd.includes("cm")) return n / 100;
  return n;
}

export function ImageryStage() {
  const s = useSatQuery();
  const stageRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);

  const primary = s.mission?.assets.find((a) => a.role === "primary") ?? s.mission?.assets[0];
  const secondary = s.mission?.assets.find((a) => a.role === "secondary");
  const active = new Set((s.result?.layers ?? []).filter((l) => l.active).map((l) => l.id));
  const layerFilterOn = (s.result?.layers.length ?? 0) > 0;




  if (!primary) return null;

  const onPointer = (e: React.PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    if (s.measuring) {
      s.addMeasurePt({ x, y });
      return;
    }
  };

  const meters =
    s.measurePts.length === 2 && primary
      ? distKm(s.measurePts[0], s.measurePts[1], primary)
      : null;

  return (
    <div className="relative flex h-full w-full flex-col">
      <div
        ref={stageRef}
        className={`relative min-h-0 flex-1 overflow-hidden bg-black ${s.measuring ? "cursor-crosshair" : "cursor-default"}`}
      >
        <div
          className="absolute inset-0 overflow-hidden"
          onClick={onPointer}
        >
          <Frame
            primary={primary}
            secondary={secondary}
            compare={s.compare}
            swipe={s.swipe}
            running={s.running}
            acquiring={s.acquiring}
            hasResult={!!s.result && !s.running}
          />
          <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none">
            {s.result?.masks.map((m) => {
              const layerId = MASK_LAYER[m.id];
              if (layerFilterOn && layerId && !active.has(layerId)) return null;
              return (
                <path
                  key={m.id}
                  d={m.d}
                  fill={m.color}
                  fillOpacity={m.opacity}
                  stroke={m.color}
                  strokeWidth={0.25}
                  className="transition-opacity duration-700"
                />
              );
            })}
            {s.result?.boxes.map((b) => {
              const layerId = BOX_LAYER[b.kind ?? ""] ?? "grounding";
              if (layerFilterOn && !active.has(layerId) && !active.has("grounding") && !active.has("settlements")) return null;
              const selected = s.selectedId === b.id;
              return (
                <g key={b.id} onClick={(e) => { e.stopPropagation(); s.selectFeature(b.id); }} className="cursor-pointer">
                  <rect
                    x={b.x}
                    y={b.y}
                    width={b.w}
                    height={b.h}
                    fill="none"
                    stroke={selected ? "#7ee0d0" : "#d4b56a"}
                    strokeWidth={selected ? 0.55 : 0.35}
                  />
                  <text x={b.x} y={b.y - 0.8} fill="#d4b56a" fontSize="2.2" fontFamily="IBM Plex Mono, monospace">
                    {b.label}
                  </text>
                </g>
              );
            })}
            {s.measurePts.map((p, i) => (
              <circle key={i} cx={p.x} cy={p.y} r={0.7} fill="#7ee0d0" />
            ))}
            {s.measurePts.length === 2 && (
              <line
                x1={s.measurePts[0].x}
                y1={s.measurePts[0].y}
                x2={s.measurePts[1].x}
                y2={s.measurePts[1].y}
                stroke="#7ee0d0"
                strokeWidth={0.25}
              />
            )}
          </svg>
        </div>

        {s.compare === "swipe" && secondary && (
          <input
            type="range"
            min={8}
            max={92}
            value={s.swipe}
            onChange={(e) => s.setSwipe(Number(e.target.value))}
            className="absolute inset-x-8 bottom-6 z-10 h-1 cursor-ew-resize appearance-none bg-transparent"
          />
        )}

        {s.acquiring && (
          <div className="scan absolute inset-0 z-20 bg-void/30" />
        )}



        {meters !== null && (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 bg-void/80 px-3 py-1 font-mono text-[11px] text-signal">
            {meters < 1 ? `${Math.round(meters * 1000)} m` : `${meters.toFixed(2)} km`}
          </div>
        )}
      </div>
    </div>
  );
}

function distKm(a: { x: number; y: number }, b: { x: number; y: number }, asset: Asset) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const pxFrac = Math.hypot(dx, dy) / 100;
  const sceneM = gsdMeters(asset.gsd) * 1792;
  return (pxFrac * sceneM) / 1000;
}

function Frame({
  primary,
  secondary,
  compare,
  swipe,
  running,
  acquiring,
  hasResult,
}: {
  primary: Asset;
  secondary?: Asset;
  compare: CompareMode;
  swipe: number;
  running: boolean;
  acquiring: boolean;
  hasResult?: boolean;
}) {
  if (compare === "split" && secondary) {
    return (
      <div className="grid h-full grid-cols-2 gap-px bg-brass/30">
        <Plate asset={primary} label="A" running={running} hasResult={hasResult} />
        <Plate asset={secondary} label="B" running={running} hasResult={hasResult} />
      </div>
    );
  }

  if (compare === "swipe" && secondary) {
    return (
      <div className="relative h-full w-full">
        <Plate asset={secondary} label="B" running={running} hasResult={hasResult} />
        <div className="absolute inset-0 overflow-hidden" style={{ width: `${swipe}%` }}>
          <div className="h-full" style={{ width: `${10000 / swipe}%` }}>
            <Plate asset={primary} label="A" running={running} hasResult={hasResult} />
          </div>
        </div>
        <div className="absolute top-0 bottom-0 w-0.5 bg-brass" style={{ left: `${swipe}%` }}>
          <div className="absolute top-1/2 left-1/2 h-10 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-sm bg-brass" />
        </div>
      </div>
    );
  }

  if (compare === "diff" && secondary) {
    return (
      <div className="relative h-full w-full">
        <Plate asset={primary} label="t₀" running={running} hasResult={hasResult} />
        <div className="absolute inset-0 mix-blend-difference opacity-80">
          <Plate asset={secondary} label="t₁" running={running} hasResult={hasResult} />
        </div>
      </div>
    );
  }

  const shown = compare === "secondary" && secondary ? secondary : primary;
  return (
    <div className={`h-full w-full ${acquiring ? "opacity-40" : "opacity-100"} transition-opacity duration-700`}>
      <Plate asset={shown} running={running} hasResult={hasResult} />
    </div>
  );
}

import { X as XIcon } from "lucide-react";

function Plate({ asset, label, running, hasResult }: { asset: Asset; label?: string; running?: boolean; hasResult?: boolean }) {
  const sar = asset.modality === "sar";
  const s = useSatQuery();
  const isCustom = s.mission?.id === "upload";

  return (
    <div className="relative h-full w-full overflow-hidden bg-[#0a0c10]">
      <img
        src={asset.src}
        alt={asset.name}
        className={`h-full w-full object-cover ${sar ? "contrast-125 saturate-50" : ""} ${running ? "brightness-75" : ""}`}
        draggable={false}
      />
      {sar && !hasResult && <div className="speckle absolute inset-0" />}
      {isCustom && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            s.removeAsset(asset.id);
          }}
          title="Remove image"
          className="absolute top-2 right-2 flex h-6 w-6 items-center justify-center rounded-full bg-black/60 text-white/70 backdrop-blur-sm transition-colors hover:bg-red-500/80 hover:text-white"
        >
          <XIcon size={14} />
        </button>
      )}
    </div>
  );
}

function CompareSwitch({
  mode,
  compare,
  setCompare,
  hasSecondary,
}: {
  mode: string;
  compare: CompareMode;
  setCompare: (c: CompareMode) => void;
  hasSecondary: boolean;
}) {
  if (!hasSecondary) return <span className="text-faint">one frame</span>;
  const opts: { id: CompareMode; label: string }[] =
    mode === "bi-temporal"
      ? [
          { id: "primary", label: "t₀" },
          { id: "secondary", label: "t₁" },
          { id: "swipe", label: "Swipe" },
          { id: "split", label: "Split" },
          { id: "diff", label: "Diff" },
        ]
      : [
          { id: "primary", label: "Optical" },
          { id: "secondary", label: "SAR" },
          { id: "swipe", label: "Swipe" },
          { id: "split", label: "Split" },
        ];
  return (
    <div className="flex rounded-md bg-white/5 p-0.5">
      {opts.map((o) => (
        <button
          key={o.id}
          type="button"
          onClick={() => setCompare(o.id)}
          className={`rounded px-2 py-0.5 text-[11px] ${
            compare === o.id ? "bg-panel text-brass" : "text-faint hover:text-mute"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}


