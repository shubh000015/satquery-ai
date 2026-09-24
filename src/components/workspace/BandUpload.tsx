"use client";

import { S2_BANDS, SAR_BANDS, type UploadKind } from "@/lib/bandPlan";
import { useSatQuery } from "@/lib/store";

export function BandKindDialog({ onChosen }: { onChosen?: (kind: UploadKind) => void }) {
  const s = useSatQuery();
  if (!s.kindDialog) return null;

  const pick = (kind: UploadKind) => {
    s.chooseUploadKind(kind);
    onChosen?.(kind);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4">
      <div className="w-[min(480px,92vw)] rounded-xl border border-white/15 bg-[#000b1a] p-8">
        <p className="text-[11px] tracking-[0.28em] text-[#bcbcbc] uppercase">Scene type</p>
        <p className="mt-3 text-2xl tracking-[0.04em] text-white">What are you uploading?</p>
        <p className="mt-2 text-[14px] text-[#9e9e9e]">
          Optical MSI is 12 Sentinel-2 band GeoTIFFs. SAR is VV and VH. Specialists are much more
          accurate when the full stack is present.
        </p>
        <div className="mt-7 flex flex-col gap-3">
          <button
            type="button"
            onClick={() => pick("optical")}
            className="bg-white py-3 text-[12px] tracking-[0.16em] text-black uppercase"
          >
            Optical — 12 spectral bands
          </button>
          <button
            type="button"
            onClick={() => pick("sar")}
            className="border border-white/20 py-3 text-[12px] tracking-[0.16em] uppercase text-white"
          >
            SAR — VV + VH
          </button>
          <button
            type="button"
            onClick={() => pick("fusion")}
            className="border border-white/20 py-3 text-[12px] tracking-[0.16em] uppercase text-white"
          >
            Optical + SAR fusion
          </button>
        </div>
        <button
          type="button"
          onClick={s.cancelKindDialog}
          className="mt-4 text-[11px] tracking-[0.2em] text-[#9e9e9e] uppercase"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

export function BandChecklist({ onAdd }: { onAdd?: () => void }) {
  const s = useSatQuery();
  const plan = s.bandPlan;
  if (!plan || !plan.scenes.length) return null;
  if (plan.ready && plan.scenes.every((sc) => sc.kind === "rgb")) return null;

  return (
    <div className="rounded-xl border border-sat-hairline bg-[#1c1c1c]/70 p-4">
      <h3 className="mb-2 text-[12px] font-bold tracking-wide text-sat-heading uppercase">
        Band stack
      </h3>
      <div className="space-y-3">
        {plan.scenes.map((scene) => {
          const required = scene.kind === "sar" ? SAR_BANDS : S2_BANDS;
          const have = new Set(scene.names);
          return (
            <div key={`${scene.kind}-${scene.label}`}>
              <p className="text-[13px] text-sat-heading">{scene.label}</p>
              {scene.kind !== "rgb" ? (
                <div className="mt-2 flex flex-wrap gap-1">
                  {required.map((band) => (
                    <span
                      key={band}
                      className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${
                        have.has(band)
                          ? "bg-sat-green/20 text-sat-green"
                          : "bg-white/5 text-sat-subtitle"
                      }`}
                    >
                      {band}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="mt-1 text-[12px] text-sat-subtitle">
                  RGB preview only. Add the 12 Sentinel-2 GeoTIFFs for a full optical scene.
                </p>
              )}
            </div>
          );
        })}
      </div>
      {!plan.ready ? (
        <p className="mt-3 text-[12px] text-sat-subtitle">{plan.message}</p>
      ) : null}
      {onAdd && !plan.ready ? (
        <button
          type="button"
          onClick={onAdd}
          className="mt-3 text-[12px] tracking-wide text-sat-heading underline"
        >
          Add remaining band files
        </button>
      ) : null}
    </div>
  );
}
