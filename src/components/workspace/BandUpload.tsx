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
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/75 p-4 backdrop-blur-sm">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="band-upload-title"
        className="w-[min(520px,92vw)] border border-white/[0.065] bg-[#111111] p-6 shadow-[0_2px_40px_rgba(0,0,0,0.45)] sm:p-8"
      >
        <p className="text-[11px] font-medium tracking-[0.18em] text-[#9e9e9e] uppercase">Scene type</p>
        <h2 id="band-upload-title" className="mt-3 text-[clamp(1.65rem,5vw,2rem)] font-medium leading-tight tracking-[-0.02em] text-[#fafafa]">
          What are you uploading?
        </h2>
        <p className="mt-3 text-[14px] leading-relaxed text-[#9e9e9e]">
          Optical MSI is 12 Sentinel-2 band GeoTIFFs. SAR is VV and VH. Specialists are much more
          accurate when the full stack is present.
        </p>
        <div className="mt-7 flex flex-col gap-2.5">
          <button
            type="button"
            onClick={() => pick("optical")}
            className="rounded-full bg-white px-5 py-3.5 text-left text-[14px] font-medium text-[#0c0c0c] transition-colors hover:bg-[#ededed] active:scale-[0.99]"
          >
            Optical MSI <span className="ml-1 font-normal text-black/60">· 12 spectral bands</span>
          </button>
          <button
            type="button"
            onClick={() => pick("sar")}
            className="rounded-full border border-white/[0.12] bg-white/[0.028] px-5 py-3.5 text-left text-[14px] font-medium text-[#fafafa] transition-colors hover:bg-white/[0.07] active:scale-[0.99]"
          >
            SAR <span className="ml-1 font-normal text-[#9e9e9e]">· VV + VH bands</span>
          </button>
          <button
            type="button"
            onClick={() => pick("fusion")}
            className="rounded-full border border-white/[0.12] bg-white/[0.028] px-5 py-3.5 text-left text-[14px] font-medium text-[#fafafa] transition-colors hover:bg-white/[0.07] active:scale-[0.99]"
          >
            Optical + SAR <span className="ml-1 font-normal text-[#9e9e9e]">· fusion</span>
          </button>
        </div>
        <button
          type="button"
          onClick={s.cancelKindDialog}
          className="mt-5 rounded-full px-1 py-1 text-[13px] text-[#9e9e9e] transition-colors hover:text-[#f2f2f2]"
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
