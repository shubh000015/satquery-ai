"use client";

import { useRef, useState } from "react";
import { DESTINATIONS } from "./destinations";
import { useSatQuery } from "@/lib/store";
import { Hero } from "./Hero";
import { Brief } from "./Brief";
import { Join, SiteFooter } from "./Join";
import { Laboratory } from "./Laboratory";
import { Missions } from "./Missions";
import { SiteNav } from "./SiteNav";

export function Landing() {
  const { openMission, ingestFiles, pairChoice, confirmPair, cancelPair } = useSatQuery();
  const inputRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [activeId, setActiveId] = useState<string>(DESTINATIONS[0].id);
  const active = DESTINATIONS.find((d) => d.id === activeId) ?? DESTINATIONS[0];

  const upload = () => inputRef.current?.click();

  return (
    <div
      className="relative min-h-dvh bg-black text-white"
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        if (e.dataTransfer.files.length) ingestFiles(e.dataTransfer.files);
      }}
    >
      <SiteNav />
      <Hero
        active={active}
        onSelect={setActiveId}
        onAsk={(id, q) => openMission(id, q ? { query: q } : {})}
      />
      <Brief />
      <Missions active={active} onSelect={setActiveId} onLaunch={(id) => openMission(id)} />
      <Laboratory />
      <Join drag={drag} onUpload={upload} />
      <SiteFooter />

      <input
        ref={inputRef}
        type="file"
        accept=".tif,.tiff,.png,.jpg,.jpeg,.webp,image/*"
        multiple
        className="hidden"
        onChange={(e) => {
          if (e.target.files?.length) ingestFiles(e.target.files);
          e.target.value = "";
        }}
      />

      {pairChoice && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4">
          <div className="w-[min(440px,92vw)] border border-white/15 bg-[#000b1a] p-8">
            <p className="text-[11px] tracking-[0.28em] text-[#00b4ff] uppercase">Pair detected</p>
            <p className="font-display mt-3 text-3xl tracking-[0.08em]">Two files</p>
            <p className="mt-2 text-[14px] text-white/55">
              Same place across sensors — or the same place across time.
            </p>
            <div className="mt-7 grid grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => confirmPair("cross-modal")}
                className="bg-[#00b4ff] py-3 text-[12px] tracking-[0.2em] text-black uppercase"
              >
                Optical × SAR
              </button>
              <button
                type="button"
                onClick={() => confirmPair("bi-temporal")}
                className="border border-white/20 py-3 text-[12px] tracking-[0.2em] uppercase"
              >
                Two epochs
              </button>
            </div>
            <button
              type="button"
              onClick={cancelPair}
              className="mt-4 text-[11px] tracking-[0.2em] text-white/40 uppercase"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
