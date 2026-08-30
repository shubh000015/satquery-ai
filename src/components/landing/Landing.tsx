"use client";

import { useRef, useState } from "react";
import { useSatQuery } from "@/lib/store";
import { Hero } from "./Hero";
import { Brief } from "./Brief";
import { SiteFooter } from "./Join";
import { Laboratory } from "./Laboratory";
import { SiteNav } from "./SiteNav";

export function Landing() {
  const { ingestFiles, pairChoice, confirmPair, cancelPair, setScreen } = useSatQuery();
  const inputRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);

  const upload = () => inputRef.current?.click();

  const handleAsk = (query: string) => {
    // If they ask a query without uploading an image, we can just switch to workspace, or alert them.
    // Let's just switch to workspace so they are in the chat.
    setScreen("workspace");
  };

  return (
    <div
      className="relative min-h-dvh bg-black text-[#fafafa]"
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
        onAsk={handleAsk}
        onUpload={upload}
      />
      <Laboratory />
      <Brief />
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
            <p className="text-[11px] tracking-[0.28em] text-[#bcbcbc] uppercase">Pair detected</p>
            <p className="font-sans mt-3 text-3xl tracking-[0.08em]">Two files</p>
            <p className="mt-2 text-[14px] text-[#9e9e9e]">
              Same place across sensors — or the same place across time.
            </p>
            <div className="mt-7 grid grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => confirmPair("cross-modal")}
                className="bg-white py-3 text-[12px] tracking-[0.2em] text-black uppercase"
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
              className="mt-4 text-[11px] tracking-[0.2em] text-[#9e9e9e] uppercase"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}





