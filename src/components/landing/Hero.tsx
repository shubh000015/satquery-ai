"use client";

import { useState } from "react";
import { DESTINATIONS, type Destination } from "./destinations";
import { Paperclip } from "lucide-react";
import { Typewriter } from "./Typewriter";

export function Hero({
  onAsk,
  onUpload,
}: {
  onAsk: (query: string) => void;
  onUpload?: () => void;
}) {
  const [draft, setDraft] = useState("");

  return (
    <section id="top" className="relative h-dvh min-h-[760px] overflow-hidden bg-black">
      <img
        src="/space/satellite-hero.jpg"
        alt=""
        className="absolute inset-0 h-full w-full object-cover object-[center_30%]"
      />
      <div className="absolute inset-0 bg-gradient-to-b from-black/55 via-black/15 to-black/80" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_40%,rgba(0,0,0,0.55)_100%)]" />

      <div className="relative z-10 flex h-full flex-col">
        <div className="flex flex-1 flex-col items-center justify-center px-6 text-center">
          <h1 className="font-sans mt-3 text-[clamp(3rem,10vw,8rem)] font-bold leading-[0.84] tracking-[0.06em] text-[#fafafa]">
            SatQuery AI
          </h1>
          <Typewriter />
        </div>

        <div id="book" className="mx-auto mb-16 w-[min(860px,calc(100%-2rem))] scroll-mt-24">
          <form
            className="flex items-center gap-2 rounded-2xl border border-white/25 bg-black/45 px-3 py-2.5 backdrop-blur-md"
            onSubmit={(e) => {
              e.preventDefault();
              onAsk(draft.trim());
            }}
          >
            {onUpload && (
              <button
                type="button"
                onClick={onUpload}
                title="Attach images"
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-[#9e9e9e] transition-colors hover:bg-white/10 hover:text-white"
              >
                <Paperclip size={18} />
              </button>
            )}
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  onAsk(draft.trim());
                }
              }}
              rows={1}
              placeholder="Upload a satellite scene and ask a query..."
              className="max-h-28 flex-1 resize-none bg-transparent py-1.5 text-[15px] leading-relaxed text-[#fafafa] outline-none placeholder:text-[#9e9e9e]"
            />
            <button
              type="submit"
              className="rounded-full bg-white px-5 py-2.5 text-[12px] font-semibold tracking-[0.2em] text-black uppercase hover:bg-gray-200"
            >
              Ask
            </button>
          </form>
        </div>
      </div>
    </section>
  );
}




