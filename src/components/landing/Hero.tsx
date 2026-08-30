"use client";

import { useState } from "react";
import { DESTINATIONS, type Destination } from "./destinations";

export function Hero({
  active,
  onSelect,
  onAsk,
}: {
  active: Destination;
  onSelect: (id: string) => void;
  onAsk: (id: string, query: string) => void;
}) {
  const [draft, setDraft] = useState("");

  return (
    <section id="top" className="relative h-dvh min-h-[760px] overflow-hidden bg-black">
      <img
        src="/space/hero.png"
        alt=""
        className="absolute inset-0 h-full w-full object-cover object-[center_30%]"
      />
      <div className="absolute inset-0 bg-gradient-to-b from-black/55 via-black/15 to-black/80" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_40%,rgba(0,0,0,0.55)_100%)]" />

      <div className="relative z-10 flex h-full flex-col">
        <div className="flex flex-1 flex-col items-center justify-center px-6 text-center">
          <h1 className="font-sans mt-3 text-[clamp(3rem,10vw,8rem)] font-bold leading-[0.84] tracking-[0.12em] text-[#fafafa]">
            EARTH QUERY
          </h1>
        </div>

        <div id="book" className="mx-auto mb-8 w-[min(860px,calc(100%-2rem))] scroll-mt-24">
          <div className="mb-3 flex flex-wrap justify-center gap-2">
            {DESTINATIONS.map((d) => (
              <button
                key={d.id}
                type="button"
                onClick={() => onSelect(d.id)}
                className={`rounded-full px-3 py-1.5 text-[11px] tracking-[0.16em] uppercase ${
                  d.id === active.id
                    ? "bg-white text-black"
                    : "border border-white/20 text-[#fafafa]/70 hover:text-[#fafafa]"
                }`}
              >
                {d.name}
              </button>
            ))}
          </div>
          <p className="mb-2 text-center text-[11px] tracking-[0.18em] text-[#9e9e9e] uppercase">
            {active.flight} · {active.task} · {active.place}
          </p>
          <form
            className="flex items-end gap-2 rounded-2xl border border-white/25 bg-black/45 px-3 py-2.5 backdrop-blur-md"
            onSubmit={(e) => {
              e.preventDefault();
              onAsk(active.id, draft.trim());
            }}
          >
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  onAsk(active.id, draft.trim());
                }
              }}
              rows={2}
              placeholder={`Ask about ${active.name}… e.g. “Identify flooded areas and show affected settlements.”`}
              className="max-h-28 flex-1 resize-none bg-transparent text-[15px] leading-relaxed text-[#fafafa] outline-none placeholder:text-[#9e9e9e]"
            />
            <button
              type="submit"
              className=" mb-0.5 rounded-full bg-white px-5 py-2.5 text-[12px] font-semibold tracking-[0.2em] text-black uppercase hover:bg-gray-200"
            >
              Ask
            </button>
          </form>
        </div>
      </div>
    </section>
  );
}




