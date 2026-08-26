"use client";

import { DESTINATIONS, type Destination } from "./destinations";

export function Missions({
  onLaunch,
}: {
  active: Destination;
  onSelect: (id: string) => void;
  onLaunch: (id: string) => void;
}) {
  return (
    <section id="journeys" className="relative bg-black px-6 py-24 md:px-10 md:py-32">
      <div className="mx-auto max-w-[1200px] text-center">
        <p className="font-display text-[12px] font-normal tracking-[0.42em] text-white/50">
          Demo scenes
        </p>
        <h2 className="font-display mt-3 text-[clamp(2.2rem,5vw,4.2rem)] font-semibold tracking-[0.14em]">
          Every required task
        </h2>
        <p className="mx-auto mt-4 max-w-xl text-[14px] text-white/50">
          Four Indian scenes. Open a chat, ask in English, get grounded evidence.
        </p>
      </div>

      <div className="mx-auto mt-16 grid max-w-[1200px] gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {DESTINATIONS.map((d) => (
          <button
            key={d.id}
            type="button"
            onClick={() => onLaunch(d.id)}
            className="group relative aspect-[3/4] overflow-hidden bg-black text-left"
          >
            <img
              src={d.card}
              alt=""
              className="h-full w-full object-cover transition duration-700 group-hover:scale-[1.04]"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black via-black/25 to-transparent" />
            <div className="absolute inset-x-0 bottom-0 p-4">
              <p className="text-[10px] tracking-[0.22em] text-[#00b4ff] uppercase">{d.task}</p>
              <p className="font-display mt-1 text-[14px] tracking-[0.16em] text-white">{d.label}</p>
              <p className="mt-1 text-[11px] text-white/50">{d.proves}</p>
            </div>
          </button>
        ))}
      </div>
    </section>
  );
}
