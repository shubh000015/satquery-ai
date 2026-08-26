"use client";

export function Join({
  drag,
  onUpload,
}: {
  drag: boolean;
  onUpload: () => void;
}) {
  return (
    <section id="join" className="relative bg-black px-6 pb-24 md:px-10 md:pb-32">
      <div
        className={`mx-auto max-w-[1200px] border px-8 py-20 text-center md:px-16 ${
          drag ? "border-[#00b4ff] bg-[#00b4ff]/10" : "border-white/15"
        }`}
      >
        <p className="font-display text-[12px] font-normal tracking-[0.42em] text-white/50">
          User scene
        </p>
        <h2 className="font-display mt-4 text-[clamp(2.2rem,5vw,4rem)] tracking-[0.12em]">
          Bring a GeoTIFF
        </h2>
        <p className="mx-auto mt-5 max-w-md text-[15px] leading-relaxed text-white/50">
          One frame or a pair. Optical + SAR, or two dates of the same footprint. The agent classifies the query the same way.
        </p>
        <button
          type="button"
          onClick={onUpload}
          className="cyan-glow mt-10 bg-[#00b4ff] px-10 py-3.5 text-[12px] font-semibold tracking-[0.28em] text-black uppercase hover:bg-[#5ad2ff]"
        >
          Upload scene
        </button>
      </div>
    </section>
  );
}

export function SiteFooter() {
  return (
    <footer className="border-t border-white/10 bg-black px-6 py-8 md:px-10">
      <div className="mx-auto flex max-w-[1200px] flex-col items-center justify-between gap-4 md:flex-row">
        <p className="font-display text-[14px] tracking-[0.36em]">Satquery</p>
        <p className="text-center text-[11px] tracking-[0.18em] text-white/40 uppercase">
          SIH26167 · ISRO · VRSBench · Smart India Hackathon 2026
        </p>
      </div>
    </footer>
  );
}
