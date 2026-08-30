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
          drag ? "border-[#fafafa] bg-white/10" : "border-white/15"
        }`}
      >
        <p className="font-sans text-[12px] font-normal tracking-[0.42em] text-[#9e9e9e]">
          User scene
        </p>
        <h2 className="font-sans mt-4 text-[clamp(2.2rem,5vw,4rem)] tracking-[0.12em]">
          Bring a GeoTIFF
        </h2>
        <p className="mx-auto mt-5 max-w-md text-[15px] leading-relaxed text-[#9e9e9e]">
          One frame or a pair. Optical + SAR, or two dates of the same footprint. The agent classifies the query the same way.
        </p>
        <button
          type="button"
          onClick={onUpload}
          className=" mt-10 bg-white px-10 py-3.5 text-[12px] font-semibold tracking-[0.28em] text-black uppercase hover:bg-gray-200"
        >
          Upload scene
        </button>

        <div className="mx-auto mt-12 max-w-lg border border-[#fafafa]/10 bg-black/50 p-5 text-left shadow-lg">
          <p className="mb-3 text-[12px] font-semibold tracking-wide text-[#fafafa] uppercase">
            Accepted formats for SIH problem statement:
          </p>
          <ul className="space-y-2 text-[13px] text-[#9e9e9e]">
            <li className="flex items-center gap-2">
              <span>✅</span> <span><strong>GeoTIFF</strong> (.tif / .tiff) — supported</span>
            </li>
            <li className="flex items-center gap-2">
              <span>✅</span> <span><strong>TIFF</strong> (.tif / .tiff) — supported</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="mt-0.5">⚠️</span> 
              <span><strong>PNG</strong> (.png) — only for the prescribed public benchmark datasets</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="mt-0.5">⚠️</span> 
              <span><strong>JPEG/JPG</strong> (.jpg / .jpeg) — only for the prescribed public benchmark datasets</span>
            </li>
          </ul>
        </div>
      </div>
    </section>
  );
}

export function SiteFooter() {
  return (
    <footer className="border-t border-white/10 bg-black px-6 py-8 md:px-10">
      <div className="mx-auto flex max-w-[1200px] flex-col items-center justify-between gap-4 md:flex-row">
        <p className="font-sans text-[14px] font-bold tracking-[0.36em]">SATQUERY</p>
        <p className="text-center text-[11px] tracking-[0.18em] text-[#9e9e9e] uppercase">
          SIH · ISRO · Smart India Hackathon 2026
        </p>
      </div>
    </footer>
  );
}





