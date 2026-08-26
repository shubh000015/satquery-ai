import { PS_COVER } from "./destinations";

export function Brief() {
  return (
    <section id="brief" className="relative bg-black px-6 py-20 md:px-10">
      <div className="mx-auto max-w-[1200px]">
        <p className="font-display text-center text-[12px] font-normal tracking-[0.42em] text-[#00b4ff]">
          SIH26167
        </p>
        <h2 className="font-display mt-3 text-center text-[clamp(1.8rem,4vw,3.4rem)] tracking-[0.12em]">
          SatQuery AI
        </h2>
        <p className="mx-auto mt-5 max-w-2xl text-center text-[16px] leading-relaxed text-white/60">
          An interactive vision-language assistant for multimodal remote sensing
          image analysis through text queries — optical, SAR, or two dates.
        </p>

        <div className="mt-14 grid gap-px bg-white/10 sm:grid-cols-2 lg:grid-cols-4">
          {PS_COVER.map((row) => (
            <div key={row.k} className="bg-black px-6 py-8">
              <p className="font-display text-[15px] tracking-[0.18em] text-[#00b4ff]">{row.k}</p>
              <p className="mt-3 text-[14px] leading-relaxed text-white/55">{row.v}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
