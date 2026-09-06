import { PS_COVER } from "./destinations";

export function Brief() {
  return (
    <section id="models" className="relative bg-black px-6 py-20 md:px-10">
      <div className="mx-auto max-w-[1200px]">
        <h2 className="font-sans text-center text-[clamp(1.8rem,4vw,3.4rem)] font-bold tracking-[0.12em] text-[#00D964]">
          Models
        </h2>

        <div className="mt-14 grid gap-px bg-white/10 sm:grid-cols-2 lg:grid-cols-4">
          {PS_COVER.map((row) => (
            <div key={row.k} className="bg-black px-6 py-8 text-center flex flex-col gap-2">
              <p className="font-sans text-[15px] tracking-[0.18em] text-[#bcbcbc] font-medium">{row.k}</p>
              <p className="text-[16px] font-medium tracking-[0.1em] text-[#00D964]">{row.model}</p>
              <p className="mt-2 text-[14px] leading-relaxed text-[#9e9e9e]">{row.v}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}





