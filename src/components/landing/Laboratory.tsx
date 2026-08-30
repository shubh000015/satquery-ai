import { LABS } from "./destinations";

export function Laboratory() {
  return (
    <section id="features" className="relative bg-black px-6 py-24 md:px-10 md:py-32">
      <div className="mx-auto max-w-[1100px]">
        <h2 className="font-sans text-center text-[clamp(2.2rem,5vw,4.2rem)] font-semibold tracking-[0.14em] text-[#00D964]">
          Features
        </h2>

        <ul className="mt-16 divide-y divide-white/10 border-y border-white/10 px-4 md:px-8">
          {LABS.map((lab) => (
            <li
              key={lab.no}
              className="grid grid-cols-1 items-baseline gap-4 py-8 text-left md:grid-cols-[4.5rem_17rem_1fr]"
            >
              <span className="text-[11px] tracking-[0.28em] text-[#bcbcbc] uppercase">{lab.no}</span>
              <span className="font-sans text-[1.35rem] font-medium tracking-[0.06em] text-[#fafafa] md:text-[1.45rem]">
                {lab.title}
              </span>
              <span className="text-[14px] leading-relaxed text-[#9e9e9e] md:text-[15px]">
                {lab.body}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}




