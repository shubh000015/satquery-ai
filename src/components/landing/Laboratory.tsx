import { LABS } from "./destinations";

export function Laboratory() {
  return (
    <section id="laboratory" className="relative bg-black px-6 py-24 md:px-10 md:py-32">
      <div className="mx-auto max-w-[1200px]">
        <p className="font-sans text-center text-[12px] font-normal tracking-[0.42em] text-[#9e9e9e]">
          Agentic routing
        </p>
        <h2 className="font-sans mt-3 text-center text-[clamp(2.2rem,5vw,4.2rem)] tracking-[0.14em]">
          Five specialists
        </h2>
        <p className="mx-auto mt-4 max-w-xl text-center text-[14px] text-[#9e9e9e]">
          The sentence picks the model. Captioning, VQA, grounding, change, fusion — never a manual toolbox.
        </p>

        <ul className="mt-16 divide-y divide-white/10 border-y border-white/10">
          {LABS.map((lab) => (
            <li
              key={lab.no}
              className="grid grid-cols-1 items-baseline gap-3 py-8 md:grid-cols-[4.5rem_16rem_1fr]"
            >
              <span className="text-[11px] tracking-[0.28em] text-[#bcbcbc]">{lab.no}</span>
              <span className="font-sans text-[1.5rem] tracking-[0.1em]">{lab.title}</span>
              <span className="max-w-xl text-[15px] leading-relaxed text-[#9e9e9e]">{lab.body}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}




