"use client";

import { useRef } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { LABS } from "./destinations";

export function Laboratory() {
  const root = useRef<HTMLElement>(null);

  useGSAP(
    () => {
      gsap.registerPlugin(ScrollTrigger);

      const card = root.current?.querySelector<HTMLElement>("[data-feature-card]");
      const stage = card?.parentElement;
      if (!card || !stage) return;

      const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      if (reduced) {
        gsap.set(card, { rotateX: 0, y: 0, width: "100%" });
        return;
      }

      gsap.fromTo(
        card,
        {
          width: "68%",
          rotateX: 22,
          y: 48,
          transformOrigin: "50% 80%",
        },
        {
          width: "100%",
          rotateX: 0,
          y: 0,
          ease: "none",
          scrollTrigger: {
            trigger: stage,
            start: "top 88%",
            end: "center 48%",
            scrub: 0.7,
            invalidateOnRefresh: true,
          },
        },
      );
    },
    { scope: root },
  );

  return (
    <section
      ref={root}
      id="features"
      className="relative overflow-x-hidden bg-black px-6 py-24 md:px-10 md:py-32"
    >
      <div className="mx-auto max-w-[1100px]">
        <h2 className="font-sans text-center text-[clamp(2.2rem,5vw,4.2rem)] font-bold tracking-[0.14em] text-[#00D964]">
          Features
        </h2>

        <div
          className="mt-16 flex min-h-[36rem] items-center md:mt-20 md:min-h-[46rem]"
          style={{ perspective: "1100px" }}
        >
          <div
            data-feature-card
            className="mx-auto w-[68%] origin-[50%_80%] border border-white/10 will-change-transform [transform:translateY(48px)_rotateX(22deg)] motion-reduce:w-full motion-reduce:[transform:none]"
          >
            <ul className="divide-y divide-white/10 px-4 md:px-8">
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
        </div>
      </div>
    </section>
  );
}
