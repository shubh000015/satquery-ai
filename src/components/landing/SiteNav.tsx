"use client";

import { useSatQuery } from "@/lib/store";

export function SiteNav() {
  const { setScreen } = useSatQuery();
  const link =
    "text-[12px] font-sans tracking-[0.28em] text-[#9e9e9e] uppercase hover:text-[#00D964] transition-colors cursor-pointer";

  return (
    <header className="pointer-events-none fixed inset-x-0 top-0 z-40">
      <div className="pointer-events-auto mx-auto grid h-20 max-w-[1280px] grid-cols-[auto_1fr_auto] md:grid-cols-3 items-center px-6 md:px-10">
        <div className="text-[12px] font-sans tracking-[0.28em] text-[#9e9e9e] uppercase">
          SIH26167
        </div>
        <nav className="flex items-center justify-center gap-8 sm:gap-14">
          <a href="#features" className={link}>
            Features
          </a>
          <a href="#models" className={link}>
            Models
          </a>
          <button
            type="button"
            onClick={() => setScreen("workspace")}
            className={link}
          >
            Chat
          </button>
        </nav>
        <div className="hidden md:block"></div>
      </div>
    </header>
  );
}





