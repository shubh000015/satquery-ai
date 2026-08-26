"use client";

export function SiteNav() {
  const link =
    "text-[11px] tracking-[0.28em] text-white/80 uppercase hover:text-[#00b4ff] transition-colors";
  return (
    <header className="pointer-events-none fixed inset-x-0 top-0 z-40">
      <div className="pointer-events-auto mx-auto grid h-20 max-w-[1280px] grid-cols-[1fr_auto_1fr] items-center px-6 md:px-10">
        <nav className="hidden items-center gap-10 md:flex">
          <a href="#journeys" className={link}>
            Scenes
          </a>
          <a href="#book" className={link}>
            Chat
          </a>
        </nav>
        <a href="#top" className="text-center">
          <span className="font-display block text-[22px] tracking-[0.42em] text-white">
            Satquery
          </span>
          <span className="mt-0.5 hidden text-[9px] tracking-[0.28em] text-[#00b4ff] uppercase sm:block">
            SIH26167 · ISRO
          </span>
        </a>
        <nav className="hidden items-center justify-end gap-10 md:flex">
          <a href="#laboratory" className={link}>
            Instruments
          </a>
          <a href="#join" className={link}>
            Upload
          </a>
        </nav>
      </div>
    </header>
  );
}
