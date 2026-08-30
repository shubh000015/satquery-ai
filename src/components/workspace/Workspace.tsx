"use client";

import { useState, useEffect } from "react";
import { MessageSquare, X } from "lucide-react";
import { HistorySidebar } from "./HistorySidebar";
import { SatelliteQAPanel } from "./SatelliteQAPanel";
import { useSatQuery } from "@/lib/store";

export function Workspace() {
  const s = useSatQuery();
  const m = s.mission;
  
  const [historyOpen, setHistoryOpen] = useState(false);
  const [qaOpen, setQaOpen] = useState(false);

  // Close sidebars on resize to desktop
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 768) {
        setHistoryOpen(false);
        setQaOpen(false);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div className="flex h-dvh w-full overflow-hidden bg-sat-bg text-sat-heading font-sans selection:bg-white/20 selection:text-white">
      {/* Left Column: History */}
      <HistorySidebar isOpen={historyOpen} onClose={() => setHistoryOpen(false)} />

      {/* Middle Column: Satellite Imagery Viewer */}
      <main className="flex min-w-0 flex-1 flex-col">
        {/* Header with features/details */}
        <header className="flex h-14 shrink-0 items-center gap-3 border-b border-sat-hairline bg-sat-bg px-4">
          <button onClick={() => setHistoryOpen(true)} className="text-sat-nav md:hidden">
            <span className="sr-only">Open History</span>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          
          <div className="hidden h-4 w-px bg-sat-hairline md:block" />
          
          <div className="min-w-0 flex-1">
            <p className="truncate text-[13px] font-medium text-sat-heading">
              {m ? m.title : "Earth Observation"}
            </p>
            <p className="truncate font-mono text-[10px] text-sat-subtitle">
              {m ? `${m.location} · ${m.mode === "cross-modal" ? "optical + SAR" : m.mode === "bi-temporal" ? "bi-temporal" : "single optical"}` : "Global View · High Resolution"}
            </p>
          </div>

          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              title="Measure"
              onClick={() => s.setMeasuring(!s.measuring)}
              className={`rounded-md p-1.5 transition-colors ${s.measuring ? "bg-sat-hairline text-sat-heading" : "text-sat-nav hover:bg-sat-hairline/50 hover:text-sat-heading"}`}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
            </button>
            <button
              type="button"
              title="Export report"
              onClick={() => s.setReportOpen(true)}
              className="rounded-md p-1.5 text-sat-nav transition-colors hover:bg-sat-hairline/50 hover:text-sat-heading"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
            </button>
          </div>
        </header>
        
        <div className="relative flex-1 overflow-hidden bg-black">
          <img
            src="https://images.unsplash.com/photo-1614730321146-b6fa6a46bcb4?q=100&w=3840&auto=format&fit=crop"
            alt="4K Earth"
            className="h-full w-full object-cover object-center"
          />
        </div>
      </main>

      {/* Right Column: Q&A Panel (Desktop) */}
      <div className="hidden md:block">
        <SatelliteQAPanel />
      </div>

      {/* Mobile Q&A Toggle */}
      <button 
        onClick={() => setQaOpen(true)}
        className="fixed bottom-6 right-6 z-30 flex h-14 w-14 items-center justify-center rounded-full bg-sat-heading text-sat-bg shadow-xl md:hidden"
      >
        <MessageSquare size={24} className="fill-sat-bg" />
      </button>

      {/* Mobile Q&A Drawer */}
      {qaOpen && (
        <div className="fixed inset-0 z-50 flex flex-col bg-sat-bg md:hidden">
          <header className="flex h-14 shrink-0 items-center justify-between border-b border-sat-hairline px-4">
            <h2 className="text-sm font-medium text-sat-heading">Analysis</h2>
            <button onClick={() => setQaOpen(false)} className="text-sat-nav">
              <X size={20} />
            </button>
          </header>
          <div className="flex-1 overflow-hidden">
            <SatelliteQAPanel isMobile />
          </div>
        </div>
      )}
    </div>
  );
}


