"use client";

import { useState, useEffect } from "react";
import { MessageSquare, X } from "lucide-react";
import { HistorySidebar } from "./HistorySidebar";
import { ImageryStage } from "./ImageryStage";
import { SatelliteQAPanel } from "./SatelliteQAPanel";

export function Workspace() {
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
        {/* Simple header for mobile toggle */}
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-sat-hairline bg-sat-bg px-4 md:hidden">
          <button onClick={() => setHistoryOpen(true)} className="text-sat-nav">
            <span className="sr-only">Open History</span>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <span className="text-[13px] font-semibold text-sat-heading">SatQuery Workspace</span>
          <div className="w-6" /> {/* Spacer */}
        </header>
        
        <div className="relative flex-1 overflow-hidden">
          <ImageryStage />
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


