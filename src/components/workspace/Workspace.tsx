"use client";

import { useState, useEffect } from "react";
import { MessageSquare, X } from "lucide-react";
import { HistorySidebar } from "./HistorySidebar";
import { SatelliteQAPanel } from "./SatelliteQAPanel";
import { ImageryStage } from "./ImageryStage";
import { useSatQuery } from "@/lib/store";

export function Workspace() {
  const s = useSatQuery();
  const m = s.mission;
  
  const [historyOpen, setHistoryOpen] = useState(false);
  const [qaOpen, setQaOpen] = useState(false);
  const [historyCollapsed, setHistoryCollapsed] = useState(false);

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
      <HistorySidebar 
        isOpen={historyOpen} 
        onClose={() => setHistoryOpen(false)} 
        isDesktopCollapsed={historyCollapsed}
        onToggleDesktop={() => setHistoryCollapsed(!historyCollapsed)}
      />

      {/* Middle/Right Column based on presence of image */}
      {!m ? (
        <div className="flex-1 flex flex-col h-full bg-sat-panel-bg">
          <div className="flex items-center p-4 md:hidden">
            <button onClick={() => setHistoryOpen(true)} className="text-sat-nav">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 6h16M4 12h16M4 18h16" /></svg>
            </button>
          </div>
          <div className="flex-1 overflow-hidden max-w-4xl mx-auto w-full">
            <SatelliteQAPanel isFullWidth />
          </div>
        </div>
      ) : (
        <>
          <main className="relative flex min-w-0 flex-1 flex-col">
            {/* Mobile History Toggle */}
            <div className="absolute top-4 left-4 z-[60] md:hidden">
              <button onClick={() => setHistoryOpen(true)} className="rounded-md bg-black/50 p-2 text-white backdrop-blur">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
            </div>
            
            <div className="relative flex-1 overflow-hidden bg-[#0a0c10]">
              <ImageryStage />
            </div>
          </main>

          {/* Right Column: Q&A Panel (Desktop) */}
          <div className="hidden md:block">
            <SatelliteQAPanel />
          </div>
        </>
      )}

      {/* Mobile Q&A Toggle */}
      {m && (
        <button 
          onClick={() => setQaOpen(true)}
          className="fixed bottom-6 right-6 z-30 flex h-14 w-14 items-center justify-center rounded-full bg-sat-heading text-sat-bg shadow-xl md:hidden"
        >
          <MessageSquare size={24} className="fill-sat-bg" />
        </button>
      )}

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


