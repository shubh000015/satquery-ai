import { ArrowUp, Target, Activity, FileSearch, Layers } from "lucide-react";

export function SatelliteQAPanel({ isMobile }: { isMobile?: boolean }) {
  return (
    <aside className={`flex h-full w-full flex-col bg-sat-panel-bg ${isMobile ? "" : "border-l border-sat-hairline md:w-[400px] lg:w-[450px]"}`}>
      
      {/* Header */}
      {!isMobile && (
        <header className="flex h-14 shrink-0 items-center gap-2 border-b border-sat-hairline px-5">
          <Target size={16} className="text-sat-heading" />
          <h2 className="text-sm font-medium text-sat-heading">Analysis & Queries</h2>
          <div className="ml-auto flex items-center gap-1.5">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sat-green opacity-75"></span>
              <span className="relative inline-flex h-2 w-2 rounded-full bg-sat-green"></span>
            </span>
            <span className="text-[11px] text-sat-subtitle uppercase tracking-wide">Model Active</span>
          </div>
        </header>
      )}

      {/* Analysis Content */}
      <div className="flex-1 overflow-y-auto px-5 py-6">
        <div className="flex flex-col gap-6">
          
          <div className="rounded-xl border border-sat-hairline bg-sat-box p-5 shadow-sm">
            <h3 className="mb-4 flex items-center gap-2 text-[14px] font-semibold text-sat-heading">
              <FileSearch size={16} className="text-sat-nav" />
              Target Query
            </h3>
            <p className="text-[14px] leading-relaxed text-sat-msg-text">
              "Identify flooded areas in this region and show the affected settlements."
            </p>
          </div>

          <div className="rounded-xl border border-sat-hairline bg-sat-box p-5 shadow-sm">
            <h3 className="mb-4 flex items-center gap-2 text-[14px] font-semibold text-sat-heading">
              <Activity size={16} className="text-sat-nav" />
              Extraction Results
            </h3>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-sat-hairline pb-2">
                <span className="text-[13px] text-sat-nav">Flooded Area</span>
                <span className="text-[13px] font-medium text-sat-heading">12.48 km²</span>
              </div>
              <div className="flex items-center justify-between border-b border-sat-hairline pb-2">
                <span className="text-[13px] text-sat-nav">Affected Settlements</span>
                <span className="text-[13px] font-medium text-sat-heading">7 locations</span>
              </div>
              <div className="flex items-center justify-between border-b border-sat-hairline pb-2">
                <span className="text-[13px] text-sat-nav">Confidence Score</span>
                <span className="text-[13px] font-medium text-sat-green">94.6%</span>
              </div>
            </div>
            
            <div className="mt-5 text-[13px] leading-relaxed text-sat-subtitle">
              <p className="mb-2">• Major flooding detected along low-lying regions.</p>
              <p className="mb-2">• SAR data confirms optical observations.</p>
              <p>• Cloud impact was reduced through multimodal fusion.</p>
            </div>
          </div>

          <div className="rounded-xl border border-sat-hairline bg-sat-box p-5 shadow-sm">
            <h3 className="mb-4 flex items-center gap-2 text-[14px] font-semibold text-sat-heading">
              <Layers size={16} className="text-sat-nav" />
              Suggested Actions
            </h3>
            <div className="flex flex-col gap-2">
              <button className="flex w-full items-center justify-between rounded-lg bg-[#1c1c1c] px-3 py-2 text-left transition-colors hover:bg-sat-hairline">
                <span className="text-[13px] text-sat-msg-text">Calculate total affected population</span>
                <ArrowUp size={14} className="rotate-45 text-sat-nav" />
              </button>
              <button className="flex w-full items-center justify-between rounded-lg bg-[#1c1c1c] px-3 py-2 text-left transition-colors hover:bg-sat-hairline">
                <span className="text-[13px] text-sat-msg-text">Compare with pre-flood SAR image</span>
                <ArrowUp size={14} className="rotate-45 text-sat-nav" />
              </button>
            </div>
          </div>

        </div>
      </div>

      {/* Input Area */}
      <div className="shrink-0 p-4 border-t border-sat-hairline">
        <div className="mb-3 text-[11px] font-medium tracking-wide text-sat-nav uppercase">
          Follow-up Query
        </div>
        <div className="flex items-center gap-2 rounded-lg bg-sat-input-bg px-2 py-1.5 shadow-[0_2px_12px_rgba(0,0,0,0.5)]">
          <input 
            type="text" 
            placeholder="Ask about this satellite scene..."
            className="flex-1 bg-transparent px-2 text-[14px] text-sat-bg placeholder:text-[#6b6b6d] focus:outline-none"
          />
          <button className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-sat-bg text-white hover:opacity-90">
            <ArrowUp size={16} strokeWidth={2.5} />
          </button>
        </div>
      </div>

    </aside>
  );
}

