import { ArrowUp, Mic, Sparkles } from "lucide-react";

export function AiChatPanel({ isMobile }: { isMobile?: boolean }) {
  return (
    <aside className={`flex h-full w-full flex-col bg-sat-panel-bg ${isMobile ? "" : "border-l border-sat-hairline md:w-[380px] lg:w-[420px]"}`}>
      
      {/* Header - hide on mobile since drawer has its own header */}
      {!isMobile && (
        <header className="flex h-14 shrink-0 items-center gap-2 border-b border-sat-hairline px-5">
          <Sparkles size={16} className="text-sat-heading" />
          <h2 className="text-sm font-medium text-sat-heading">SAT Query AI</h2>
          <div className="ml-auto flex items-center gap-1.5">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sat-green opacity-75"></span>
              <span className="relative inline-flex h-2 w-2 rounded-full bg-sat-green"></span>
            </span>
            <span className="text-[11px] text-sat-subtitle uppercase tracking-wide">Online</span>
          </div>
        </header>
      )}

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto px-5 py-6">
        <div className="flex flex-col gap-6">
          
          {/* AI Message */}
          <div className="flex flex-col items-start gap-2">
            <div className="flex items-center gap-2">
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-gradient-to-br from-[#f5c40a] via-[#6ac6a0] to-[#22c0cf]">
                <Sparkles size={12} className="text-white" />
              </div>
              <span className="text-xs font-medium text-sat-heading">SAT Query AI</span>
            </div>
            <div className="max-w-[85%] rounded-[14px] rounded-tl-sm bg-sat-bubble-bg px-4 py-3 text-[14px] leading-relaxed text-sat-msg-text">
              <p>
                I've broken down the solution for you in the main workspace. 
                Would you like me to explain any specific step in more detail, or should we try a similar question to practice?
              </p>
            </div>
          </div>

          {/* User Message */}
          <div className="flex flex-col items-end gap-2">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-sat-nav">You</span>
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-sat-hairline">
                <span className="text-xs text-sat-heading">U</span>
              </div>
            </div>
            <div className="max-w-[85%] rounded-[14px] rounded-tr-sm bg-sat-box border border-sat-hairline px-4 py-3 text-[14px] leading-relaxed text-sat-msg-text">
              <p>Could you explain step 2 again? Why did you add 1.5 instead of 3/2?</p>
            </div>
          </div>

          {/* AI Message */}
          <div className="flex flex-col items-start gap-2">
            <div className="flex items-center gap-2">
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-gradient-to-br from-[#f5c40a] via-[#6ac6a0] to-[#22c0cf]">
                <Sparkles size={12} className="text-white" />
              </div>
              <span className="text-xs font-medium text-sat-heading">SAT Query AI</span>
            </div>
            <div className="max-w-[85%] rounded-[14px] rounded-tl-sm bg-sat-bubble-bg px-4 py-3 text-[14px] leading-relaxed text-sat-msg-text">
              <p>
                Great question! <span className="font-mono text-xs">1.5</span> and <span className="font-mono text-xs">3/2</span> are exactly the same value. 
              </p>
              <div className="my-2 space-y-1 text-sm">
                <p>• 3 divided by 2 = 1.5</p>
              </div>
              <p>
                I converted it to a decimal because some students find it easier to work with <span className="font-mono text-xs">12 + 1.5 = 13.5</span> in their head rather than finding common denominators for fractions. Both ways are perfectly correct!
              </p>
            </div>
          </div>

        </div>
      </div>

      {/* Input Area */}
      <div className="shrink-0 p-4">
        <div className="flex items-center gap-2 rounded-full bg-sat-input-bg p-1.5 shadow-[0_2px_12px_rgba(0,0,0,0.5)]">
          <button className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-[#111] hover:bg-black/5">
            <Mic size={18} className="fill-[#111]" />
          </button>
          
          <input 
            type="text" 
            placeholder="Ask anything about this question..."
            className="flex-1 bg-transparent px-2 text-[14px] text-sat-bg placeholder:text-sat-placeholder focus:outline-none"
          />
          
          <button className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-sat-bg text-white hover:opacity-90">
            <ArrowUp size={18} strokeWidth={2.5} />
          </button>
        </div>
        
        <div className="mt-3 flex justify-center gap-2">
          {["Explain this simpler", "Similar question", "Why is A wrong?"].map((suggestion, i) => (
            <button key={i} className="rounded-full border border-sat-hairline bg-sat-box px-3 py-1.5 text-[11px] text-sat-nav transition-colors hover:bg-sat-hairline hover:text-sat-heading">
              {suggestion}
            </button>
          ))}
        </div>
      </div>

    </aside>
  );
}

