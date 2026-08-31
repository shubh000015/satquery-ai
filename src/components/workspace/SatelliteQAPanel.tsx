import { useRef } from "react";
import { ArrowUp, Target, Activity, FileSearch, Layers, Paperclip, MessageSquare } from "lucide-react";
import { useSatQuery } from "@/lib/store";

export function SatelliteQAPanel({ isMobile, isFullWidth }: { isMobile?: boolean, isFullWidth?: boolean }) {
  const s = useSatQuery();
  const fileInputRef = useRef<HTMLInputElement>(null);

  return (
    <aside className={`flex h-full w-full flex-col bg-sat-panel-bg ${isMobile || isFullWidth ? "" : "border-l border-sat-hairline md:w-[400px] lg:w-[450px]"}`}>
      
      {/* Header */}
      {!isMobile && !isFullWidth && (
        <header className="flex h-14 shrink-0 items-center gap-2 border-b border-sat-hairline px-5">
          <h2 className="text-sm font-medium text-sat-heading">Analysis & Queries</h2>
        </header>
      )}

      {/* Analysis Content */}
      <div className="flex-1 overflow-y-auto px-5 py-6">
        <div className="flex flex-col gap-6">
          
          <div className="rounded-xl border border-sat-hairline bg-[#1c1c1c]/50 p-4 shadow-sm">
            <h3 className="mb-2 text-[12px] font-bold tracking-wide text-sat-heading uppercase">
              Accepted format
            </h3>
            <ul className="space-y-1.5 text-[12px] text-sat-subtitle">
              <li className="flex items-center gap-1.5">
                <span><strong>GeoTIFF</strong> / <strong>TIFF</strong> — supported</span>
              </li>
              <li className="flex items-start gap-1.5">
                <span><strong>PNG</strong> / <strong>JPEG</strong> — prescribed public benchmark datasets only</span>
              </li>
            </ul>
          </div>

          {!s.thread.length && !s.result ? (
            <div className="mt-10 flex flex-col items-center justify-center text-sat-nav">
              <MessageSquare size={32} className="mb-3 opacity-20" />
              <p className="text-[13px]">No chat history.</p>
              <p className="text-[12px] opacity-70">Upload an image to begin.</p>
            </div>
          ) : (
            <>
              {s.thread.map((msg: any) => (
                <div key={msg.id} className="rounded-xl border border-sat-hairline bg-sat-box p-5 shadow-sm">
                  {msg.role === "user" ? (
                    <>
                      <h3 className="mb-4 flex items-center gap-2 text-[14px] font-semibold text-sat-heading">
                        <FileSearch size={16} className="text-sat-nav" />
                        Target Query
                      </h3>
                      <p className="text-[14px] leading-relaxed text-sat-msg-text">
                        "{msg.text}"
                      </p>
                    </>
                  ) : (
                    <>
                      <h3 className="mb-4 flex items-center gap-2 text-[14px] font-semibold text-sat-heading">
                        <Activity size={16} className="text-sat-nav" />
                        Extraction Results
                      </h3>
                      
                      <p className="text-[14px] leading-relaxed text-sat-msg-text mb-4">
                        {msg.text}
                      </p>

                      {msg.result && (
                        <>
                          <div className="space-y-4">
                            {msg.result.metrics?.map((m: any, i: number) => (
                              <div key={i} className="flex items-center justify-between border-b border-sat-hairline pb-2">
                                <span className="text-[13px] text-sat-nav">{m.label}</span>
                                <span className="text-[13px] font-medium text-sat-heading">{m.value}</span>
                              </div>
                            ))}
                            {msg.result.confidence && (
                              <div className="flex items-center justify-between border-b border-sat-hairline pb-2">
                                <span className="text-[13px] text-sat-nav">Confidence Score</span>
                                <span className="text-[13px] font-medium text-sat-heading">{(msg.result.confidence * 100).toFixed(1)}%</span>
                              </div>
                            )}
                          </div>
                          
                          {msg.result.observations && (
                            <div className="mt-5 text-[13px] leading-relaxed text-sat-subtitle">
                              {msg.result.observations.map((obs: any, i: number) => (
                                <p key={i} className="mb-2">• {obs}</p>
                              ))}
                            </div>
                          )}
                        </>
                      )}
                    </>
                  )}
                </div>
              ))}

              {s.result && (
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
              )}
            </>
          )}

        </div>
      </div>

      {/* Input Area */}
      <div className="shrink-0 p-4 border-t border-sat-hairline">
        <div className="flex items-center gap-2 rounded-lg bg-sat-input-bg px-2 py-1.5 shadow-[0_2px_12px_rgba(0,0,0,0.5)]">
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept=".tif,.tiff,.png,.jpg,.jpeg,.webp,image/*"
            multiple
            onChange={(e) => {
              if (e.target.files?.length) s.ingestFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <button 
            type="button" 
            title="Attach images"
            onClick={() => fileInputRef.current?.click()}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sat-nav hover:bg-sat-hairline"
          >
            <Paperclip size={18} />
          </button>
          <input 
            type="text" 
            placeholder="Ask about this satellite scene..."
            className="flex-1 bg-transparent px-2 text-[14px] text-sat-bg placeholder:text-[#6b6b6d] focus:outline-none"
            onKeyDown={(e) => {
              if (e.key === "Enter" && e.currentTarget.value.trim()) {
                s.submit(e.currentTarget.value);
                e.currentTarget.value = "";
              }
            }}
          />
          <button 
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-sat-bg text-white hover:opacity-90"
            onClick={() => {
              const input = document.querySelector('input[type="text"]') as HTMLInputElement;
              if (input?.value.trim()) {
                s.submit(input.value);
                input.value = "";
              }
            }}
          >
            <ArrowUp size={16} strokeWidth={2.5} />
          </button>
        </div>
      </div>

    </aside>
  );
}

