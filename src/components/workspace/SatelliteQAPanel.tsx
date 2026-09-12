"use client";

import { useEffect, useRef } from "react";
import { ArrowUp, Activity, FileSearch, Layers, Paperclip, MessageSquare } from "lucide-react";
import { useSatQuery } from "@/lib/store";

const DEMO_QUERIES = [
  "Which settlements are affected by water-covered areas?",
  "Has built-up area increased between the two dates?",
  "Highlight the ships and bulk tanks in this harbour.",
  "Describe the land-cover and major objects visible in this image.",
];

export function SatelliteQAPanel({ isMobile, isFullWidth }: { isMobile?: boolean; isFullWidth?: boolean }) {
  const s = useSatQuery();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [s.thread, s.running, s.result]);

  const send = (text?: string) => {
    const q = (text ?? s.query).trim();
    if (!q || s.running) return;
    s.submit(q);
  };

  return (
    <aside
      className={`flex h-full w-full flex-col bg-sat-panel-bg ${
        isMobile || isFullWidth ? "" : "border-l border-sat-hairline md:w-[400px] lg:w-[450px]"
      }`}
    >
      {!isMobile && !isFullWidth && (
        <header className="flex h-14 shrink-0 items-center gap-2 border-b border-sat-hairline px-5">
          <h2 className="text-sm font-medium text-sat-heading">Analysis & Queries</h2>
        </header>
      )}

      <div className="flex-1 overflow-y-auto px-5 py-6">
        <div className="flex flex-col gap-6">
          <div className="rounded-xl border border-sat-hairline bg-[#1c1c1c]/50 p-4 shadow-sm">
            <h3 className="mb-2 text-[12px] font-bold tracking-wide text-sat-heading uppercase">
              Accepted format
            </h3>
            <ul className="space-y-1.5 text-[12px] text-sat-subtitle">
              <li>
                <strong>GeoTIFF</strong> / <strong>TIFF</strong> — supported
              </li>
              <li>
                <strong>PNG</strong> / <strong>JPEG</strong> — prescribed public benchmark datasets only
              </li>
            </ul>
          </div>

          {!s.thread.length && !s.result && !s.running ? (
            <div className="flex flex-col text-sat-nav">
              <div className="mt-4 flex flex-col items-center text-center">
                <MessageSquare size={32} className="mb-3 opacity-20" />
                <p className="text-[13px] text-sat-heading">Ask in plain English.</p>
                <p className="mt-1 text-[12px] opacity-70">
                  Upload a scene first. The backend will route your question to the specialist models.
                </p>
              </div>
              <div className="mt-6 flex flex-col gap-2">
                {DEMO_QUERIES.map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => s.setQuery(q)}
                    className="rounded-lg border border-sat-hairline bg-sat-box px-3 py-2.5 text-left text-[13px] leading-relaxed text-sat-msg-text transition-colors hover:bg-sat-hairline"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {s.thread.map((msg) => (
                <div key={msg.id} className="rounded-xl border border-sat-hairline bg-sat-box p-5 shadow-sm">
                  {msg.role === "user" ? (
                    <>
                      <h3 className="mb-4 flex items-center gap-2 text-[14px] font-semibold text-sat-heading">
                        <FileSearch size={16} className="text-sat-nav" />
                        Target Query
                      </h3>
                      <p className="text-[14px] leading-relaxed text-sat-msg-text">&ldquo;{msg.text}&rdquo;</p>
                    </>
                  ) : (
                    <>
                      <h3 className="mb-4 flex items-center gap-2 text-[14px] font-semibold text-sat-heading">
                        <Activity size={16} className="text-sat-nav" />
                        Extraction Results
                      </h3>
                      <p className="mb-4 text-[14px] leading-relaxed text-sat-msg-text">{msg.text}</p>
                      {msg.result && (
                        <>
                          <div className="space-y-4">
                            {msg.result.metrics?.map((m, i) => (
                              <div
                                key={`${m.label}-${i}`}
                                className="flex items-center justify-between border-b border-sat-hairline pb-2"
                              >
                                <span className="text-[13px] text-sat-nav">{m.label}</span>
                                <span className="text-[13px] font-medium text-sat-heading">{m.value}</span>
                              </div>
                            ))}
                            {msg.result.confidence ? (
                              <div className="flex items-center justify-between border-b border-sat-hairline pb-2">
                                <span className="text-[13px] text-sat-nav">Confidence Score</span>
                                <span className="text-[13px] font-medium text-sat-green">
                                  {(msg.result.confidence * 100).toFixed(1)}%
                                </span>
                              </div>
                            ) : null}
                          </div>
                          {msg.result.observations?.length ? (
                            <div className="mt-5 text-[13px] leading-relaxed text-sat-subtitle">
                              {msg.result.observations.map((obs, i) => (
                                <p key={i} className="mb-2">
                                  • {obs}
                                </p>
                              ))}
                            </div>
                          ) : null}
                        </>
                      )}
                    </>
                  )}
                </div>
              ))}

              {s.running && (
                <div className="overflow-hidden rounded-xl border border-sat-hairline bg-[#0c0c0c] p-4">
                  <div className="flex items-center gap-2">
                    <span
                      className="h-2 w-2 rounded-full"
                      style={{
                        background: "#23d92c",
                        boxShadow: "0 0 6px rgba(45,220,55,.65)",
                      }}
                    />
                    <span className="text-[12px] font-semibold tracking-[0.15em] text-sat-heading uppercase">
                      Agent routing
                    </span>
                  </div>
                  {s.steps.length ? (
                    <ol className="mt-3 space-y-2">
                      {s.steps.map((st) => (
                        <li key={st.id} className="flex items-center gap-2 text-[12px]">
                          <span
                            className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                              st.status === "running" ? "live-dot" : ""
                            }`}
                            style={{
                              background:
                                st.status === "done"
                                  ? "#23d92c"
                                  : st.status === "running"
                                    ? "#f5c40a"
                                    : "#6b6b6d",
                            }}
                          />
                          <span
                            className={st.status === "pending" ? "text-sat-subtitle opacity-40" : "text-sat-msg-text"}
                          >
                            {st.label}
                          </span>
                        </li>
                      ))}
                    </ol>
                  ) : (
                    <p className="mt-2 text-[13px] text-sat-subtitle">Reading the question…</p>
                  )}
                </div>
              )}

              {s.result && s.mission && (
                <div className="rounded-xl border border-sat-hairline bg-sat-box p-5 shadow-sm">
                  <h3 className="mb-4 flex items-center gap-2 text-[14px] font-semibold text-sat-heading">
                    <Layers size={16} className="text-sat-nav" />
                    Suggested follow-ups
                  </h3>
                  <div className="flex flex-col gap-2">
                    {s.mission.suggested.map((q) => (
                      <button
                        key={q}
                        type="button"
                        disabled={s.running}
                        onClick={() => s.setQuery(q)}
                        className="flex w-full items-center justify-between rounded-lg bg-[#1c1c1c] px-3 py-2 text-left transition-colors hover:bg-sat-hairline disabled:opacity-40"
                      >
                        <span className="text-[13px] text-sat-msg-text">{q}</span>
                        <ArrowUp size={14} className="rotate-45 text-sat-nav" />
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
          <div ref={endRef} />
        </div>
      </div>

      <div className="shrink-0 border-t border-sat-hairline p-4">
        <form
          className="flex items-center gap-2 rounded-lg bg-sat-input-bg px-2 py-1.5 shadow-[0_2px_12px_rgba(0,0,0,0.5)]"
          onSubmit={(e) => {
            e.preventDefault();
            send();
          }}
        >
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
            id="SatQuery-input"
            type="text"
            value={s.query}
            onChange={(e) => s.setQuery(e.target.value)}
            placeholder="Ask about a satellite scene..."
            disabled={s.running}
            className="flex-1 bg-transparent px-2 text-[14px] text-sat-bg placeholder:text-[#6b6b6d] focus:outline-none disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={s.running || !s.query.trim()}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-sat-bg text-white hover:opacity-90 disabled:opacity-30"
          >
            <ArrowUp size={16} strokeWidth={2.5} />
          </button>
        </form>
      </div>
    </aside>
  );
}
