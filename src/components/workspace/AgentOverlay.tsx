"use client";

import { useSatQuery } from "@/lib/store";

export function AgentOverlay() {
  const { running, steps, auditOpen, setAuditOpen, result } = useSatQuery();
  if (!running && !steps.length) return null;

  return (
    <div className="absolute top-4 right-4 z-20 w-[min(320px,45vw)]">
      <div className="overflow-hidden rounded-xl border border-sat-hairline bg-[#0c0c0c] p-3 shadow-2xl">
        {/* Live reasoning header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span
              className="h-2 w-2 rounded-full"
              style={{
                background: "#23d92c",
                boxShadow: "0 0 6px rgba(45,220,55,.65)",
              }}
            />
            <span className="text-[12px] font-medium text-sat-heading">
              Live reasoning
            </span>
          </div>
          <button
            type="button"
            onClick={() => setAuditOpen(!auditOpen)}
            className="text-[11px] text-sat-subtitle hover:text-sat-heading transition-colors"
          >
            {auditOpen ? "hide" : "show"}
          </button>
        </div>

        {/* Steps list */}
        {(auditOpen || running) && (
          <ol className="mt-3 space-y-2">
            {steps.map((st) => (
              <li key={st.id} className="flex items-start gap-2">
                <span
                  className={`mt-[5px] h-1.5 w-1.5 shrink-0 rounded-full ${
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
                <div>
                  <p className="text-[12px] text-sat-msg-text">{st.label}</p>
                  <p className="text-[11px] leading-snug text-sat-subtitle">{st.detail}</p>
                </div>
              </li>
            ))}
            {result && (
              <li className="mt-1 border-t border-sat-hairline pt-2 font-mono text-[10px] text-sat-subtitle">
                {result.task} · {(result.confidence * 100).toFixed(1)}% confidence
              </li>
            )}
          </ol>
        )}
      </div>
    </div>
  );
}
