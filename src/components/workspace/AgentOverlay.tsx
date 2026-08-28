"use client";

import { useSatQuery } from "@/lib/store";

export function AgentOverlay() {
  const { running, steps, auditOpen, setAuditOpen, result } = useSatQuery();
  if (!running && !steps.length) return null;

  return (
    <>
      {running && (
        <div className="pointer-events-none absolute inset-x-0 top-10 z-20 flex justify-center px-4">
          <ol className="flex max-w-3xl flex-wrap items-center justify-center gap-1 rounded-md bg-void/75 px-2 py-1.5 backdrop-blur-sm">
            {steps.map((st, i) => (
              <li key={st.id} className="flex items-center gap-1">
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    st.status === "done"
                      ? "bg-ok"
                      : st.status === "running"
                        ? "bg-signal live-dot"
                        : "bg-white/20"
                  }`}
                />
                <span
                  className={`text-[11px] ${
                    st.status === "pending" ? "text-faint" : "text-ink"
                  }`}
                >
                  {st.label}
                </span>
                {i < steps.length - 1 && <span className="mx-1 text-white/15">/</span>}
              </li>
            ))}
          </ol>
        </div>
      )}

      <div className="absolute top-10 right-3 z-20 w-[min(280px,38vw)]">
        <button
          type="button"
          onClick={() => setAuditOpen(!auditOpen)}
          className="mb-1 text-[11px] text-brass"
        >
          {auditOpen ? "hide routing" : "agentic routing"}
        </button>
        {auditOpen && (
          <ol className="rounded-md bg-void/85 p-2.5 text-[12px] backdrop-blur-sm ring-1 ring-white/8">
            {steps.map((st) => (
              <li key={st.id} className="flex gap-2 py-1">
                <span
                  className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${
                    st.status === "done"
                      ? "bg-ok"
                      : st.status === "running"
                        ? "bg-signal live-dot"
                        : "bg-faint"
                  }`}
                />
                <div>
                  <p className="text-ink">{st.label}</p>
                  <p className="text-[11px] leading-snug text-mute">{st.detail}</p>
                </div>
              </li>
            ))}
            {result && (
              <li className="mt-1 border-t border-white/10 pt-1.5 font-mono text-[10px] text-mute">
                {result.task} · {(result.confidence * 100).toFixed(1)}%
              </li>
            )}
          </ol>
        )}
      </div>
    </>
  );
}


