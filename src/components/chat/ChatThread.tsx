"use client";

import { useEffect, useRef } from "react";
import { useSatQuery } from "@/lib/store";

export function ChatThread() {
  const s = useSatQuery();
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth" });
  }, [s.thread, s.running, s.steps, s.result]);

  if (!s.mission) return null;

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-y-auto px-4 py-4">
      <p className="mb-4 text-[11px] tracking-[0.2em] text-white/35 uppercase">
        Conversation · {s.mission.title}
      </p>

      {s.thread.length === 0 && !s.running && (
        <div className="mt-6">
          <p className="text-[15px] text-white/70">
            Ask this scene in plain English. The agent picks the specialist.
          </p>
          <div className="mt-4 flex flex-col gap-2">
            {s.mission.suggested.map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => s.submit(q)}
                className="rounded-2xl border border-white/10 bg-white/4 px-4 py-3 text-left text-[13px] leading-relaxed text-white/80 hover:border-[#00b4ff]/50 hover:text-white"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      <ol className="space-y-4">
        {s.thread.map((item) =>
          item.role === "user" ? (
            <li key={item.id} className="flex justify-end">
              <div className="max-w-[92%] rounded-2xl rounded-br-sm bg-[#00b4ff] px-4 py-2.5 text-[14px] leading-relaxed text-black">
                {item.text}
              </div>
            </li>
          ) : (
            <li key={item.id} className="flex justify-start">
              <div className="max-w-[92%] rounded-2xl rounded-bl-sm border border-white/10 bg-white/5 px-4 py-3">
                <p className="text-[10px] tracking-[0.2em] text-[#00b4ff] uppercase">
                  SatQuery · {item.result?.task ?? "reply"}
                </p>
                <p className="mt-2 text-[14px] leading-relaxed text-white/90">{item.text}</p>
                {item.result && (
                  <>
                    <dl className="mt-3 grid grid-cols-2 gap-2">
                      {item.result.metrics.slice(0, 4).map((m) => (
                        <div key={m.label} className="rounded-lg bg-black/40 px-2.5 py-2">
                          <dt className="text-[10px] text-white/40">{m.label}</dt>
                          <dd className="font-mono text-[13px]">{m.value}</dd>
                        </div>
                      ))}
                    </dl>
                    <p className="mt-2 font-mono text-[10px] text-white/40">
                      {(item.result.confidence * 100).toFixed(1)}% confidence · evidence on the scene
                    </p>
                  </>
                )}
              </div>
            </li>
          )
        )}
      </ol>

      {s.running && (
        <div className="mt-4 max-w-[92%] rounded-2xl rounded-bl-sm border border-[#00b4ff]/30 bg-[#00b4ff]/8 px-4 py-3">
          <p className="text-[10px] tracking-[0.2em] text-[#00b4ff] uppercase">Agent routing</p>
          <ol className="mt-2 space-y-1.5">
            {s.steps.map((st) => (
              <li key={st.id} className="flex items-center gap-2 text-[12px]">
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    st.status === "done"
                      ? "bg-ok"
                      : st.status === "running"
                        ? "bg-[#00b4ff] live-dot"
                        : "bg-white/20"
                  }`}
                />
                <span className={st.status === "pending" ? "text-white/35" : "text-white/85"}>
                  {st.label}
                </span>
              </li>
            ))}
          </ol>
        </div>
      )}

      <div ref={end} />
    </div>
  );
}


