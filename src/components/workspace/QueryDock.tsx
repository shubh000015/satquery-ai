"use client";

import { ArrowUp } from "lucide-react";
import { useSatQuery } from "@/lib/store";

export function QueryDock() {
  const s = useSatQuery();
  if (!s.mission) return null;

  return (
    <div className="border-t border-white/8 bg-panel px-3 py-2.5">
      {s.intent && (
        <div className="mb-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]">
          <span className="text-brass">{s.intent.label}</span>
          <span className="text-mute">{s.intent.specialists.join(" → ")}</span>
          {s.intent.warning && <span className="text-change">{s.intent.warning}</span>}
        </div>
      )}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          s.submit();
        }}
        className="flex items-end gap-2"
      >
        <label className="flex-1">
          <span className="mb-1 block text-[10px] tracking-[0.18em] text-mute uppercase">
            Natural-language query
          </span>
          <textarea
            value={s.query}
            onChange={(e) => s.setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                s.submit();
              }
            }}
            rows={2}
            placeholder={
              s.mission.mode === "bi-temporal"
                ? "What changed between these two dates, and where?"
                : s.mission.mode === "cross-modal"
                  ? "Use optical and SAR together — flooded settlements, water, built-up…"
                  : "Describe the scene, count objects, or highlight something…"
            }
            className="w-full resize-none bg-transparent text-[14px] leading-relaxed text-ink outline-none placeholder:text-faint"
            id="satquery-input"
          />
        </label>
        <button
          type="submit"
          disabled={s.running || !s.query.trim()}
          className="mb-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-brass text-void disabled:opacity-35"
          title="Ask"
        >
          <ArrowUp size={16} strokeWidth={2.4} />
        </button>
      </form>
      <div className="mt-1.5 flex flex-wrap gap-1.5">
        {s.mission.suggested.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => s.submit(q)}
            className="max-w-full truncate rounded-full px-2.5 py-1 text-left text-[11px] text-mute ring-1 ring-white/10 hover:text-ink hover:ring-white/25"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
