"use client";

import { ArrowUp } from "lucide-react";
import { useSatQuery } from "@/lib/store";

export function ChatComposer() {
  const s = useSatQuery();
  if (!s.mission) return null;

  return (
    <div className="border-t border-white/10 p-3">
      {s.intent && (
        <p className="mb-2 text-[11px] text-white/45">
          <span className="text-[#00b4ff]">{s.intent.label}</span>
          <span className="mx-2">·</span>
          {s.intent.specialists.join(" → ")}
          {s.intent.warning && <span className="ml-2 text-change">{s.intent.warning}</span>}
        </p>
      )}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          s.submit();
        }}
        className="flex items-end gap-2 rounded-2xl border border-white/15 bg-white/5 px-3 py-2"
      >
        <textarea
          id="SatQuery-input"
          value={s.query}
          onChange={(e) => s.setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              s.submit();
            }
          }}
          rows={2}
          placeholder="Ask a follow-up, or a new question about this scene…"
          className="max-h-28 flex-1 resize-none bg-transparent text-[14px] leading-relaxed text-white outline-none placeholder:text-white/35"
        />
        <button
          type="submit"
          disabled={s.running || !s.query.trim()}
          className="mb-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-[#00b4ff] text-black disabled:opacity-30"
          title="Send"
        >
          <ArrowUp size={16} strokeWidth={2.4} />
        </button>
      </form>
    </div>
  );
}


