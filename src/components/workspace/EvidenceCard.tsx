"use client";

import { useSatQuery } from "@/lib/store";

export function EvidenceCard() {
  const s = useSatQuery();
  if (!s.result) return null;
  const r = s.result;
  const selected = r.boxes.find((b) => b.id === s.selectedId);

  return (
    <aside className="flex h-full w-full flex-col justify-between overflow-y-auto bg-panel">
      <div className="p-4">
        <p className="text-[11px] tracking-[0.16em] text-brass uppercase">Grounded answer</p>
        <h2 className="mt-0.5 text-xl leading-snug font-medium">{r.title}</h2>
        <div className="mt-2 flex items-center gap-2">
          <div className="h-1 flex-1 overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full rounded-full bg-brass"
              style={{ width: `${Math.round(r.confidence * 100)}%` }}
            />
          </div>
          <span className="font-mono text-[11px] tabular-nums text-mute">
            {(r.confidence * 100).toFixed(1)}%
          </span>
        </div>
        <p className="mt-3 text-[13px] leading-relaxed text-ink/90">{r.answer}</p>

        <dl className="mt-4 grid grid-cols-2 gap-2">
          {r.metrics.map((m) => (
            <div key={m.label} className="rounded-md bg-void-2 px-2.5 py-2">
              <dt className="text-[10px] text-mute">{m.label}</dt>
              <dd className="font-mono text-[15px] tabular-nums">{m.value}</dd>
              {m.hint && <p className="text-[10px] text-faint">{m.hint}</p>}
            </div>
          ))}
        </dl>

        <p className="mt-4 text-[11px] text-mute">Notes</p>
        <ul className="mt-1.5 space-y-1.5">
          {r.observations.map((o) => (
            <li key={o} className="text-[12px] leading-relaxed text-mute">
              {o}
            </li>
          ))}
        </ul>

        {selected && (
          <div className="mt-3 rounded-md bg-signal/10 px-2.5 py-2 ring-1 ring-signal/25">
            <p className="text-[12px] text-signal">{selected.label}</p>
            <p className="mt-0.5 text-[11px] text-mute">
              {(selected.score * 100).toFixed(0)}% · {selected.kind ?? "object"}
            </p>
          </div>
        )}

        {r.layers.length > 0 && (
          <>
            <p className="mt-4 text-[11px] text-mute">Layers</p>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {r.layers.map((l) => (
                <button
                  key={l.id}
                  type="button"
                  onClick={() => s.toggleLayer(l.id)}
                  className="flex items-center gap-1.5 rounded-full px-2 py-1 text-[11px] ring-1"
                  style={{
                    borderColor: "transparent",
                    boxShadow: `inset 0 0 0 1px ${l.active ? l.color : "rgba(255,255,255,0.1)"}`,
                    color: l.active ? l.color : "var(--mute)",
                  }}
                >
                  <span className="h-1.5 w-1.5 rounded-full" style={{ background: l.color }} />
                  {l.label}
                </button>
              ))}
            </div>
          </>
        )}

        <p className="mt-4 text-[11px] text-mute">Specialists selected</p>
        <ul className="mt-1.5 space-y-1">
          {r.models.map((m) => (
            <li key={m.name} className="flex justify-between text-[11px] text-mute">
              <span className="text-ink">{m.name}</span>
              <span>{m.status}</span>
            </li>
          ))}
        </ul>
      </div>
      <div className="grid grid-cols-2 gap-px border-t border-white/8">
        <button
          type="button"
          onClick={() => s.setReportOpen(true)}
          className="py-2.5 text-[12px] text-brass"
        >
          Report
        </button>
        <button
          type="button"
          onClick={() => downloadJson(s)}
          className="py-2.5 text-[12px] text-mute"
        >
          JSON
        </button>
      </div>
    </aside>
  );
}

function downloadJson(s: ReturnType<typeof useSatQuery>) {
  if (!s.result || !s.mission) return;
  const blob = new Blob(
    [
      JSON.stringify(
        {
          mission: s.mission.title,
          task: s.result.task,
          answer: s.result.answer,
          metrics: s.result.metrics,
          confidence: s.result.confidence,
          models: s.result.models,
          trace: s.steps,
        },
        null,
        2
      ),
    ],
    { type: "application/json" }
  );
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `satquery-${s.mission.id}.json`;
  a.click();
}
