"use client";

import { useSatQuery } from "@/lib/store";

export function Report() {
  const s = useSatQuery();
  if (!s.reportOpen || !s.mission) return null;
  const r = s.result;
  const primary = s.mission.assets[0];

  return (
    <div className="absolute inset-0 z-40 overflow-auto bg-void/90">
      <article className="mx-auto my-8 w-[min(760px,94vw)] rounded-lg bg-[#f3eee6] px-10 py-9 text-[#1c1914] print:my-0 print:w-auto print:rounded-none">
        <header className="flex items-start justify-between border-b border-black/10 pb-3">
          <div>
            <p className="text-[12px] tracking-[0.14em] text-black/50 uppercase">SatQuery · SIH26167 · ISRO</p>
            <h1 className="mt-0.5 text-2xl font-medium">Grounded scene report</h1>
          </div>
          <p className="font-mono text-[11px] text-black/45">{s.mission.code}</p>
        </header>
        <p className="mt-5 text-xl font-medium">{s.mission.title}</p>
        <p className="mt-1 text-sm text-black/55">
          {s.mission.location} · {primary.coords} · {primary.gsd}
        </p>

        <section className="mt-5 grid grid-cols-2 gap-3 text-sm">
          {s.mission.assets.map((a) => (
            <div key={a.id} className="rounded-md border border-black/10 p-3">
              <p className="text-[11px] text-black/45">{a.role}</p>
              <p className="mt-0.5 font-medium">{a.name}</p>
              <p className="text-black/55">
                {a.sensor} · {a.date} · {a.format}
              </p>
            </div>
          ))}
        </section>

        {r && (
          <>
            <h2 className="mt-7 text-lg font-medium">{r.title}</h2>
            <p className="mt-2 text-[15px] leading-relaxed">{r.answer}</p>
            <table className="mt-4 w-full text-sm">
              <tbody>
                {r.metrics.map((m) => (
                  <tr key={m.label} className="border-b border-black/8">
                    <td className="py-2">{m.label}</td>
                    <td className="py-2 text-right font-medium">{m.value}</td>
                  </tr>
                ))}
                <tr>
                  <td className="py-2">Confidence</td>
                  <td className="py-2 text-right font-medium">{(r.confidence * 100).toFixed(1)}%</td>
                </tr>
              </tbody>
            </table>
            <h3 className="mt-6 text-[15px] font-medium">Notes</h3>
            <ol className="mt-2 list-decimal pl-5 text-sm leading-relaxed">
              {r.observations.map((o) => (
                <li key={o} className="mb-1">{o}</li>
              ))}
            </ol>
            <h3 className="mt-6 text-[15px] font-medium">Agent</h3>
            <ol className="mt-2 text-sm">
              {s.steps.map((st) => (
                <li key={st.id} className="flex justify-between gap-4 border-b border-black/8 py-1">
                  <span>{st.label}</span>
                  <span className="text-right text-black/50">{st.detail}</span>
                </li>
              ))}
            </ol>
            <h3 className="mt-6 text-[15px] font-medium">Models</h3>
            <ul className="mt-2 text-sm">
              {r.models.map((m) => (
                <li key={m.name}>
                  {m.name} — {m.role}
                </li>
              ))}
            </ul>
          </>
        )}

        <p className="mt-8 text-[11px] text-black/40">
          SIH26167 demo UI · not an official ISRO product
        </p>
      </article>
      <div className="no-print sticky bottom-0 flex justify-center gap-3 bg-void/90 py-3">
        <button
          type="button"
          onClick={() => window.print()}
          className="rounded-md bg-brass px-4 py-2 text-sm font-medium text-void"
        >
          Print / PDF
        </button>
        <button
          type="button"
          onClick={() => s.setReportOpen(false)}
          className="rounded-md px-4 py-2 text-sm text-mute"
        >
          Close
        </button>
      </div>
    </div>
  );
}
