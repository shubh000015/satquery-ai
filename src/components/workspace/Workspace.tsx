"use client";

import { useRef } from "react";
import { Download, FolderOpen, Ruler } from "lucide-react";
import { ChatComposer } from "@/components/chat/ChatComposer";
import { ChatThread } from "@/components/chat/ChatThread";
import { EvidenceCard } from "@/components/workspace/EvidenceCard";
import { ImageryStage } from "@/components/workspace/ImageryStage";
import { Report } from "@/components/workspace/Report";
import { useSatQuery } from "@/lib/store";

export function Workspace() {
  const s = useSatQuery();
  const fileRef = useRef<HTMLInputElement>(null);
  if (!s.mission) return null;
  const m = s.mission;

  return (
    <div className="flex h-full min-h-0 flex-col bg-black">
      <header className="flex items-center gap-3 border-b border-white/10 px-3 py-2.5">
        <button
          type="button"
          onClick={s.goIngress}
          className="font-display px-1 text-[13px] tracking-[0.22em]"
        >
          SATQUERY
        </button>
        <span className="hidden text-[10px] tracking-[0.16em] text-[#00b4ff] uppercase sm:inline">
          SIH26167 · Chat
        </span>
        <span className="hidden h-4 w-px bg-white/10 sm:block" />
        <div className="min-w-0">
          <p className="truncate text-[13px]">{m.title}</p>
          <p className="truncate font-mono text-[10px] text-white/40">
            {m.location} ·{" "}
            {m.mode === "cross-modal"
              ? "optical + SAR"
              : m.mode === "bi-temporal"
                ? "bi-temporal"
                : "single optical"}
          </p>
        </div>
        <div className="ml-auto flex items-center gap-1">
          <button
            type="button"
            title="Measure"
            onClick={() => s.setMeasuring(!s.measuring)}
            className={`rounded-md p-2 ${s.measuring ? "bg-[#00b4ff]/15 text-[#00b4ff]" : "text-white/45 hover:text-white"}`}
          >
            <Ruler size={15} />
          </button>
          <button
            type="button"
            title="Load files"
            onClick={() => fileRef.current?.click()}
            className="rounded-md p-2 text-white/45 hover:text-white"
          >
            <FolderOpen size={15} />
          </button>
          <button
            type="button"
            title="Export report"
            onClick={() => s.setReportOpen(true)}
            className="rounded-md p-2 text-white/45 hover:text-white"
          >
            <Download size={15} />
          </button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <section className="flex w-full min-w-0 flex-col border-white/10 md:w-[42%] md:max-w-[480px] md:border-r">
          <ChatThread />
          <ChatComposer />
        </section>
        <section className="hidden min-w-0 flex-1 flex-col md:flex">
          <ImageryStage />
          {s.result && (
            <div className="max-h-[38%] overflow-y-auto border-t border-white/10">
              <EvidenceCard />
            </div>
          )}
        </section>
      </div>

      <div className="max-h-[32vh] md:hidden">{s.result && <EvidenceCard />}</div>

      <Report />
      <input
        ref={fileRef}
        type="file"
        accept=".tif,.tiff,.png,.jpg,.jpeg,.webp,image/*"
        multiple
        className="hidden"
        onChange={(e) => {
          if (e.target.files?.length) s.ingestFiles(e.target.files);
        }}
      />
    </div>
  );
}
